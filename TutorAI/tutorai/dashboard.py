from typing import Optional
import gradio as gr

from .utils import process_book_url, download_pdf
from .vector_index import ChunkingConfig, TutorAIVectorIndex
from .llm import QueryIntent, classify_intent, LLMConfig
from .llm import LLMManager
from .llm import TutorAIConversationPipeline, GuardrailConfig
from .utils import read_pdf_pages, stable_pdf_id
import traceback
from typing import Any, Dict

class TutorAIDashboard:
    """
    Gradio dashboard wrapper with:
      - Chatbot
      - Text input
      - Model dropdown
      - Context (top-k) slider
      - Temperature slider
      - Similarity (max distance) slider
      - One-button initializer that:
          * downloads the PDF
          * reads pages
          * builds FAISS index
          * sets up the conversation pipeline
    """

    def __init__(
        self,
        default_book_url: str,
        default_model_key: Optional[str] = None,
        chunk_cfg: Optional[ChunkingConfig] = None,
        guard_mode: str = "append_once",  # or "per_paragraph"
        device_embeddings: Optional[str] = None,  # "cuda" for GPU embeddings
    ):
        self.default_book_url = default_book_url
        self.default_model_key = default_model_key  # if None, we'll use the first listed model
        self.chunk_cfg = chunk_cfg or ChunkingConfig(chunk_size=900, chunk_overlap=150)
        self.guard_mode = guard_mode
        self.device_embeddings = device_embeddings

        # Runtime objects created during initialization
        self.indexer: Optional[TutorAIVectorIndex] = None
        self.conv: Optional[TutorAIConversationPipeline] = None

        # Presentation strings
        self.default_system = "TutorAI for employees learning from 'Storytelling with Data'."
        self.default_guidelines = (
            "Answer ONLY from the provided context. Include (p. N) citation(s). "
            "If the information is not in the context, refuse."
        )

        # LLM catalog from LLMManager (so we can fill the dropdown even before init)
        self.llm_manager = LLMManager()
        self.model_choices = list(self.llm_manager.catalog.keys())
        if self.default_model_key is None and self.model_choices:
            self.default_model_key = self.model_choices[0]

    # ------------------ Initialization orchestration ------------------
    def initialize_and_build(
        self,
        book_url: str,
        embedding_model: str,
        llm_key: str,
        progress=gr.Progress(track_tqdm=False),
    ) -> str:
        """
        Runs the end-to-end setup:
          1) Download & read PDF
          2) Chunk with LangChain splitter
          3) Build FAISS index
          4) Create conversation pipeline with guardrails
        Returns a short status string for the UI.
        """
        try:
            progress(0, desc="Processing book URL")
            processed_url = process_book_url(book_url.strip() or self.default_book_url)

            progress(0.15, desc="Downloading PDF")
            pdf_bytes, saved_path = download_pdf(processed_url, save_path="data/storytelling_with_data.pdf")
            pdf_id = stable_pdf_id(pdf_bytes)

            progress(0.35, desc="Reading pages from PDF")
            pages = read_pdf_pages(pdf_bytes)
            if not pages:
                return "No text could be extracted from the PDF. Consider adding OCR."

            progress(0.5, desc="Building indexer and chunking")
            self.indexer = TutorAIVectorIndex(
                chunk_cfg=self.chunk_cfg,
                embedding_model=embedding_model,
                device=self.device_embeddings,           # "cuda" if you want GPU embeddings
            )
            chunk_docs = self.indexer.make_chunks(pages, source_id=pdf_id)

            progress(0.7, desc="Constructing FAISS index")
            self.indexer.build_index(chunk_docs)

            progress(0.85, desc=f"Loading model: {llm_key}")
            # Build conversation pipeline with guardrails
            guard_cfg = GuardrailConfig(mode=self.guard_mode)
            self.conv = TutorAIConversationPipeline(
                indexer=self.indexer,
                guard_cfg=guard_cfg
            )
            # Optionally pre-load model here so first query is fast
            self.conv.llm.load(llm_key)

            progress(1.0, desc="Initialization complete")
            return f"Initialized successfully. Chunks: {len(chunk_docs)}. Model loaded: {llm_key}."
        except Exception as e:
            return "Initialization failed:\n" + "".join(traceback.format_exception_only(type(e), e)).strip()

    # ------------------ Chat engine with similarity cutoff ------------------
    def _ensure_model_loaded(self, model_key: str):
        if self.conv is None:
            raise RuntimeError("Pipeline not initialized yet. Click 'Initialize & Build' first.")
        if self.conv.llm.pipe is None or self.conv.llm.model_name != model_key:
            self.conv.llm.load(model_key)

    def _rag_chat_with_cutoff(
        self,
        user_query: str,
        llm_name: str,
        top_k: int,
        temperature: float,
        max_distance: float,
        system_description: str,
        guidelines: str,
    ) -> Dict[str, Any]:
        if self.conv is None:
            raise RuntimeError("Pipeline not initialized yet. Click 'Initialize & Build' first.")

        # intent
        intent = classify_intent(user_query)

        self.conv.cfg.top_k = max(1, int(top_k))
        self.conv.llm.gen_defaults["temperature"] = float(temperature)

        if intent == QueryIntent.CHITCHAT:
            # bypass retrieval entirely; no citations
            return self.conv.chat(
                user_query, llm_name=llm_name,
                system_description=system_description, guidelines=guidelines,
                llm_overrides=None, force_intent=QueryIntent.CHITCHAT
            )

        # BOOK_QA: retrieve -> apply cutoff
        raw_blocks = self.conv.indexer.retrieve(user_query, k=self.conv.cfg.top_k)
        blocks = [b for b in raw_blocks if (b.get("score") is None or b["score"] <= max_distance)]
        # IMPORTANT: do NOT fallback to a random page; let the system refuse if nothing relevant
        if not blocks:
            # build prompt with empty context by calling chat() with no forced blocks.
            # We can temporarily monkey-patch a minimal method to feed empty blocks,
            # but simpler is to set top_k=0 here and let chat() re-retrieve.
            # Instead, we directly emulate its steps:
            context = "(No relevant context found.)"
            prompt = self.conv._render_prompt(
                user_query=user_query, context=context,
                system_description=system_description, guidelines=guidelines
            )
            answer = self.conv.llm.generate(prompt).strip()
            # require refusal (no context) and no auto-citation
            answer = self.conv._guardrail_verify_and_fix(answer, [], require_citations=True)

            self.conv.memory.chat_memory.add_user_message(user_query)
            self.conv.memory.chat_memory.add_ai_message(answer)
            return {"model": llm_name, "answer": answer, "pages": []}

        # If we have blocks, proceed normally but **without** forcing extra pages
        context = self.conv._format_context(blocks)
        prompt = self.conv._render_prompt(
            user_query=user_query, context=context,
            system_description=system_description, guidelines=guidelines
        )
        answer = self.conv.llm.generate(prompt).strip()
        answer = self.conv._guardrail_verify_and_fix(answer, blocks, require_citations=True)

        self.conv.memory.chat_memory.add_user_message(user_query)
        self.conv.memory.chat_memory.add_ai_message(answer)
        pages = sorted({b["page_number"] for b in blocks if b.get("page_number")})
        return {"model": llm_name, "answer": answer, "pages": pages}


    # ------------------ Gradio wiring ------------------
    def _on_initialize_click(self, book_url, embedding_model, model_key):
        status = self.initialize_and_build(book_url, embedding_model, model_key)
        return status

    def _on_send(self, message, chat_history, model, top_k, temperature, cutoff):
        try:
            self._ensure_model_loaded(model)
            result = self._rag_chat_with_cutoff(
                user_query=message,
                llm_name=model,
                top_k=top_k,
                temperature=temperature,
                max_distance=cutoff,
                system_description=self.default_system,
                guidelines=self.default_guidelines,
            )
            chat_history = chat_history + [
                (message, f"{result['answer']}\n\nCitations: pages {result['pages']}")
            ]
            return "", chat_history
        except Exception as e:
            chat_history = chat_history + [(message, f"Error: {e}")]
            return "", chat_history

    def _on_clear(self):
        if self.conv is not None:
            self.conv.memory.clear()
        return []

    # Public: build the Blocks app
    def build_app(self):
        with gr.Blocks(theme=gr.themes.Soft(), fill_height=True) as demo:
            gr.Markdown("## TutorAI — RAG Dashboard")

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="TutorAI",
                        height=500,
                        show_copy_button=True,
                        bubble_full_width=False,
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your question",
                            placeholder="Ask about 'Storytelling with Data' (answers must cite p. N)",
                            scale=8,
                        )
                        send = gr.Button("Send", variant="primary", scale=1)
                    clear = gr.Button("Clear conversation")

                with gr.Column(scale=2):
                    gr.Markdown("### Initialization")
                    book_url = gr.Textbox(
                        label="Book URL",
                        value=self.default_book_url,
                        interactive=True,
                    )
                    embedding_model = gr.Textbox(
                        label="Embedding model (HF)",
                        value="sentence-transformers/all-MiniLM-L6-v2",
                        interactive=True,
                    )
                    model = gr.Dropdown(
                        label="Model",
                        choices=self.model_choices,
                        value=self.default_model_key,
                        interactive=True,
                    )
                    init_status = gr.Markdown(value="Not initialized.")
                    init_btn = gr.Button("Initialize & Build")

                    gr.Markdown("### Retrieval and Generation Controls")
                    top_k = gr.Slider(
                        minimum=1, maximum=12, value=5, step=1,
                        label="Context documents (top-k)"
                    )
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.5, value=0.2, step=0.05,
                        label="Temperature"
                    )
                    cutoff = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                        label="Similarity (max distance)",
                        info="Lower distance = more similar. Chunks above this distance are dropped.",
                    )

            # Events
            init_btn.click(
                fn=self._on_initialize_click,
                inputs=[book_url, embedding_model, model],
                outputs=[init_status],
            )

            send.click(
                fn=self._on_send,
                inputs=[msg, chatbot, model, top_k, temperature, cutoff],
                outputs=[msg, chatbot],
            )
            msg.submit(
                fn=self._on_send,
                inputs=[msg, chatbot, model, top_k, temperature, cutoff],
                outputs=[msg, chatbot],
            )
            clear.click(fn=self._on_clear, outputs=[chatbot])

        return demo

    # Optional helper to launch directly
    def launch(self, **kwargs):
        app = self.build_app()
        return app.launch(**kwargs)



