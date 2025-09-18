# llm.py

from __future__ import annotations

# =========================
# Standard library imports
# =========================
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, List
from openai import OpenAI  # OpenAI Python SDK v1.x
import torch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from .config import DEFAULT_LLM_CATALOG, LLMConfig
# =========================
# Third-party (optional)
# =========================
# We import these lazily inside the loader functions where possible to avoid
# hard failures if a backend isn't installed. Kept here for type hints / clarity.
try:
    from openai import OpenAI  # OpenAI Python SDK v1.x
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = pipeline = None  # type: ignore


# =========================
# Intent classification
# =========================

_CIT_RE = re.compile(r"\(p\.\s*(\d+)(?:\s*[-–]\s*\d+)?\)")
CITATION_REGEX = re.compile(r"\(p\.\s*\d+(?:\s*[-–]\s*\d+)?\)")  # (p. N) or (p. N–M)

# --- add near the top of your conv file ---
import re
from enum import Enum

class QueryIntent(str, Enum):
    BOOK_QA = "book_qa"
    CHITCHAT = "chitchat"

_CHITCHAT_PATTERNS = [
    r"^hi[!.]?$", r"^hello[!.]?$", r"^hey[!.]?$",
    r"\bthank(s| you)\b",
    r"\bhow (are|r) (you|u)\b",
    r"\bwhat can (you|u) do\b",
    r"\bwho are you\b", r"\bhelp\b", r"\bcapabilit(y|ies)\b",
    r"\btest\b", r"\bping\b",
]

def classify_intent(text: str) -> QueryIntent:
    t = (text or "").strip().lower()
    for pat in _CHITCHAT_PATTERNS:
        if re.search(pat, t):
            return QueryIntent.CHITCHAT
    # default to book QA
    return QueryIntent.BOOK_QA


# -----------------------------
# LLM registry & loader
# -----------------------------


class LLMManager:
    """Load and run LLMs (OpenAI via API or local HF via transformers)."""
    def __init__(self, catalog: Optional[Dict[str, LLMConfig]] = None):
        self.catalog = catalog or DEFAULT_LLM_CATALOG
        self.pipe = None  # for HF pipeline
        self.client = None  # for OpenAI
        self.model_name: Optional[str] = None
        self.provider: Optional[str] = None
        self.gen_defaults: Dict[str, Any] = {}

    def load(self, name: str, gen_kwargs: Optional[Dict[str, Any]] = None):
        if name not in self.catalog:
            raise ValueError(f"Unknown model '{name}'. Available: {list(self.catalog)}")
        cfg = self.catalog[name]
        self.model_name = name
        self.provider = cfg.provider

        if cfg.provider == "openai":
            # Uses OPENAI_API_KEY from env (dotenv already loaded at import time)
            self.client = OpenAI()
            self.pipe = None
            self.gen_defaults = {
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 512,
                **(gen_kwargs or {}),
            }
        else:
            # HuggingFace transformers local pipeline (kept for completeness)
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                None: None,
            }[cfg.dtype]

            tok = AutoTokenizer.from_pretrained(cfg.repo_id, trust_remote_code=cfg.trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.repo_id,
                torch_dtype=dtype,
                device_map=cfg.device_map,
                trust_remote_code=cfg.trust_remote_code,
            )
            self.pipe = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                return_full_text=False,
                pad_token_id=tok.eos_token_id,
            )
            self.client = None
            self.gen_defaults = {
                "max_new_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.9,
                "do_sample": True,
                **(gen_kwargs or {}),
            }

    def generate(self, prompt: str, **overrides) -> str:
        if self.provider == "openai":
            if self.client is None or self.model_name is None:
                raise RuntimeError("OpenAI client not initialized. Call load(name) first.")
            params = {**self.gen_defaults, **overrides}
            # translate common knobs where relevant
            temperature = params.get("temperature", 0.2)
            top_p = params.get("top_p", 0.9)
            max_tokens = params.get("max_tokens", 512)
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        else:
            if self.pipe is None:
                raise RuntimeError("HF LLM not loaded. Call load(name) first.")
            out = self.pipe(prompt, **{**self.gen_defaults, **overrides})
            return out[0]["generated_text"]

# -----------------------------
# Prompt template (with placeholders)
# -----------------------------
SYSTEM_TEMPLATE = """\
You are TutorAI, an AI tutor that answers STRICTLY from the provided book context.
Rules:
- Only use the context (extracted from the book) to answer.
- If the answer is not in the context, politely refuse and say it's not in the text.
- Always include page citations in the form (p. <page_number>) for each claim.
- Be concise but clear. Use bullet points where appropriate.
- Maintain continuity with conversation history when helpful.
{system_description}
"""

GUIDELINES_TEMPLATE = """\
Guidelines:
{guidelines}
"""

# ChatPromptTemplate supports message roles + placeholders
CONV_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder("history"),
        (
            "human",
            # context is injected separately (retrieved chunks)
            "User query:\n{user_query}\n\n"
            "Book context (use ONLY this; provide citations for each relevant statement):\n{context}\n\n"
            + GUIDELINES_TEMPLATE
        ),
    ]
)


# -----------------------------
# Conversation pipeline
# -----------------------------
@dataclass
class ConversationConfig:
    top_k: int = 5
    max_context_chars: int = 6000         # guardrail to avoid exceeding model limits
    min_chars_per_chunk: int = 120        # filter very tiny chunks
    join_delimiter: str = "\n\n---\n\n"   # separator between chunks

@dataclass
class GuardrailConfig:
    mode: str = "append_once"   # "append_once" | "per_paragraph"
    require_on_any_output: bool = True
    paragraph_min_chars: int = 20          # ignore tiny paras when per_paragraph
    max_auto_citations_per_answer: int = 1 # for append_once
    refusal_text: str = (
        "I can’t find that in the provided text. Please point me to a specific section, "
        "or ask about content that appears in the book."
    )

class TutorAIConversationPipeline:
    def __init__(
        self,
        indexer,  # instance of TutorAIVectorIndex
        llm_manager: Optional[LLMManager] = None,
        conv_cfg: ConversationConfig = ConversationConfig(),
        guard_cfg: GuardrailConfig = GuardrailConfig(),
    ):
        self.indexer = indexer
        self.llm = llm_manager or LLMManager()
        self.cfg = conv_cfg
        self.guard = guard_cfg

        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="history", input_key="user_query"
        )

    def read_user_query(self, text: str) -> str:
        return (text or "").strip()

    def _retrieve_context_blocks(self, query: str) -> List[Dict[str, Any]]:
        hits = self.indexer.retrieve(query, k=self.cfg.top_k)
        cleaned = [h for h in hits if len(h["text"]) >= self.cfg.min_chars_per_chunk]
        return cleaned or hits

    def _format_context(self, blocks: List[Dict[str, Any]]) -> str:
        parts, running_len = [], 0
        for h in blocks:
            txt = h["text"].strip()
            cite = f"(p. {h.get('page_number')})"
            chunk = f"[Page {h.get('page_number')}]\n{txt}\n{cite}"
            if running_len + len(chunk) > self.cfg.max_context_chars:
                break
            parts.append(chunk)
            running_len += len(chunk) + len(self.cfg.join_delimiter)
        return self.cfg.join_delimiter.join(parts)

    def _render_prompt(
        self,
        user_query: str,
        context: str,
        system_description: str = "",
        guidelines: str = "Answer only from the context and include page citations.",
    ) -> str:
        

        SYSTEM_TEMPLATE = """You are TutorAI, an AI tutor that answers STRICTLY from the provided book context.
Rules:
- Only use the context (extracted from the book) to answer.
- If the answer is not in the context, politely refuse and say it's not in the text.
- Always include page citations in the form (p. <page_number>) for each claim.
- Be concise but clear. Use bullet points where appropriate.
- Maintain continuity with conversation history when helpful.
{system_description}
"""
        GUIDELINES_TEMPLATE = "Guidelines:\n{guidelines}"
        CONV_PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_TEMPLATE),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "User query:\n{user_query}\n\n"
                    "Book context (use ONLY this; provide citations for each relevant statement):\n{context}\n\n"
                    + GUIDELINES_TEMPLATE
                ),
            ]
        )

        history = self.memory.load_memory_variables({}).get("history", [])
        msgs = CONV_PROMPT.format_messages(
            system_description=system_description,
            user_query=user_query,
            context=context if context.strip() else "(No relevant context found.)",
            guidelines=guidelines,
            history=history,
        )

        rendered = []
        for m in msgs:
            if isinstance(m, SystemMessage):
                rendered.append(f"<|system|>\n{m.content}\n")
            elif isinstance(m, HumanMessage):
                rendered.append(f"<|user|>\n{m.content}\n")
            else:
                rendered.append(f"<|assistant|>\n{m.content}\n")
        rendered.append("<|assistant|>\n")
        return "\n".join(rendered)

    # ---------- Guardrail verifier ----------
    def _extract_pages_from_blocks(self, blocks: List[Dict[str, Any]]) -> List[int]:
        pages = [b.get("page_number") for b in blocks if b.get("page_number")]
        # preserve order of appearance, dedupe
        seen, ordered = set(), []
        for p in pages:
            if p not in seen:
                ordered.append(int(p))
                seen.add(p)
        return ordered

    def _has_any_citation(self, text: str) -> bool:
        return bool(CITATION_REGEX.search(text))

    def _append_once_citation(self, answer: str, pages: List[int]) -> str:
        if not pages:
            return answer
        pages_str = ", ".join(str(p) for p in pages)
        suffix = f" (p. {pages_str})"
        # Avoid appending twice
        if not answer.rstrip().endswith(")"):
            return answer.rstrip() + " " + suffix
        return answer.rstrip() + " " + suffix

    def _per_paragraph_citations(self, answer: str, pages: List[int]) -> str:
        if not pages:
            return answer
        paras = [p for p in answer.split("\n")]

        # cycle through pages to distribute citations
        i = 0
        fixed = []
        for para in paras:
            clean = para.rstrip()
            if len(clean.strip()) < self.cfg.min_chars_per_chunk:
                fixed.append(clean)
                continue
            if CITATION_REGEX.search(clean):
                fixed.append(clean)
            else:
                page = pages[i % len(pages)]
                fixed.append(f"{clean} (p. {page})")
                i += 1
        return "\n".join(fixed)

    def _enforce_refusal_when_no_context(self, answer: str, blocks: List[Dict[str, Any]]) -> str:
        if blocks:
            return answer
        # No context → ensure refusal
        if self.guard.refusal_text.lower() not in answer.lower() and "(p." not in answer:
            return self.guard.refusal_text
        return answer

    def _guardrail_verify_and_fix(self, answer: str, blocks: List[Dict[str, Any]]) -> str:
        # First, if we had no retrieved context, enforce refusal
        answer = self._enforce_refusal_when_no_context(answer, blocks)

        pages = self._extract_pages_from_blocks(blocks)
        if not self.guard.require_on_any_output:
            return answer

        if self.guard.mode == "per_paragraph":
            # Ensure each paragraph has a citation
            return self._per_paragraph_citations(answer, pages)

        # append_once (default)
        if not self._has_any_citation(answer):
            # Cap the total appended groups to avoid spam
            return self._append_once_citation(answer, pages[: max(1, self.guard.max_auto_citations_per_answer)])
        return answer
    # ---------- /Guardrail verifier ----------

    def _guardrail_verify_and_fix(
        self, answer: str, blocks: List[Dict[str, Any]], require_citations: bool = True
    ) -> str:
        # If no citations required (like chit-chat), only enforce refusal
        if not require_citations:
            return self._enforce_refusal_when_no_context(answer, blocks)

        # Otherwise run the full citation guard
        answer = self._enforce_refusal_when_no_context(answer, blocks)
        pages = self._extract_pages_from_blocks(blocks)

        if not self.guard.require_on_any_output:
            return answer

        if self.guard.mode == "per_paragraph":
            return self._per_paragraph_citations(answer, pages)

        if not self._has_any_citation(answer):
            return self._append_once_citation(
                answer, pages[: max(1, self.guard.max_auto_citations_per_answer)]
            )
        return answer

    def _chitchat_reply(self, user_query: str) -> str:
        # keep it short; no book claims, no citations
        return (
            "I’m TutorAI. I can answer questions strictly from *Storytelling with Data* "
            "and cite pages. Ask about charts, storytelling, design principles, or any topic from the book."
        )


    def chat(
        self,
        user_query: str,
        *,
        llm_name: str,
        system_description: str = "",
        guidelines: str = "Answer only from the context and include page citations.",
        llm_overrides: Optional[Dict[str, Any]] = None,
        force_intent: Optional[QueryIntent] = None,   # allow override from UI if needed
    ) -> Dict[str, Any]:
        if self.llm.pipe is None or self.llm.model_name != llm_name:
            self.llm.load(llm_name)

        query = self.read_user_query(user_query)
        intent = force_intent or classify_intent(query)

        # --- CHITCHAT path: bypass RAG, no citations ---
        if intent == QueryIntent.CHITCHAT:
            answer = self._chitchat_reply(query)
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(answer)
            return {
                "model": llm_name,
                "answer": answer,
                "citations_pages": [],
                "retrieved_context": [],
                "history_len": len(self.memory.chat_memory.messages),
            }

        # --- BOOK_QA path: regular RAG ---
        blocks = self._retrieve_context_blocks(query)
        context = self._format_context(blocks)

        prompt = self._render_prompt(
            user_query=query,
            context=context if context.strip() else "(No relevant context found.)",
            system_description=system_description,
            guidelines=guidelines,
        )

        answer = self.llm.generate(prompt, **(llm_overrides or {})).strip()

        # require citations only for BOOK_QA
        answer = self._guardrail_verify_and_fix(answer, blocks, require_citations=True)

        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(answer)

        pages = sorted({b["page_number"] for b in blocks if b.get("page_number")})
        return {
            "model": llm_name,
            "answer": answer,
            "citations_pages": pages,
            "retrieved_context": blocks,
            "history_len": len(self.memory.chat_memory.messages),
        }

    def _extract_pages_from_answer(self, text: str):
        pages = [int(m.group(1)) for m in _CIT_RE.finditer(text or "")]
        # dedupe, keep order
        seen, out = set(), []
        for p in pages:
            if p not in seen:
                out.append(p); seen.add(p)
        return out

