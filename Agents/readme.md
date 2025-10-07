# GenAI Newsletter Agent (LangGraph + MCP)

## Quickstart
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill `OPENAI_API_KEY`.
4. `python mcp_server.py` to launch the MCP server.

## From an MCP client
- Tool: `genai_newsletter.generate`
- Example args:
```json
{
  "days": 30,
  "topics": "agents, safety, evals",
  "persona_path": "personas/default.yaml",
  "filename_prefix": "zenon-genei"
}
```

## Project Structure

├── README.md
├── requirements.txt
├── .env
├── mcp_server.py                 # MCP server exposing the agent as a tool
├── agent/
│   ├── graph.py                  # LangGraph assembly
│   ├── tools.py                  # IO / search / fetch / extract / dedupe / save
│   ├── prompts.py                # System & summarization prompts
│   ├── personalization.py        # Persona handling and tailoring
│   └── utils.py                  # Helpers (logging, timing, hashing)
├── personas/
│   └── default.yaml              # Example persona profile
└── run_local.py                  # Optional: run the pipeline without MCP