import os, json, argparse
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
from typing import Any, Dict

from agent.graph import build_graph
from agent.personalization import load_persona
from agent.tools import write_markdown, save_markdown

server = Server("genai-newsletter-server")

@server.call_tool()
def generate(days: int = 30,
             topics: str | None = None,
             persona_path: str | None = None,
             out_dir: str | None = None,
             filename_prefix: str = "genai-newsletter") -> Dict[str, Any]:
    """Generate a personalized GenAI newsletter for the last `days` days and save as markdown.

    Args:
      days: lookback window in days
      topics: comma-separated topics filter (optional)
      persona_path: path to YAML/JSON persona file (optional)
      out_dir: directory to save the markdown (defaults to OUTPUT_DIR)
      filename_prefix: filename prefix for the output file
    Returns: path to markdown and basic stats
    """
    persona = load_persona(persona_path)
    graph = build_graph()
    state = {
        "days": int(days),
        "topics": topics or "",
        "persona": persona
    }
    final = graph.invoke(state)
    md = final["markdown"]
    out_dir = out_dir or os.getenv("OUTPUT_DIR", "./out")
    path = save_markdown(md, out_dir, filename_prefix)
    return {"path": os.path.abspath(path),
            "items_considered": final.get("stats",{}).get("candidates", 0),
            "items_included": final.get("stats",{}).get("included", 0)}

if __name__ == "__main__":
    stdio_server(server)