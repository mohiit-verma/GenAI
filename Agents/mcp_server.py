import os
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP

from agent.graph import build_graph
from agent.personalization import load_persona
from agent.tools import save_markdown

mcp = FastMCP("genai-newsletter-server")


@mcp.tool()
def generate(
    days: int = 30,
    topics: str | None = None,
    persona_path: str | None = None,
    out_dir: str | None = None,
    filename_prefix: str = "genai-newsletter"
) -> Dict[str, Any]:
    """Generate a personalized GenAI newsletter for the last `days` days and save as markdown."""
    print(f"[generate] Called with days={days}, topics={topics}, persona_path={persona_path}, out_dir={out_dir}, filename_prefix={filename_prefix}")
    persona = load_persona(persona_path)
    print("[generate] Loaded persona")
    graph = build_graph()
    print("[generate] Built graph")
    final = graph.invoke({
        "days": int(days),
        "topics": topics or "",
        "persona": persona,
    })
    print("[generate] Invoked graph")
    md = final["markdown"]
    out_dir = out_dir or os.getenv("OUTPUT_DIR", "./out")
    print(f"[generate] Output directory set to: {out_dir}")
    path = save_markdown(md, out_dir, filename_prefix)
    print(f"[generate] Markdown saved at: {path}")
    result = {
        "path": os.path.abspath(path),
        "items_considered": final.get("stats", {}).get("candidates", 0),
        "items_included": final.get("stats", {}).get("included", 0),
    }
    print(f"[generate] Returning result: {result}")
    return result


if __name__ == "__main__":
    print("[main] Starting MCP server...")
    # FastMCP handles stdio transport and initialization for you.
    mcp.run()  # or mcp.run(transport="stdio")