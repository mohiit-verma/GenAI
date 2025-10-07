import argparse, os
from agent.graph import build_graph
from agent.personalization import load_persona
from agent.tools import save_markdown

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--topics", type=str, default="")
    ap.add_argument("--persona", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="./out")
    ap.add_argument("--prefix", type=str, default="genai-newsletter")
    args = ap.parse_args()

    graph = build_graph()
    state = {
        "days": args.days,
        "topics": args.topics,
        "persona": load_persona(args.persona or None)
    }
    final = graph.invoke(state)
    path = save_markdown(final["markdown"], args.out_dir, args.prefix)
    print(f"Saved → {os.path.abspath(path)}")