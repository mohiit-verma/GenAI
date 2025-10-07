from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Annotated

class S(TypedDict):
    days: int
    topics: List[str]
    persona: Dict[str, Any]
    candidates: List[Dict[str, Any]]  # raw discoveries
    articles: List[Dict[str, Any]]    # extracted + cleaned
    clusters: List[List[int]]         # groups of similar articles
    top: List[int]                    # representative item idx per cluster
    markdown: str
    stats: Dict[str, Any]

from .tools import discover, extract_all, cluster_and_rank, write_markdown
from .personalization import load_persona, craft_intro, render_items

def reducer(a: list, b: int | None) -> list:
        if b is not None:
            return a + [b]
        return a

class State(TypedDict):
    x: Annotated[list, reducer]

def build_graph():
    g = StateGraph(state_type=S, state_schema=S)

    def _init(state: S):
        print("[init] Starting initialization")
        if isinstance(state.get("topics"), str):
            state["topics"] = [t.strip() for t in state["topics"].split(",") if t.strip()]
            print(f"[init] Converted topics to list: {state['topics']}")
        if not state.get("persona"):
            state["persona"] = load_persona(None)
            print("[init] Loaded default persona")
        print("[init] Initialization complete")
        return state

    def _discover(state: S):
        print("[discover] Discovering candidates")
        state["candidates"] = discover(state["days"], state.get("topics", []))
        state.setdefault("stats", {})["candidates"] = len(state["candidates"])
        print(f"[discover] Found {len(state['candidates'])} candidates")
        return state

    def _extract(state: S):
        print("[extract] Extracting articles")
        state["articles"] = extract_all(state["candidates"])  # content, meta
        state["stats"]["articles"] = len(state["articles"])
        print(f"[extract] Extracted {len(state['articles'])} articles")
        return state

    def _cluster(state: S):
        print("[cluster] Clustering and ranking articles")
        clusters, top = cluster_and_rank(state["articles"])  # dedupe + ranking
        state["clusters"], state["top"] = clusters, top
        print(f"[cluster] Formed {len(clusters)} clusters, top indices: {top}")
        return state

    def _compose(state: S):
        print("[compose] Composing markdown")
        intro = craft_intro(state["persona"], state["days"], state.get("topics", []))
        rendered = render_items(state["persona"], [state["articles"][i] for i in state["top"]])
        state["markdown"] = write_markdown(state["persona"], intro, rendered, state["days"])
        state["stats"]["included"] = len(state["top"])
        print(f"[compose] Markdown composed with {len(state['top'])} items included")
        return state

    g.add_node("init", _init)
    g.add_node("discover", _discover)
    g.add_node("extract", _extract)
    g.add_node("cluster", _cluster)
    g.add_node("compose", _compose)

    g.set_entry_point("init")
    g.add_edge("init", "discover")
    g.add_edge("discover", "extract")
    g.add_edge("extract", "cluster")
    g.add_edge("cluster", "compose")
    g.add_edge("compose", END)

    print("[build_graph] Graph construction complete")
    return g.compile()