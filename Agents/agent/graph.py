from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

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


def build_graph():
    g = StateGraph(state_type=S)

    def _init(state: S):
        if isinstance(state.get("topics"), str):
            state["topics"] = [t.strip() for t in state["topics"].split(",") if t.strip()]
        if not state.get("persona"):
            state["persona"] = load_persona(None)
        return state

    def _discover(state: S):
        state["candidates"] = discover(state["days"], state.get("topics", []))
        state.setdefault("stats", {})["candidates"] = len(state["candidates"])
        return state

    def _extract(state: S):
        state["articles"] = extract_all(state["candidates"])  # content, meta
        state["stats"]["articles"] = len(state["articles"])
        return state

    def _cluster(state: S):
        clusters, top = cluster_and_rank(state["articles"])  # dedupe + ranking
        state["clusters"], state["top"] = clusters, top
        return state

    def _compose(state: S):
        intro = craft_intro(state["persona"], state["days"], state.get("topics", []))
        rendered = render_items(state["persona"], [state["articles"][i] for i in state["top"]])
        state["markdown"] = write_markdown(state["persona"], intro, rendered, state["days"])
        state["stats"]["included"] = len(state["top"])
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

    return g.compile()