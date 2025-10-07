import os, json, yaml
from typing import Dict, Any, List

DEFAULT = {
  "display_name": "Reader",
  "seniority": "executive",
  "company": "Your Company",
  "tone": "crisp, practical, executive-ready",
  "interests": ["agents", "evals", "safety", "infra"],
  "callouts": ["cost/latency", "reliability", "open vs closed"]
}

def load_persona(path: str | None) -> Dict[str, Any]:
    if not path:
        return DEFAULT
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) if ext in [".yml", ".yaml"] else json.load(f)


def craft_intro(persona: Dict[str, Any], days: int, topics: List[str]) -> str:
    topic_str = (", ".join(topics)) if topics else "GenAI"
    return (
        f"This curated brief covers {topic_str} developments from the last {days} days, tailored for "
        f"a {persona.get('seniority','leader')} at {persona.get('company','your org')}."
    )


def render_items(persona: Dict[str, Any], items: List[Dict[str, Any]]):
    from .tools import summarize_item
    return [ summarize_item(persona, a) for a in items ]