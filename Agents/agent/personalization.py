import os
import json
import yaml
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
    print(f"[load_persona] Called with path: {path}")
    if not path:
        print("[load_persona] No path provided, using DEFAULT persona.")
        return DEFAULT
    ext = os.path.splitext(path)[1].lower()
    print(f"[load_persona] File extension detected: {ext}")
    with open(path, "r", encoding="utf-8") as f:
        if ext in [".yml", ".yaml"]:
            print("[load_persona] Loading YAML file.")
            data = yaml.safe_load(f)
        else:
            print("[load_persona] Loading JSON file.")
            data = json.load(f)
        print(f"[load_persona] Persona loaded: {data}")
        return data


def craft_intro(persona: Dict[str, Any], days: int, topics: List[str]) -> str:
    print(f"[craft_intro] Called with persona: {persona}, days: {days}, topics: {topics}")
    topic_str = ", ".join(topics) if topics else "GenAI"
    intro = (
        f"This curated brief covers {topic_str} developments from the last {days} days, tailored for "
        f"a {persona.get('seniority', 'leader')} at {persona.get('company', 'your org')}."
    )
    print(f"[craft_intro] Generated intro: {intro}")
    return intro


def render_items(persona: Dict[str, Any], items: List[Dict[str, Any]]):
    print(f"[render_items] Called with persona: {persona}, number of items: {len(items)}")
    from .tools import summarize_item
    summaries = [summarize_item(persona, a) for a in items]
    print(f"[render_items] Summaries generated: {summaries}")
    return summaries