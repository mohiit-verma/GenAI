import os, time, math, hashlib, json, re, datetime as dt
from typing import List, Dict, Any, Tuple
import feedparser
import trafilatura
from urllib.parse import urlparse

import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_FEEDS = [
  # Labs & vendors
  "https://openai.com/blog/rss.xml",
  "https://research.google/blog/feed/",
  "https://deepmind.google/resources/blog.xml",
  "https://ai.meta.com/blog/rss/",
  "https://anthropic.com/news/rss.xml",
  "https://microsoft.github.io/ai-blog/feed.xml",
  "https://huggingface.co/blog/rss.xml",
  # Community & standards
  "https://arxiv.org/rss/cs.AI",
  "https://arxiv.org/rss/cs.CL",
  "https://www.llamaindex.ai/blog/rss.xml",
]

TRY_SOURCES = [
  "openai", "deepmind", "google", "meta", "anthropic", "microsoft", "huggingface",
  "arxiv", "nvidia", "mistral", "cohere", "databricks"
]


def _iso(dt_struct):
    try:
        return dt.datetime(*dt_struct[:6]).isoformat()
    except Exception:
        return None


def discover(days: int, topics: List[str]) -> List[Dict[str, Any]]:
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    items = []
    for url in DEFAULT_FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries[:50]:
            published = None
            if hasattr(e, 'published_parsed') and e.published_parsed:
                published = _iso(e.published_parsed)
            elif hasattr(e, 'updated_parsed') and e.updated_parsed:
                published = _iso(e.updated_parsed)
            ts = dt.datetime.fromisoformat(published) if published else None
            if ts and ts < cutoff:
                continue
            title = e.title
            link = e.link
            if topics:
                if not any(t.lower() in (title or '').lower() for t in topics):
                    # keep but lower priority — handled at ranking time
                    pass
            items.append({
                "title": title,
                "url": link,
                "published": published,
                "source": urlparse(link).netloc,
                "summary": getattr(e, 'summary', '')[:500]
            })
    # TODO: optionally enrich with SerpAPI/NewsAPI here
    return items


def _extract(url: str) -> Dict[str, Any]:
    downloaded = trafilatura.fetch_url(url, no_ssl=True)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) if downloaded else None
    return {"url": url, "content": text or "", "ok": bool(text)}


def extract_all(candidates: List[Dict[str, Any]]):
    out = []
    for c in candidates:
        ex = _extract(c["url"])
        if not ex["ok"]:  
            continue
        ex.update({k: c.get(k) for k in ("title", "published", "source", "summary")})
        out.append(ex)
    return out


def _embeddings(texts: List[str]):
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _cos(a, b):
    import numpy as np
    a = np.array(a); b = np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cluster_and_rank(articles: List[Dict[str, Any]]):
    # simple greedy clustering by cosine similarity
    vecs = _embeddings([ (a.get("title") or "") + "\n" + (a.get("content")[:1500] or "") for a in articles ])
    clusters = []
    centers = []
    for i, v in enumerate(vecs):
        placed = False
        for ci, center in enumerate(centers):
            if _cos(v, center) > 0.85:
                clusters[ci].append(i)
                centers[ci] = [(x+y)/2 for x,y in zip(centers[ci], v)]
                placed = True
                break
        if not placed:
            clusters.append([i])
            centers.append(v)
    # rank clusters by recency + source weight + topic hit
    def score_idx(idx):
        a = articles[idx]
        rec = 0.0
        if a.get("published"):
            try:
                days_ago = (dt.datetime.utcnow() - dt.datetime.fromisoformat(a["published"])) .days
                rec = max(0.0, 30 - days_ago) / 30.0
            except Exception:
                pass
        src = (a.get("source") or "").lower()
        src_bonus = 0.2 if any(s in src for s in TRY_SOURCES) else 0.0
        return rec + src_bonus
    cluster_reps = [max(g, key=score_idx) for g in clusters]
    # sort reps by score desc
    tops = sorted(cluster_reps, key=score_idx, reverse=True)[:12]
    return clusters, tops


def _chat(messages):
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content


def summarize_item(persona: Dict[str, Any], a: Dict[str, Any]) -> str:
    system = f"You are an expert editor for {persona.get('company','a company')}. Tone: {persona.get('tone','crisp')}"
    user = (
        f"Title: {a.get('title')}\nURL: {a.get('url')}\nSource: {a.get('source')}\nPublished: {a.get('published')}\n\n"
        f"Content:\n{a.get('content','')[:4000]}\n\n"
        "Write ≤120 words covering: What changed, Why it matters, Enterprise angle. Include the title as a bold heading and end with the source as a markdown link."
    )
    out = _chat([{"role":"system","content":system},{"role":"user","content":user}])
    return out


def write_markdown(persona: Dict[str, Any], intro: str, items_md: List[str], days: int) -> str:
    date_range = f"Last {days} days"
    parts = [f"# GenAI — {date_range}", "", intro, "", "## Top Developments", ""]
    parts.extend(items_md)
    parts.extend(["", "## Signals to Watch", "- (Add your notes here)", "", f"---\n*Generated for {persona.get('display_name','Reader')}.*"])
    return "\n".join(parts)


def save_markdown(md: str, out_dir: str, prefix: str = "genai-newsletter") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(out_dir, f"{prefix}_{ts}.md"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path