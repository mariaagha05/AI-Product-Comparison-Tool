"""
AI Product Comparison Tool  (Streamlit + Groq)

Upload product specification documents, describe what you need, and get a
comparison table, explained trade-offs and a constrained recommendation.

Run:  streamlit run app.py
"""

import hashlib
import html
import io
import json
import os
import re
import time

import pandas as pd
import streamlit as st

try:
    from groq import Groq
except ImportError:  # shown as a friendly message instead of a crash
    Groq = None

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Product Comparison Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "csv", "xlsx", "json"]
MAX_PRODUCTS = 6

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
  --ink: #1B2A41;
  --muted: #4B5B6B;
  --page: #F3F6F9;
  --card: #FFFFFF;
  --line: #D5DEE6;
  --accent: #0B6E6E;
  --accent-soft: #E1F2F1;
  --ok: #1B7F4B;      --ok-soft: #E6F4EC;
  --warn: #9A5B00;    --warn-soft: #FEF3DC;
  --bad: #B42318;     --bad-soft: #FDECEA;
  --unk: #475467;     --unk-soft: #EAECF0;
  --body: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif;
  --head: 'Manrope', 'Inter', 'Segoe UI', Arial, sans-serif;
}

.stApp { background: var(--page); }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }

.stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input,
.stApp td, .stApp th, .stApp button, .stApp .stMarkdown {
  font-family: var(--body);
  color: var(--ink);
}
.stApp p, .stApp li { font-size: 1.02rem; line-height: 1.6; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: var(--head); color: var(--ink); letter-spacing: -0.01em;
}
.stApp h2 { font-size: 1.45rem; margin-bottom: .2rem; }
.stApp h3 { font-size: 1.15rem; }
.stApp label p, .stApp label { font-weight: 600; font-size: 0.98rem; }
.stApp small, .stApp [data-testid="stCaptionContainer"] p { color: var(--muted); font-size: .9rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: var(--ink); }

/* Inputs */
.stApp textarea, .stApp input {
  background: #FFFFFF !important; color: var(--ink) !important;
  border-radius: 10px !important;
}
.stApp [data-baseweb="select"] > div { background: #FFFFFF; color: var(--ink); border-radius: 10px; }
.stApp textarea::placeholder, .stApp input::placeholder { color: #7A8794 !important; opacity: 1; }
[data-testid="stFileUploader"] section {
  background: #FFFFFF; border: 2px dashed #9FB3C4; border-radius: 14px;
}
[data-testid="stFileUploader"] section * { color: var(--ink); }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; font-size: 1rem; padding: .6rem 1.2rem;
  border: 1px solid var(--line); background: #FFFFFF; color: var(--ink);
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--accent) !important; color: #FFFFFF !important; border: 1px solid var(--accent) !important;
}
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color: #FFFFFF !important; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 3px solid #7CC4C4; outline-offset: 2px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 1rem; padding: .6rem 1rem; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }

/* Header */
.app-title { font-family: var(--head); font-weight: 800; font-size: 2.2rem; color: var(--ink); margin: 0; }
.app-sub { color: var(--muted); font-size: 1.08rem; margin: .35rem 0 1.2rem 0; max-width: 60ch; }
.steps { display: flex; gap: .75rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
.step { flex: 1 1 220px; background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: .75rem 1rem; display: flex; gap: .75rem; align-items: center; }
.step-n { width: 30px; height: 30px; border-radius: 50%; background: var(--accent); color: #fff;
          display: flex; align-items: center; justify-content: center; font-weight: 700; flex: none; }
.step-t { font-weight: 600; color: var(--ink); }
.step-d { color: var(--muted); font-size: .88rem; }

/* Section blocks */
.section-head { margin: 1.6rem 0 .6rem 0; }
.section-head h2 { margin: 0; }
.section-head p { color: var(--muted); margin: .15rem 0 0 0; }
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); border-radius: 14px; }

/* Result cards */
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: .9rem; }
.card h3 { margin: 0 0 .3rem 0; }
.card p { margin: .25rem 0; }
.winner { border: 2px solid var(--accent); background: linear-gradient(0deg, #FFFFFF, #F6FBFB); }
.tag { display: inline-block; font-weight: 700; font-size: .85rem; border-radius: 999px; padding: .15rem .65rem; }
.tag-ok   { background: var(--ok-soft);   color: var(--ok); }
.tag-warn { background: var(--warn-soft); color: var(--warn); }
.tag-bad  { background: var(--bad-soft);  color: var(--bad); }
.tag-unk  { background: var(--unk-soft);  color: var(--unk); }
.tag-acc  { background: var(--accent-soft); color: var(--accent); }
.rank-row { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
.rank-name { font-weight: 700; font-size: 1.05rem; flex: 1 1 200px; color: var(--ink); }
.bar { height: 10px; background: #E3E9EE; border-radius: 999px; overflow: hidden; flex: 2 1 160px; }
.fill { height: 100%; background: var(--accent); border-radius: 999px; }
.fill.dq { background: #98A2B3; }
.score { font-weight: 700; min-width: 3.2rem; text-align: right; color: var(--ink); }
.reason { color: var(--bad); font-size: .93rem; margin: .35rem 0 0 0; }
.note { color: var(--muted); font-size: .93rem; }

/* Comparison table */
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; background: #fff; }
table.cmp { border-collapse: collapse; width: 100%; min-width: 640px; }
table.cmp th { background: #EAF0F5; color: var(--ink); font-weight: 700; text-align: left;
               padding: .8rem .9rem; border-bottom: 1px solid var(--line); font-size: .97rem; }
table.cmp td { padding: .75rem .9rem; border-bottom: 1px solid #E8EDF2; vertical-align: top;
               color: var(--ink); font-size: .95rem; }
table.cmp tr.group td { background: #F7F9FB; font-weight: 700; color: var(--accent); }
table.cmp td.req { font-weight: 600; min-width: 200px; }
table.cmp .ev { display: block; color: var(--muted); font-size: .85rem; margin-top: .3rem; line-height: 1.4; }
table.cmp tr.total td { background: #F3F8F8; font-weight: 700; }

@media (max-width: 640px) { .app-title { font-size: 1.7rem; } }
</style>
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Raised with a user-friendly message."""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def default_api_key() -> str:
    try:
        value = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        value = ""
    return value or os.environ.get("GROQ_API_KEY", "")


def decode_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


@st.cache_data(show_spinner=False, max_entries=64)
def parse_document(name: str, data: bytes) -> str:
    """Return plain text from an uploaded file. Raises ValueError with a clear message."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    text = ""

    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("The PDF is password protected.")
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if not text.strip():
            raise ValueError("No selectable text found (this may be a scanned PDF).")

    elif ext == "docx":
        import docx

        document = docx.Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        text = "\n".join(parts)

    elif ext == "xlsx":
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl")
        parts = []
        for sheet_name, frame in sheets.items():
            parts.append(f"Sheet: {sheet_name}")
            parts.append(frame.to_csv(index=False))
        text = "\n".join(parts)

    elif ext == "csv":
        try:
            frame = pd.read_csv(io.BytesIO(data))
            text = frame.to_csv(index=False)
        except Exception:
            text = decode_bytes(data)

    elif ext in ("txt", "md", "json"):
        text = decode_bytes(data)

    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 15:
        raise ValueError("The file contains too little text to analyse.")
    return text


def parse_lines(raw: str) -> list:
    items = []
    for line in (raw or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def extract_json(text: str):
    if not text or not text.strip():
        raise ValueError("Empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("No JSON object found")


def wait_seconds(message: str) -> float:
    match = re.search(r"try again in\s*(?:(\d+)m)?\s*([\d.]+)\s*(ms|s)", message)
    if not match:
        return 15.0
    minutes = float(match.group(1) or 0)
    value = float(match.group(2))
    seconds = minutes * 60 + (value / 1000 if match.group(3) == "ms" else value)
    return min(max(seconds + 1.0, 3.0), 60.0)


def call_llm(client, model, system, user, temperature=0.2, max_tokens=3000) -> dict:
    """Call Groq and return parsed JSON. Retries on rate limits / malformed JSON."""
    use_json_mode = True
    reminder = ""
    for attempt in range(4):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + reminder},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            data = extract_json(content)
            if not isinstance(data, dict):
                raise ValueError("JSON root is not an object")
            return data
        except ValueError:
            reminder = "\n\nIMPORTANT: Reply with one valid JSON object only. No prose, no markdown fences."
            continue
        except Exception as exc:  # noqa: BLE001
            name, msg = type(exc).__name__, str(exc)
            low = msg.lower()
            if name == "AuthenticationError" or "invalid api key" in low:
                raise LLMError("Groq rejected the API key. Check the key in the sidebar.")
            if name == "RateLimitError" or "rate limit" in low or "429" in low:
                if attempt < 3:
                    time.sleep(wait_seconds(msg))
                    continue
                raise LLMError(
                    "Groq rate limit reached. Wait a minute, lower the 'characters per document' "
                    "setting, or pick a smaller model, then try again."
                )
            if use_json_mode and ("response_format" in low or "json_validate" in low or "json mode" in low):
                use_json_mode = False
                continue
            if "decommissioned" in low or "does not exist" in low or "model_not_found" in low or name == "NotFoundError":
                raise LLMError(f"The model '{model}' is not available on your Groq account. Choose another model in the sidebar.")
            if name == "APIConnectionError" or "connection" in low or "timed out" in low:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise LLMError("Could not reach Groq. Check your internet connection and try again.")
            if "too large" in low or "413" in low or "context length" in low or "reduce your message" in low:
                raise LLMError("The documents are too long for this model. Lower 'characters per document' in the sidebar.")
            raise LLMError(f"Groq returned an error: {msg[:300]}")
    raise LLMError("The model returned an unreadable answer. Try again or choose a different model.")


# --------------------------------------------------------------------------- #
# LLM stages
# --------------------------------------------------------------------------- #
EXTRACT_SYSTEM = (
    "You are a meticulous product analyst. Extract specifications ONLY from the supplied document. "
    "Never invent facts. If a value is not stated, leave it out. Reply with one JSON object only."
)

EXTRACT_USER = """Document name: {name}

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"

Return JSON with exactly this shape:
{{
  "product_name": "string",
  "brand": "string or null",
  "price_value": number or null,
  "currency": "string like USD, EUR, PKR or null",
  "summary": "one or two sentences",
  "specs": {{"Short spec name": "value with units"}},
  "strengths": ["short strings taken from the document"],
  "limitations": ["short strings taken from the document"]
}}
Use short, common spec names (for example "Battery life", "Weight", "Warranty", "Display size")
so that different products can be compared. Include at most 40 specs."""

ASSESS_SYSTEM = (
    "You judge whether products satisfy a buyer's requirements, using ONLY the product data provided. "
    "Be strict and honest. Use 'unknown' when the data does not say. Reply with one JSON object only."
)

ASSESS_USER = """USE CASE: {use_case}

REQUIREMENTS:
{reqs}

DEAL-BREAKERS (things the buyer does NOT accept):
{breakers}

PRODUCTS (JSON):
{products}

Rules:
- status "met": the data clearly satisfies the requirement.
- status "partial": only partly satisfied or close but not exact.
- status "not_met": the data clearly contradicts or lacks it although the spec is normally stated.
- status "unknown": the data says nothing about it.
- For a deal-breaker, "present" is true only if the data shows the product has it, false if the data shows it does not, null if unclear.
- "evidence" is at most 20 words and refers to the provided data.

Return JSON:
{{"products": [{{"id": "P1",
  "requirements": [{{"id": "R1", "status": "met|partial|not_met|unknown", "evidence": "..."}}],
  "deal_breakers": [{{"id": "D1", "present": true, "evidence": "..."}}]}}]}}
Include every product id and every requirement id."""

NARRATE_SYSTEM = (
    "You are an impartial buying advisor. Explain results using ONLY the facts supplied. "
    "Never recommend a product that is marked disqualified. Reply with one JSON object only."
)

NARRATE_USER = """USE CASE: {use_case}

COMPUTED RANKING (already calculated, do not change it):
{ranking}

Eligible winner id: {winner}

Return JSON:
{{
  "headline": "one sentence verdict",
  "recommendation": "2-4 sentences. If winner is null, say no product satisfies all constraints, name the closest and what to relax.",
  "trade_offs": [{{"title": "Product A vs Product B", "text": "what you gain and give up"}}],
  "best_for": {{"P1": "who or what this product suits"}},
  "watch_outs": ["important caveats, including must-haves that are unknown and need verification"]
}}
Keep every string concise. Mention only facts present in the ranking data."""


def clean_extraction(raw: dict, fallback_name: str) -> dict:
    specs = raw.get("specs") if isinstance(raw.get("specs"), dict) else {}
    clean_specs = {}
    for key, value in list(specs.items())[:40]:
        if value is None or str(value).strip() == "":
            continue
        clean_specs[str(key).strip()] = str(value).strip()

    price = raw.get("price_value")
    try:
        price = float(str(price).replace(",", "")) if price not in (None, "") else None
    except Exception:
        price = None

    def as_list(value):
        return [str(v).strip() for v in value if str(v).strip()][:8] if isinstance(value, list) else []

    return {
        "doc_product_name": str(raw.get("product_name") or fallback_name),
        "brand": raw.get("brand"),
        "price": price,
        "currency": (str(raw.get("currency")).strip() if raw.get("currency") else ""),
        "summary": str(raw.get("summary") or ""),
        "specs": clean_specs,
        "strengths": as_list(raw.get("strengths")),
        "limitations": as_list(raw.get("limitations")),
    }


def norm_status(value) -> str:
    s = str(value or "").lower().replace(" ", "_").replace("-", "_")
    if s in ("met", "yes", "pass", "satisfied", "fully_met"):
        return "met"
    if s in ("partial", "partially_met", "partly_met", "partially"):
        return "partial"
    if s in ("not_met", "unmet", "no", "fail", "missing", "not_satisfied"):
        return "not_met"
    return "unknown"


def norm_bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "yes", "present"):
        return True
    if s in ("false", "no", "absent"):
        return False
    return None


def compute_results(products, reqs, breakers, budget, currency, assessment_raw) -> dict:
    """Deterministic scoring and constraint checking (the 'constrained' part)."""
    lookup = {}
    items = assessment_raw.get("products") if isinstance(assessment_raw, dict) else []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            lookup[str(item.get("id"))] = item

    for p in products:
        item = lookup.get(p["id"], {})
        req_map, db_map = {}, {}
        for r in item.get("requirements", []) if isinstance(item.get("requirements"), list) else []:
            if isinstance(r, dict):
                req_map[str(r.get("id"))] = {"status": norm_status(r.get("status")), "evidence": str(r.get("evidence") or "")}
        for d in item.get("deal_breakers", []) if isinstance(item.get("deal_breakers"), list) else []:
            if isinstance(d, dict):
                db_map[str(d.get("id"))] = {"present": norm_bool(d.get("present")), "evidence": str(d.get("evidence") or "")}

        total = got = 0.0
        reasons, checks = [], []
        p["req"], p["db"] = {}, {}
        for r in reqs:
            info = req_map.get(r["id"], {"status": "unknown", "evidence": "Not assessed by the model."})
            weight = 3 if r["type"] == "must" else 1
            total += weight
            got += weight * {"met": 1.0, "partial": 0.5}.get(info["status"], 0.0)
            p["req"][r["id"]] = info
            if r["type"] == "must" and info["status"] == "not_met":
                reasons.append(f"Fails must-have: {r['text']}")
            if r["type"] == "must" and info["status"] == "unknown":
                checks.append(f"Unconfirmed must-have: {r['text']}")

        for b in breakers:
            info = db_map.get(b["id"], {"present": None, "evidence": "Not assessed by the model."})
            p["db"][b["id"]] = info
            if info["present"] is True:
                reasons.append(f"Has deal-breaker: {b['text']}")

        p["budget_note"] = ""
        if budget and p["price"] is not None:
            same_currency = (not currency) or (not p["currency"]) or currency.lower() == p["currency"].lower()
            if same_currency and p["price"] > budget:
                reasons.append(f"Price {p['price']:,.0f} is above the budget of {budget:,.0f}")
            elif not same_currency:
                p["budget_note"] = f"Price is in {p['currency']}, budget is in {currency}: not compared."
        elif budget and p["price"] is None:
            checks.append("Price not found in the document")

        p["score"] = round(100 * got / total) if total else 0
        p["reasons"], p["checks"] = reasons, checks
        p["status"] = "disqualified" if reasons else ("conditional" if checks else "eligible")

    eligible = [p for p in products if p["status"] != "disqualified"]
    eligible.sort(key=lambda x: (-x["score"], len(x["checks"])))
    disq = sorted([p for p in products if p["status"] == "disqualified"], key=lambda x: -x["score"])
    ranking = [p["id"] for p in eligible + disq]
    return {"ranking": ranking, "winner": eligible[0]["id"] if eligible else None}


def fallback_narrative(result: dict) -> dict:
    by_id = {p["id"]: p for p in result["products"]}
    winner = by_id.get(result["winner"]) if result["winner"] else None
    if winner:
        rec = f"{winner['name']} scores highest ({winner['score']}%) among the products that meet every hard constraint."
        headline = f"Best match: {winner['name']}"
    else:
        closest = by_id[result["ranking"][0]]
        rec = f"No product satisfies every must-have, deal-breaker and budget limit. {closest['name']} is the closest ({closest['score']}%)."
        headline = "No product meets all constraints"
    return {"headline": headline, "recommendation": rec, "trade_offs": [], "best_for": {}, "watch_outs": []}


def run_analysis(api_key, model, temperature, max_chars, docs, reqs, breakers, use_case, budget, currency, progress):
    if Groq is None:
        raise LLMError("The 'groq' package is not installed. Run: pip install -r requirements.txt")
    client = Groq(api_key=api_key.strip(), timeout=90.0, max_retries=1)
    cache = st.session_state.setdefault("extract_cache", {})

    products = []
    total_steps = len(docs) + 2
    for i, doc in enumerate(docs):
        progress.progress(i / total_steps, text=f"Reading {doc['name']} ({i + 1} of {len(docs)})")
        text = doc["text"][:max_chars]
        key = hashlib.sha256(f"{model}|{max_chars}|{doc['name']}|{text}".encode("utf-8", "ignore")).hexdigest()
        if key not in cache:
            raw = call_llm(client, model, EXTRACT_SYSTEM, EXTRACT_USER.format(name=doc["name"], text=text), temperature, 2000)
            cache[key] = clean_extraction(raw, doc["name"])
        data = dict(cache[key])
        data.update({"id": f"P{i + 1}", "name": doc["name"], "truncated": len(doc["text"]) > max_chars})
        products.append(data)

    progress.progress(len(docs) / total_steps, text="Checking each product against your requirements")
    req_text = "\n".join(f"{r['id']} [{r['type']}-have] {r['text']}" for r in reqs)
    db_text = "\n".join(f"{b['id']} {b['text']}" for b in breakers) or "None"
    compact = [
        {
            "id": p["id"], "name": p["name"], "price": p["price"], "currency": p["currency"],
            "summary": p["summary"], "specs": p["specs"], "strengths": p["strengths"], "limitations": p["limitations"],
        }
        for p in products
    ]
    assessment = call_llm(
        client, model, ASSESS_SYSTEM,
        ASSESS_USER.format(use_case=use_case or "Not specified", reqs=req_text, breakers=db_text,
                           products=json.dumps(compact, ensure_ascii=False)),
        0.1, 4000,
    )
    outcome = compute_results(products, reqs, breakers, budget, currency, assessment)

    result = {
        "products": products, "requirements": reqs, "breakers": breakers, "budget": budget,
        "currency": currency, "use_case": use_case, "model": model, **outcome,
    }

    progress.progress((len(docs) + 1) / total_steps, text="Writing the explanation")
    by_id = {p["id"]: p for p in products}
    ranking_data = [
        {
            "id": pid, "name": by_id[pid]["name"], "score_percent": by_id[pid]["score"], "status": by_id[pid]["status"],
            "disqualified_because": by_id[pid]["reasons"], "needs_verification": by_id[pid]["checks"],
            "price": by_id[pid]["price"], "currency": by_id[pid]["currency"],
            "met": [r["text"] for r in reqs if by_id[pid]["req"][r["id"]]["status"] == "met"],
            "partial": [r["text"] for r in reqs if by_id[pid]["req"][r["id"]]["status"] == "partial"],
            "not_met": [r["text"] for r in reqs if by_id[pid]["req"][r["id"]]["status"] == "not_met"],
            "strengths": by_id[pid]["strengths"][:4], "limitations": by_id[pid]["limitations"][:4],
        }
        for pid in result["ranking"]
    ]
    try:
        narrative = call_llm(
            client, model, NARRATE_SYSTEM,
            NARRATE_USER.format(use_case=use_case or "Not specified", ranking=json.dumps(ranking_data, ensure_ascii=False),
                                winner=json.dumps(result["winner"])),
            temperature, 2500,
        )
    except LLMError:
        narrative = fallback_narrative(result)
    if not isinstance(narrative, dict) or not narrative.get("recommendation"):
        narrative = fallback_narrative(result)
    result["narrative"] = narrative
    progress.progress(1.0, text="Done")
    return result


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
STATUS_LABEL = {"met": ("Met", "ok"), "partial": ("Partial", "warn"), "not_met": ("Not met", "bad"), "unknown": ("Unknown", "unk")}


def tag(text, kind):
    return f'<span class="tag tag-{kind}">{esc(text)}</span>'


def status_tag(product):
    return {
        "eligible": tag("Eligible", "ok"),
        "conditional": tag("Eligible, needs checks", "warn"),
        "disqualified": tag("Disqualified", "bad"),
    }[product["status"]]


def render_recommendation(result):
    by_id = {p["id"]: p for p in result["products"]}
    nar = result["narrative"]
    winner = by_id.get(result["winner"]) if result["winner"] else None

    if winner:
        st.markdown(
            f'<div class="card winner">{tag("Recommended", "acc")}'
            f'<h3>{esc(nar.get("headline") or winner["name"])}</h3>'
            f'<p>{esc(nar.get("recommendation", ""))}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="card">{tag("No product qualifies", "bad")}'
            f'<h3>{esc(nar.get("headline") or "No product meets every constraint")}</h3>'
            f'<p>{esc(nar.get("recommendation", ""))}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Ranking")
    for position, pid in enumerate(result["ranking"], start=1):
        p = by_id[pid]
        dq = p["status"] == "disqualified"
        extra = ""
        for reason in p["reasons"]:
            extra += f'<p class="reason">✕ {esc(reason)}</p>'
        for check in p["checks"]:
            extra += f'<p class="note">⚠ {esc(check)}</p>'
        if p["budget_note"]:
            extra += f'<p class="note">{esc(p["budget_note"])}</p>'
        st.markdown(
            f'<div class="card"><div class="rank-row"><span class="rank-name">{position}. {esc(p["name"])}</span>'
            f'{status_tag(p)}<div class="bar"><div class="fill {"dq" if dq else ""}" style="width:{p["score"]}%"></div></div>'
            f'<span class="score">{p["score"]}%</span></div>{extra}</div>',
            unsafe_allow_html=True,
        )
    st.caption("Score weights must-haves ×3 and nice-to-haves ×1. Any failed must-have, deal-breaker or budget limit disqualifies a product.")


def comparison_table_html(result) -> str:
    products = result["products"]
    head = "<tr><th>Requirement</th>" + "".join(f"<th>{esc(p['name'])}</th>" for p in products) + "</tr>"
    rows = []

    def group(label):
        rows.append(f'<tr class="group"><td colspan="{len(products) + 1}">{esc(label)}</td></tr>')

    def req_rows(kind, label):
        subset = [r for r in result["requirements"] if r["type"] == kind]
        if not subset:
            return
        group(label)
        for r in subset:
            cells = ""
            for p in products:
                info = p["req"][r["id"]]
                text, cls = STATUS_LABEL[info["status"]]
                cells += f'<td>{tag(text, cls)}<span class="ev">{esc(info["evidence"])}</span></td>'
            rows.append(f'<tr><td class="req">{esc(r["text"])}</td>{cells}</tr>')

    req_rows("must", "Must-have requirements")
    req_rows("nice", "Nice-to-have requirements")

    if result["breakers"]:
        group("Deal-breakers")
        for b in result["breakers"]:
            cells = ""
            for p in products:
                info = p["db"][b["id"]]
                if info["present"] is True:
                    badge = tag("Present", "bad")
                elif info["present"] is False:
                    badge = tag("Absent", "ok")
                else:
                    badge = tag("Unclear", "unk")
                cells += f'<td>{badge}<span class="ev">{esc(info["evidence"])}</span></td>'
            rows.append(f'<tr><td class="req">{esc(b["text"])}</td>{cells}</tr>')

    group("Price and overall")
    def price_text(p):
        if p["price"] is None:
            return "Not stated"
        return ("%s %s" % (p["currency"], format(p["price"], ",.2f"))).strip()

    price_cells = "".join("<td>%s</td>" % esc(price_text(p)) for p in products)
    rows.append(f'<tr><td class="req">Price</td>{price_cells}</tr>')
    total_cells = "".join(f"<td>{p['score']}%<span class='ev'>{esc(p['status'].title())}</span></td>" for p in products)
    rows.append(f'<tr class="total"><td class="req">Weighted score</td>{total_cells}</tr>')

    return f'<div class="table-wrap"><table class="cmp">{head}{"".join(rows)}</table></div>'


def specs_frame(result) -> pd.DataFrame:
    first_seen, data = {}, {}
    for p in result["products"]:
        for key, value in p["specs"].items():
            norm = re.sub(r"\s+", " ", key.strip().lower())
            first_seen.setdefault(norm, key.strip())
            data.setdefault(norm, {})[p["name"]] = value
    rows = []
    for norm, values in data.items():
        row = {"Specification": first_seen[norm]}
        for p in result["products"]:
            row[p["name"]] = values.get(p["name"], "—")
        rows.append(row)
    columns = ["Specification"] + [p["name"] for p in result["products"]]
    return pd.DataFrame(rows, columns=columns)


def summary_frame(result) -> pd.DataFrame:
    rows = []
    for r in result["requirements"]:
        row = {"Requirement": r["text"], "Type": "Must-have" if r["type"] == "must" else "Nice-to-have"}
        for p in result["products"]:
            info = p["req"][r["id"]]
            row[p["name"]] = f"{STATUS_LABEL[info['status']][0]}: {info['evidence']}"
        rows.append(row)
    return pd.DataFrame(rows)


def render_tradeoffs(result):
    nar = result["narrative"]
    by_id = {p["id"]: p for p in result["products"]}
    trade_offs = nar.get("trade_offs") if isinstance(nar.get("trade_offs"), list) else []
    if trade_offs:
        st.markdown("### Trade-offs")
        for item in trade_offs:
            if isinstance(item, dict):
                st.markdown(f'<div class="card"><h3>{esc(item.get("title", ""))}</h3><p>{esc(item.get("text", ""))}</p></div>', unsafe_allow_html=True)
    best_for = nar.get("best_for") if isinstance(nar.get("best_for"), dict) else {}
    if best_for:
        st.markdown("### Best suited for")
        for pid, text in best_for.items():
            if pid in by_id:
                st.markdown(f'<div class="card"><h3>{esc(by_id[pid]["name"])}</h3><p>{esc(text)}</p></div>', unsafe_allow_html=True)
    watch = nar.get("watch_outs") if isinstance(nar.get("watch_outs"), list) else []
    if watch:
        st.markdown("### Verify before you buy")
        st.markdown("\n".join(f"- {w}" for w in watch if w))
    if not (trade_offs or best_for or watch):
        st.info("No extra trade-off notes were generated. The comparison table shows the full detail.")


def build_markdown_report(result) -> str:
    by_id = {p["id"]: p for p in result["products"]}
    nar = result["narrative"]
    lines = ["# Product comparison report", ""]
    if result.get("use_case"):
        lines += [f"**Use case:** {result['use_case']}", ""]
    lines += [f"## {nar.get('headline', 'Recommendation')}", "", str(nar.get("recommendation", "")), "", "## Ranking", ""]
    for i, pid in enumerate(result["ranking"], 1):
        p = by_id[pid]
        lines.append(f"{i}. **{p['name']}** - {p['score']}% - {p['status']}")
        lines += [f"   - {r}" for r in p["reasons"]] + [f"   - Check: {c}" for c in p["checks"]]
    lines += ["", "## Requirement details", ""]
    for r in result["requirements"]:
        lines.append(f"### {r['text']} ({'must-have' if r['type'] == 'must' else 'nice-to-have'})")
        for p in result["products"]:
            info = p["req"][r["id"]]
            lines.append(f"- {p['name']}: {STATUS_LABEL[info['status']][0]} - {info['evidence']}")
        lines.append("")
    trade_offs = nar.get("trade_offs") if isinstance(nar.get("trade_offs"), list) else []
    if trade_offs:
        lines += ["## Trade-offs", ""]
        lines += [f"- **{t.get('title', '')}**: {t.get('text', '')}" for t in trade_offs if isinstance(t, dict)]
    return "\n".join(lines)


def render_results(result):
    st.markdown(
        '<div class="section-head"><h2>Results</h2><p>Every statement comes from your uploaded documents.</p></div>',
        unsafe_allow_html=True,
    )
    tab_rec, tab_cmp, tab_trade, tab_specs, tab_export = st.tabs(
        ["Recommendation", "Comparison table", "Trade-offs", "Extracted specs", "Export"]
    )
    with tab_rec:
        render_recommendation(result)
    with tab_cmp:
        st.markdown(comparison_table_html(result), unsafe_allow_html=True)
    with tab_trade:
        render_tradeoffs(result)
    with tab_specs:
        for p in result["products"]:
            if p.get("truncated"):
                st.warning(f"{p['name']}: the document was long, so only the first part was analysed. Raise 'characters per document' in the sidebar to read more.")
        frame = specs_frame(result)
        if frame.empty:
            st.info("No specifications were extracted from the documents.")
        else:
            st.dataframe(frame, use_container_width=True, hide_index=True)
    with tab_export:
        st.markdown("Download the results to share or keep.")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download report (.md)", build_markdown_report(result), "comparison_report.md", "text/markdown")
        c2.download_button("Download table (.csv)", summary_frame(result).to_csv(index=False), "comparison_table.csv", "text/csv")
        c3.download_button("Download data (.json)", json.dumps(result, indent=2, default=str), "comparison_data.json", "application/json")


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def sidebar():
    with st.sidebar:
        st.markdown("## Settings")
        api_key = st.text_input(
            "Groq API key", value=default_api_key(), type="password",
            help="Get a free key at console.groq.com. You can also set GROQ_API_KEY as an environment variable or in .streamlit/secrets.toml.",
        )
        model = st.selectbox("Model", MODELS, index=0, help="The 70B model gives the best accuracy. The 8B model is faster and uses fewer tokens.")
        custom = st.text_input("Custom model name (optional)", placeholder="Overrides the choice above")
        temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.05, help="Keep this low for factual comparisons.")
        max_chars = st.slider("Characters read per document", 3000, 24000, 9000, 1000,
                              help="Longer documents are cut to this size. Lower it if you hit rate limits.")
        st.markdown("---")
        st.markdown("**Supported files**")
        st.caption("PDF, Word (.docx), Excel (.xlsx), CSV, TXT, Markdown and JSON.")
        st.caption("Scanned PDFs without selectable text cannot be read.")
    return api_key, (custom.strip() or model), temperature, max_chars


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    if Groq is None:
        st.error("The 'groq' package is missing. Run `pip install -r requirements.txt` and restart the app.")
        st.stop()

    api_key, model, temperature, max_chars = sidebar()

    st.markdown('<p class="app-title">Product comparison assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-sub">Upload specification sheets, say what you need, and get a ranked comparison '
        "with the trade-offs explained.</p>"
        '<div class="steps">'
        '<div class="step"><span class="step-n">1</span><div><div class="step-t">Add products</div><div class="step-d">Upload two or more spec documents</div></div></div>'
        '<div class="step"><span class="step-n">2</span><div><div class="step-t">Set requirements</div><div class="step-d">Must-haves, nice-to-haves, limits</div></div></div>'
        '<div class="step"><span class="step-n">3</span><div><div class="step-t">Compare</div><div class="step-d">Table, trade-offs, recommendation</div></div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Step 1: products ------------------------------------------------- #
    st.markdown('<div class="section-head"><h2>1. Add products</h2><p>Each file is treated as one product. You can rename them below.</p></div>', unsafe_allow_html=True)
    with st.container(border=True):
        files = st.file_uploader("Specification documents", type=SUPPORTED_TYPES, accept_multiple_files=True)
        docs, used_names = [], set()

        def unique(name):
            base, n = name, 2
            while name in used_names:
                name, n = f"{base} ({n})", n + 1
            used_names.add(name)
            return name

        for f in files or []:
            try:
                text = parse_document(f.name, f.getvalue())
            except ValueError as exc:
                st.warning(f"{f.name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                st.warning(f"{f.name}: could not be read ({type(exc).__name__}).")
                continue
            c1, c2 = st.columns([3, 1])
            default = os.path.splitext(f.name)[0]
            name = c1.text_input(f"Product name for {f.name}", value=default, key=f"name_{f.name}_{f.size}")
            c2.markdown(f'<p class="note" style="margin-top:2rem">{len(text):,} characters read</p>', unsafe_allow_html=True)
            docs.append({"name": unique(name.strip() or default), "text": text})

        with st.expander("Paste a product's specifications instead"):
            pasted_name = st.text_input("Product name", key="pasted_name", placeholder="e.g. Laptop X14")
            pasted_text = st.text_area("Specifications", key="pasted_text", height=150, placeholder="Paste the product specs here")
            if pasted_text.strip() and len(pasted_text.strip()) >= 15:
                docs.append({"name": unique(pasted_name.strip() or "Pasted product"), "text": pasted_text.strip()})

        if docs:
            st.success(f"{len(docs)} product{'s' if len(docs) != 1 else ''} ready.")

    # ---- Step 2: requirements -------------------------------------------- #
    st.markdown('<div class="section-head"><h2>2. Set your requirements</h2><p>Write one item per line. Be specific, for example "Battery life of at least 10 hours".</p></div>', unsafe_allow_html=True)
    with st.container(border=True):
        use_case = st.text_input("What will you use it for?", placeholder="e.g. Video editing while travelling")
        c1, c2 = st.columns(2)
        must_raw = c1.text_area("Must-have requirements", height=170, placeholder="16 GB RAM\nBattery life of at least 10 hours\nWeighs under 1.5 kg",
                                help="A product that fails any of these is disqualified.")
        nice_raw = c2.text_area("Nice-to-have requirements", height=170, placeholder="Backlit keyboard\nTouchscreen\nThunderbolt port",
                                help="These raise a product's score but never disqualify it.")
        c3, c4, c5 = st.columns([2, 1, 1])
        breaker_raw = c3.text_area("Deal-breakers", height=110, placeholder="Glossy display\nProprietary charger",
                                   help="A product that has any of these is disqualified.")
        budget = c4.number_input("Maximum budget (0 = no limit)", min_value=0.0, value=0.0, step=50.0)
        currency = c5.text_input("Budget currency", placeholder="USD", help="Used to compare against prices found in the documents.")

    reqs = [{"id": f"R{i + 1}", "type": "must", "text": t} for i, t in enumerate(parse_lines(must_raw))]
    reqs += [{"id": f"R{len(reqs) + i + 1}", "type": "nice", "text": t} for i, t in enumerate(parse_lines(nice_raw))]
    breakers = [{"id": f"D{i + 1}", "text": t} for i, t in enumerate(parse_lines(breaker_raw))]

    # ---- Step 3: compare ---------------------------------------------------- #
    st.markdown('<div class="section-head"><h2>3. Compare</h2><p>This takes about a minute for several documents.</p></div>', unsafe_allow_html=True)
    clicked = st.button("Compare products", type="primary")

    if clicked:
        problems = []
        if not api_key.strip():
            problems.append("Enter your Groq API key in the sidebar.")
        if len(docs) < 2:
            problems.append("Add at least two products (upload files or paste specs).")
        if len(docs) > MAX_PRODUCTS:
            problems.append(f"Compare at most {MAX_PRODUCTS} products at a time.")
        if not reqs:
            problems.append("Enter at least one must-have or nice-to-have requirement.")
        if problems:
            for msg in problems:
                st.error(msg)
        else:
            progress = st.progress(0.0, text="Starting")
            try:
                st.session_state["result"] = run_analysis(
                    api_key, model, temperature, max_chars, docs, reqs, breakers,
                    use_case.strip(), float(budget), currency.strip(), progress,
                )
            except LLMError as exc:
                progress.empty()
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                progress.empty()
                st.error(f"Something unexpected went wrong ({type(exc).__name__}: {str(exc)[:200]}). Please try again.")
            else:
                progress.empty()

    if st.session_state.get("result"):
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
