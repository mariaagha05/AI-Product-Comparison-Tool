"""
AI Product Comparison Tool
--------------------------
Upload several product spec documents, describe your requirements, and get:
  * a side-by-side comparison table
  * a requirement-by-requirement fit matrix
  * constrained recommendations (must-haves / budget are enforced in Python)
  * plain-language trade-off explanations

Stack: Streamlit (UI) + Groq (LLM).
Run:   streamlit run app.py
"""

import html
import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Product Comparison Tool",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]
EXCLUDED_MODEL_WORDS = (
    "whisper", "tts", "playai", "guard", "embed", "orpheus", "safeguard", "compound",
)
STATUS_SCORE = {"met": 1.0, "partial": 0.5, "unknown": 0.2, "not_met": 0.0}
STATUS_LABEL = {"met": "Met", "partial": "Partial", "not_met": "Not met", "unknown": "Unknown"}
STATUS_CLASS = {"met": "ok", "partial": "warn", "not_met": "bad", "unknown": "neutral"}
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "csv", "xlsx", "xls", "json"]

# --------------------------------------------------------------------------
# Styling  (high-contrast text, light balanced palette, Inter font w/ fallbacks)
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#F4F6FB; --card:#FFFFFF; --ink:#18202F; --muted:#4A5568; --line:#D9DFEB;
  --brand:#4F46E5; --brand-dark:#3730A3; --brand-soft:#EEF0FF;
  --ok:#0B6B42; --ok-bg:#DFF5E9; --warn:#7A4E00; --warn-bg:#FFF0CF;
  --bad:#A8231A; --bad-bg:#FDE4E1; --neutral:#364152; --neutral-bg:#E8ECF3;
}
html, body, .stApp, button, input, textarea, select{
  font-family:'Inter',-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif !important;
}
.stApp{background:var(--bg) !important; color:var(--ink) !important;}
.block-container{padding-top:2rem !important; padding-bottom:3rem !important; max-width:1280px !important;}
header[data-testid="stHeader"]{background:transparent !important;}
header[data-testid="stHeader"] *{color:var(--ink) !important;}

/* Text visibility */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
label, label *{color:var(--ink) !important; font-size:16px;}
[data-testid="stCaptionContainer"], .stCaption{color:var(--muted) !important; font-size:14.5px !important;}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{color:var(--ink) !important; font-weight:700;}

/* Sidebar */
section[data-testid="stSidebar"]{background:#FFFFFF !important; border-right:1px solid var(--line);}
section[data-testid="stSidebar"] *{color:var(--ink) !important;}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{font-weight:700;}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input{
  background:#FFFFFF !important; color:var(--ink) !important; font-size:16px !important;
  border:1.5px solid var(--line) !important; border-radius:10px !important;
  -webkit-text-fill-color:var(--ink) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus{
  border-color:var(--brand) !important; box-shadow:0 0 0 3px rgba(79,70,229,.15) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder{color:#7A8499 !important; -webkit-text-fill-color:#7A8499 !important;}
div[data-baseweb="select"] > div{background:#FFFFFF !important; border:1.5px solid var(--line) !important; border-radius:10px !important;}
div[data-baseweb="select"] *{color:var(--ink) !important; font-size:15.5px !important;}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"]{background:#FFFFFF !important;}
div[data-baseweb="popover"] *{color:var(--ink) !important;}

/* Buttons */
.stButton > button, .stDownloadButton > button{
  background:var(--brand) !important; color:#FFFFFF !important; border:none !important;
  border-radius:10px !important; padding:.65rem 1.4rem !important; font-weight:600 !important; font-size:16px !important;
  box-shadow:0 2px 6px rgba(79,70,229,.25);
}
.stButton > button *, .stDownloadButton > button *{color:#FFFFFF !important;}
.stButton > button:hover, .stDownloadButton > button:hover{background:var(--brand-dark) !important;}

/* File uploader */
[data-testid="stFileUploaderDropzone"]{background:#FFFFFF !important; border:2px dashed #AEB7D0 !important; border-radius:14px !important;}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink) !important;}
[data-testid="stFileUploaderDropzone"] button{background:var(--brand-soft) !important; border:1px solid var(--brand) !important; box-shadow:none !important;}
[data-testid="stFileUploaderDropzone"] button *{color:var(--brand-dark) !important;}
[data-testid="stFileUploaderFile"] *{color:var(--ink) !important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:6px; border-bottom:2px solid var(--line);}
.stTabs [data-baseweb="tab"]{font-size:16.5px !important; font-weight:600 !important; padding:10px 18px !important; color:var(--muted) !important;}
.stTabs [data-baseweb="tab"] *{color:var(--muted) !important;}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] *{color:var(--brand) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--brand) !important; height:3px !important;}

/* Expander / alerts / progress */
[data-testid="stExpander"]{background:#FFFFFF !important; border:1px solid var(--line) !important; border-radius:12px !important;}
[data-testid="stExpander"] *{color:var(--ink) !important;}
[data-testid="stAlert"]{border-radius:12px !important;}
[data-testid="stAlert"] *{color:var(--ink) !important; font-size:15.5px !important;}
.stProgress > div > div > div > div{background:var(--brand) !important;}

/* Hero */
.hero{background:linear-gradient(135deg,#4F46E5 0%,#6D5EF0 55%,#3B82F6 100%); border-radius:20px; padding:28px 32px; margin-bottom:22px;
  box-shadow:0 8px 24px rgba(79,70,229,.22);}
.hero h1{color:#FFFFFF !important; font-size:2.1rem; font-weight:800; margin:0 0 6px 0; letter-spacing:-.5px;}
.hero p{color:#EEF0FF !important; font-size:1.05rem; margin:0; line-height:1.5;}
.hero .steps{margin-top:16px; display:flex; gap:10px; flex-wrap:wrap;}
.hero .step{background:rgba(255,255,255,.18); color:#FFFFFF !important; border-radius:999px; padding:6px 14px; font-size:.92rem; font-weight:600;}

/* Cards */
.card{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px 22px; margin-bottom:16px; box-shadow:0 1px 3px rgba(20,30,60,.05);}
.card h3{margin:0 0 6px 0; font-size:1.15rem; color:var(--ink) !important;}
.card p, .card li, .card div{color:var(--ink); font-size:16px; line-height:1.6;}
.card .sub{color:var(--muted) !important; font-size:14.5px;}
.section-title{font-size:1.25rem; font-weight:700; color:var(--ink) !important; margin:6px 0 4px 0;}
.section-sub{color:var(--muted) !important; font-size:15px; margin-bottom:12px;}

/* Product cards */
.pcard{background:#FFFFFF; border:1.5px solid var(--line); border-radius:16px; padding:18px 20px; margin-bottom:14px; height:100%;}
.pcard.best{border-color:var(--ok); box-shadow:0 0 0 3px rgba(11,107,66,.12);}
.pcard.out{background:#FBF3F2; border-color:#EBC3BF;}
.pcard .pname{font-weight:700; font-size:1.08rem; color:var(--ink) !important; margin-bottom:6px; word-break:break-word;}
.pcard .pscore{font-size:2.2rem; font-weight:800; color:var(--brand-dark) !important; line-height:1.1;}
.pcard .pscore span{font-size:1rem; color:var(--muted) !important; font-weight:600;}
.pcard .pmeta{color:var(--ink) !important; font-size:15px; margin-top:8px; line-height:1.65;}
.pcard .viol{margin-top:8px; color:var(--bad) !important; font-size:14.5px; font-weight:500;}

/* Pills */
.pill{display:inline-block; padding:3px 11px; border-radius:999px; font-size:13.5px; font-weight:700; margin:2px 4px 2px 0;}
.pill.ok{background:var(--ok-bg); color:var(--ok) !important;}
.pill.warn{background:var(--warn-bg); color:var(--warn) !important;}
.pill.bad{background:var(--bad-bg); color:var(--bad) !important;}
.pill.neutral{background:var(--neutral-bg); color:var(--neutral) !important;}
.pill.brand{background:var(--brand-soft); color:var(--brand-dark) !important;}

/* Tables */
.tbl-wrap{overflow-x:auto; border:1px solid var(--line); border-radius:14px; background:#FFFFFF; margin-bottom:18px;}
table.cmp{border-collapse:separate; border-spacing:0; width:100%; font-size:15px;}
table.cmp th{background:#EEF0FB; color:var(--ink) !important; text-align:left; padding:13px 14px; font-weight:700; border-bottom:2px solid var(--line); position:sticky; top:0; white-space:nowrap;}
table.cmp td{padding:12px 14px; color:var(--ink) !important; border-bottom:1px solid #EBEEF5; vertical-align:top; line-height:1.5; min-width:130px;}
table.cmp td.feat{font-weight:700; background:#FAFBFE; min-width:170px;}
table.cmp tr:last-child td{border-bottom:none;}
table.cmp td.best{background:var(--ok-bg); font-weight:600;}
table.cmp td .ev{display:block; color:var(--muted) !important; font-size:13px; font-weight:400; margin-top:4px;}

/* Winner banner */
.winner{background:linear-gradient(135deg,#E6F7EE,#F0FBF5); border:1.5px solid #8FD3AF; border-radius:18px; padding:22px 26px; margin-bottom:18px;}
.winner .tag{color:var(--ok) !important; font-weight:700; font-size:.9rem; text-transform:uppercase; letter-spacing:.06em;}
.winner .title{color:var(--ink) !important; font-size:1.7rem; font-weight:800; margin:4px 0 8px 0;}
.winner p{color:var(--ink) !important; font-size:16.5px; line-height:1.65; margin:0;}
.noresult{background:#FFF6E0; border:1.5px solid #F0CD7A; border-radius:18px; padding:22px 26px; margin-bottom:18px;}
.noresult .title{color:var(--ink) !important; font-size:1.4rem; font-weight:800; margin-bottom:6px;}
.noresult p{color:var(--ink) !important; font-size:16.5px; line-height:1.65; margin:0;}
.trade{background:#FFFFFF; border:1px solid var(--line); border-left:5px solid var(--brand); border-radius:12px; padding:14px 18px; margin-bottom:12px;}
.trade .t{font-weight:700; color:var(--ink) !important; font-size:1.02rem; margin-bottom:4px;}
.trade .d{color:var(--ink) !important; font-size:15.5px; line-height:1.6;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def esc(value) -> str:
    """HTML-escape any value safely."""
    return html.escape("" if value is None else str(value), quote=True)


def pill(text, kind="neutral") -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


def get_secret_key() -> str:
    """Read GROQ_API_KEY from env or Streamlit secrets without ever raising."""
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", ""))
    except Exception:
        return ""


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "401" in low or "invalid api key" in low or "invalid_api_key" in low:
        return "Your Groq API key was rejected. Please check it in the sidebar."
    if "429" in low or "rate limit" in low or "rate_limit" in low:
        return ("Groq rate limit reached. Wait a minute and try again, use fewer/shorter documents, "
                "or choose a different model in the sidebar.")
    if "decommission" in low or "model_not_found" in low or "does not exist" in low:
        return "The selected model is not available any more. Please choose another model in the sidebar."
    if "413" in low or "too large" in low or "context length" in low or "reduce your message" in low:
        return ("The documents are too large for this model. Lower 'Max characters per document' "
                "in the sidebar or upload smaller files.")
    if "connection" in low or "timed out" in low or "timeout" in low:
        return "Could not reach Groq. Check your internet connection and try again."
    return f"Something went wrong while talking to the AI model: {msg[:300]}"


def parse_json(text: str) -> dict:
    """Parse JSON from an LLM reply, tolerating code fences and stray text."""
    if not text:
        raise ValueError("The model returned an empty response.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model did not return valid JSON.")
        data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("The model returned an unexpected JSON structure.")
    return data


def norm_status(value) -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in ("met", "yes", "pass", "passed", "satisfied", "fully_met", "true", "full"):
        return "met"
    if s in ("not_met", "no", "fail", "failed", "unmet", "not_satisfied", "false", "missing"):
        return "not_met"
    if s.startswith("part"):
        return "partial"
    return "unknown"


def dedupe_names(names):
    seen, out = {}, []
    for n in names:
        base = (n or "Product").strip()[:60] or "Product"
        if base.lower() in seen:
            seen[base.lower()] += 1
            out.append(f"{base} ({seen[base.lower()]})")
        else:
            seen[base.lower()] = 1
            out.append(base)
    return out


def resolve_name(name, names):
    """Map an LLM-provided product name back to one of our canonical names."""
    if name in names:
        return name
    low = str(name or "").strip().lower()
    for n in names:
        if n.lower() == low:
            return n
    for n in names:
        if low and (low in n.lower() or n.lower() in low):
            return n
    return None


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        try:
            return float(re.sub(r"[^\d.]", "", str(value)))
        except Exception:
            return None


# --------------------------------------------------------------------------
# Document readers
# --------------------------------------------------------------------------
def decode_bytes(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def read_uploaded_file(uploaded) -> str:
    """Extract text from an uploaded file. Raises ValueError with a friendly message."""
    name = uploaded.name
    ext = Path(name).suffix.lower().lstrip(".")
    data = uploaded.getvalue()
    if not data:
        raise ValueError("the file is empty")

    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("the PDF is password-protected")
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        if not text.strip():
            raise ValueError("no readable text found (scanned PDF? OCR is not supported)")
        return text

    if ext == "docx":
        import docx
        document = docx.Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n".join(parts)

    if ext in ("xlsx", "xls"):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        return "\n\n".join(
            f"Sheet: {sn}\n{df.to_csv(index=False)}" for sn, df in sheets.items()
        )

    if ext == "csv":
        text = decode_bytes(data)
        try:
            df = pd.read_csv(io.StringIO(text))
            return df.to_csv(index=False)
        except Exception:
            return text

    if ext == "json":
        text = decode_bytes(data)
        try:
            return json.dumps(json.loads(text), indent=1, ensure_ascii=False)
        except Exception:
            return text

    return decode_bytes(data)  # txt, md, and anything else


# --------------------------------------------------------------------------
# Groq helpers
# --------------------------------------------------------------------------
def make_client(api_key: str):
    if Groq is None:
        raise RuntimeError("The 'groq' package is not installed. Run: pip install -r requirements.txt")
    return Groq(api_key=api_key.strip(), timeout=90.0, max_retries=1)


def fetch_models(api_key: str):
    """Return chat-capable model ids available to this key (or [] on failure)."""
    try:
        client = make_client(api_key)
        ids = [m.id for m in client.models.list().data]
        ids = [i for i in ids if not any(w in i.lower() for w in EXCLUDED_MODEL_WORDS)]
        ordered = [m for m in PREFERRED_MODELS if m in ids]
        ordered += sorted(i for i in ids if i not in ordered)
        return ordered
    except Exception:
        return []


def call_llm(client, model, system, user, temperature=0.2, json_mode=True, max_tokens=4096) -> str:
    last_exc = None
    for attempt in range(3):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            low = str(exc).lower()
            if "429" in low or "rate limit" in low or "rate_limit" in low:
                time.sleep(4 * (attempt + 1))
                continue
            if json_mode and ("json" in low or "response_format" in low):
                json_mode = False  # retry without strict JSON mode, we parse manually
                continue
            raise
    raise last_exc if last_exc else RuntimeError("Unknown LLM error")


def call_json(client, model, system, user, temperature, max_tokens=4096) -> dict:
    """Call the LLM and return parsed JSON, with one repair retry."""
    raw = call_llm(client, model, system, user, temperature, True, max_tokens)
    try:
        return parse_json(raw)
    except Exception:
        raw = call_llm(
            client, model, system,
            user + "\n\nIMPORTANT: Reply with ONE valid JSON object only. No prose, no markdown.",
            0.0, False, max_tokens,
        )
        return parse_json(raw)


# --------------------------------------------------------------------------
# Analysis pipeline
# --------------------------------------------------------------------------
EXTRACT_SYSTEM = (
    "You are a precise product-specification analyst. Extract ONLY facts that appear in the document. "
    "Never invent or guess values; use null when something is not stated. Respond with JSON only."
)

EXTRACT_SCHEMA = """Return JSON with exactly this shape:
{
  "product_name": "string",
  "brand": "string or null",
  "category": "string or null",
  "price_text": "price exactly as written, or null",
  "price_value": number or null,
  "currency": "string or null",
  "specs": {"Normalised feature name": "value with units"},
  "strengths": ["short strings"],
  "limitations": ["short strings"]
}
Keep at most 25 of the most decision-relevant specs."""

COMPARE_SYSTEM = (
    "You are an impartial product-comparison analyst. Judge every product strictly from the supplied "
    "profiles; if a fact is missing, mark it 'unknown' rather than guessing. Respond with JSON only."
)

COMPARE_SCHEMA = """Return JSON with exactly this shape:
{
  "requirements": [
    {"id": "R1", "text": "short requirement", "type": "must" or "nice"}
  ],
  "assessment": {
    "<exact product name>": {
      "R1": {"status": "met" | "partial" | "not_met" | "unknown", "evidence": "<=20 words citing the spec"}
    }
  },
  "feature_table": [
    {"feature": "feature name", "values": {"<exact product name>": "value or '-'"}, "best": "<exact product name or null>"}
  ],
  "tradeoffs": [
    {"title": "short title", "explanation": "2-3 sentences explaining the trade-off between products", "favors": "<exact product name or null>"}
  ]
}
Rules:
- Use the EXACT product names given. Include every product in 'assessment' and in every 'values' map.
- Turn the user's requirements into 3-10 atomic requirements. Hard needs / deal-breakers = "must"; wishes = "nice".
- feature_table: 8-15 rows, features most relevant to the requirements first. 'best' is the product with the better value (null if tied or not comparable).
- tradeoffs: 3-5 genuine trade-offs (what you gain vs. what you give up)."""

NARRATIVE_SYSTEM = (
    "You are a clear, honest buying advisor. Explain the already-computed ranking in plain language. "
    "Do NOT change the ranking or invent facts. Respond with JSON only."
)

NARRATIVE_SCHEMA = """Return JSON with exactly this shape:
{
  "headline": "one sentence verdict",
  "explanation": "3-5 sentences explaining why the top product fits (or why none fully fits) and what the user gives up",
  "best_for": {"<exact product name>": "one sentence: who/what it is best for"},
  "watch_outs": ["short caveat strings, max 5"]
}"""


def extract_profile(client, model, doc_name, text, temperature):
    user = f"Document name: {doc_name}\n\n--- DOCUMENT START ---\n{text}\n--- DOCUMENT END ---\n\n{EXTRACT_SCHEMA}"
    data = call_json(client, model, EXTRACT_SYSTEM, user, temperature, 2500)
    specs = data.get("specs")
    if not isinstance(specs, dict):
        specs = {}
    return {
        "product_name": str(data.get("product_name") or Path(doc_name).stem),
        "brand": data.get("brand"),
        "category": data.get("category"),
        "price_text": data.get("price_text"),
        "price_value": to_float(data.get("price_value")),
        "currency": data.get("currency"),
        "specs": {str(k): str(v) for k, v in list(specs.items())[:25]},
        "strengths": [str(x) for x in (data.get("strengths") or [])][:8] if isinstance(data.get("strengths"), list) else [],
        "limitations": [str(x) for x in (data.get("limitations") or [])][:8] if isinstance(data.get("limitations"), list) else [],
        "source_file": doc_name,
    }


def run_analysis(client, model, temperature, docs, reqs, strict, progress):
    """docs: list of (filename, text). reqs: dict. Returns result dict."""
    total_steps = len(docs) + 2
    step = 0

    # 1) Extract a profile per document
    profiles = []
    for fname, text in docs:
        progress(step / total_steps, f"Reading specifications from {fname} ...")
        profiles.append(extract_profile(client, model, fname, text, temperature))
        step += 1

    names = dedupe_names([p["product_name"] for p in profiles])
    for p, n in zip(profiles, names):
        p["product_name"] = n

    # 2) Compare against requirements
    progress(step / total_steps, "Comparing products against your requirements ...")
    compact = [
        {
            "name": p["product_name"], "brand": p["brand"], "price": p["price_text"],
            "specs": p["specs"], "strengths": p["strengths"], "limitations": p["limitations"],
        }
        for p in profiles
    ]
    req_text = (
        f"Use case: {reqs['use_case'] or '-'}\n"
        f"Budget (max): {reqs['budget'] if reqs['budget'] else 'not specified'}\n"
        f"Must-haves / deal-breakers:\n{reqs['must'] or '-'}\n"
        f"Nice-to-haves:\n{reqs['nice'] or '-'}"
    )
    user = (
        f"PRODUCT PROFILES (JSON):\n{json.dumps(compact, ensure_ascii=False)}\n\n"
        f"USER REQUIREMENTS:\n{req_text}\n\n{COMPARE_SCHEMA}"
    )
    cmp_data = call_json(client, model, COMPARE_SYSTEM, user, temperature, 6000)
    step += 1

    # --- normalise requirements
    requirements = []
    for i, r in enumerate(cmp_data.get("requirements") or []):
        if not isinstance(r, dict) or not str(r.get("text", "")).strip():
            continue
        rtype = "must" if str(r.get("type", "")).lower().startswith("must") else "nice"
        requirements.append({"id": str(r.get("id") or f"R{i + 1}"), "text": str(r["text"]).strip(), "type": rtype})
    if not requirements:
        raise ValueError("The AI could not derive any requirements. Please describe your needs in more detail.")

    # --- normalise assessments
    raw_assess = cmp_data.get("assessment")
    assessment = {n: {} for n in names}
    if isinstance(raw_assess, dict):
        for pname, items in raw_assess.items():
            canon = resolve_name(pname, names)
            if canon and isinstance(items, dict):
                for rid, val in items.items():
                    if isinstance(val, dict):
                        assessment[canon][str(rid)] = {
                            "status": norm_status(val.get("status")),
                            "evidence": str(val.get("evidence") or ""),
                        }
                    else:
                        assessment[canon][str(rid)] = {"status": norm_status(val), "evidence": ""}

    # --- normalise feature table
    rows = []
    for row in cmp_data.get("feature_table") or []:
        if not isinstance(row, dict) or not row.get("feature"):
            continue
        values = {n: "-" for n in names}
        if isinstance(row.get("values"), dict):
            for k, v in row["values"].items():
                canon = resolve_name(k, names)
                if canon:
                    values[canon] = str(v) if v not in (None, "") else "-"
        rows.append({"feature": str(row["feature"]), "values": values, "best": resolve_name(row.get("best"), names)})
    if not rows:  # fall back to raw specs
        all_features = []
        for p in profiles:
            for k in p["specs"]:
                if k not in all_features:
                    all_features.append(k)
        for feat in all_features[:15]:
            rows.append({"feature": feat, "values": {p["product_name"]: p["specs"].get(feat, "-") for p in profiles}, "best": None})

    tradeoffs = []
    for t in cmp_data.get("tradeoffs") or []:
        if isinstance(t, dict) and (t.get("title") or t.get("explanation")):
            tradeoffs.append({
                "title": str(t.get("title") or "Trade-off"),
                "explanation": str(t.get("explanation") or ""),
                "favors": resolve_name(t.get("favors"), names),
            })

    # 3) Deterministic, constrained scoring in Python
    budget = reqs["budget"] or 0
    scores = {}
    for p in profiles:
        n = p["product_name"]
        violations, unverified = [], []
        earned = possible = 0.0
        must_total = must_met = 0
        for r in requirements:
            a = assessment[n].get(r["id"], {"status": "unknown", "evidence": ""})
            weight = 2.0 if r["type"] == "must" else 1.0
            possible += weight
            earned += weight * STATUS_SCORE[a["status"]]
            if r["type"] == "must":
                must_total += 1
                if a["status"] == "met":
                    must_met += 1
                elif a["status"] == "not_met":
                    violations.append(f"Fails must-have: {r['text']}")
                elif a["status"] == "unknown":
                    unverified.append(f"Could not verify: {r['text']}")
        if budget and p["price_value"] is not None and p["price_value"] > budget:
            violations.append(f"Price {p['price_value']:,.2f} exceeds budget {budget:,.2f}")
        elif budget and p["price_value"] is None:
            unverified.append("Price not found in document - budget could not be checked")
        pct = round(100 * earned / possible) if possible else 0
        scores[n] = {
            "score": pct, "must_met": must_met, "must_total": must_total,
            "violations": violations, "unverified": unverified,
            "eligible": (not violations) if strict else True,
            "price_value": p["price_value"], "price_text": p["price_text"],
        }

    def sort_key(name):
        s = scores[name]
        price = s["price_value"] if s["price_value"] is not None else float("inf")
        return (-s["score"], price)

    eligible = sorted([n for n in names if scores[n]["eligible"]], key=sort_key)
    ranked_all = sorted(names, key=sort_key)
    winner = eligible[0] if eligible else None
    closest = ranked_all[0] if ranked_all else None

    # 4) Plain-language narrative (never allowed to change the ranking)
    progress(step / total_steps, "Writing your recommendation ...")
    narrative = {}
    try:
        summary = {
            "ranking_eligible": eligible,
            "closest_if_none_eligible": closest if not eligible else None,
            "products": {
                n: {
                    "score_out_of_100": scores[n]["score"],
                    "eligible": scores[n]["eligible"],
                    "violations": scores[n]["violations"],
                    "unverified": scores[n]["unverified"],
                    "price": scores[n]["price_text"],
                } for n in names
            },
            "requirements": requirements,
            "tradeoffs": tradeoffs,
        }
        user = f"COMPUTED RESULTS (authoritative):\n{json.dumps(summary, ensure_ascii=False)}\n\nUser use case: {reqs['use_case'] or '-'}\n\n{NARRATIVE_SCHEMA}"
        narrative = call_json(client, model, NARRATIVE_SYSTEM, user, temperature, 2000)
    except Exception:
        narrative = {}

    best_for = {}
    if isinstance(narrative.get("best_for"), dict):
        for k, v in narrative["best_for"].items():
            canon = resolve_name(k, names)
            if canon:
                best_for[canon] = str(v)
    watch = narrative.get("watch_outs") if isinstance(narrative.get("watch_outs"), list) else []

    if winner:
        default_head = f"{winner} is the best match for your requirements."
        default_expl = (f"It scores {scores[winner]['score']}/100 and meets "
                        f"{scores[winner]['must_met']} of {scores[winner]['must_total']} must-have requirements.")
    else:
        default_head = "No product satisfies all of your hard constraints."
        default_expl = (f"The closest option is {closest} ({scores[closest]['score']}/100), "
                        "but it breaks at least one must-have or your budget. See the details below."
                        if closest else "There is nothing to recommend.")
    progress(1.0, "Done")

    return {
        "names": names,
        "profiles": profiles,
        "requirements": requirements,
        "assessment": assessment,
        "feature_rows": rows,
        "tradeoffs": tradeoffs,
        "scores": scores,
        "eligible": eligible,
        "winner": winner,
        "closest": closest,
        "ranked_all": ranked_all,
        "headline": str(narrative.get("headline") or default_head),
        "explanation": str(narrative.get("explanation") or default_expl),
        "best_for": best_for,
        "watch_outs": [str(w) for w in watch][:5],
        "strict": strict,
        "budget": budget,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_score_cards(res):
    names = res["ranked_all"]
    for i in range(0, len(names), 3):
        chunk = names[i:i + 3]
        cols = st.columns(len(chunk))
        for col, n in zip(cols, chunk):
            s = res["scores"][n]
            cls = "best" if n == res["winner"] else ("out" if not s["eligible"] else "")
            pills = ""
            if n == res["winner"]:
                pills += pill("Recommended", "ok")
            elif s["eligible"]:
                pills += pill("Eligible", "brand")
            else:
                pills += pill("Disqualified", "bad")
            price = s["price_text"] or (f"{s['price_value']:,.2f}" if s["price_value"] is not None else "not stated")
            viol = "".join(f"<div class='viol'>• {esc(v)}</div>" for v in s["violations"])
            unv = "".join(f"<div class='pmeta' style='font-size:13.5px;color:#7A4E00 !important'>• {esc(u)}</div>" for u in s["unverified"][:2])
            card = (
                f"<div class='pcard {cls}'><div class='pname'>{esc(n)}</div>"
                f"<div class='pscore'>{s['score']}<span> / 100</span></div>"
                f"<div>{pills}</div>"
                f"<div class='pmeta'><b>Price:</b> {esc(price)}<br>"
                f"<b>Must-haves met:</b> {s['must_met']} of {s['must_total']}</div>"
                f"{viol}{unv}</div>"
            )
            col.markdown(card, unsafe_allow_html=True)


def feature_table_html(res) -> str:
    names = res["names"]
    head = "<th>Feature</th>" + "".join(f"<th>{esc(n)}</th>" for n in names)
    body = ""
    for row in res["feature_rows"]:
        cells = f"<td class='feat'>{esc(row['feature'])}</td>"
        for n in names:
            cls = "best" if row["best"] == n else ""
            cells += f"<td class='{cls}'>{esc(row['values'].get(n, '-'))}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<div class='tbl-wrap'><table class='cmp'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def fit_matrix_html(res) -> str:
    names = res["names"]
    head = "<th>Requirement</th><th>Priority</th>" + "".join(f"<th>{esc(n)}</th>" for n in names)
    body = ""
    ordered = sorted(res["requirements"], key=lambda r: 0 if r["type"] == "must" else 1)
    for r in ordered:
        badge = pill("Must-have", "bad") if r["type"] == "must" else pill("Nice-to-have", "neutral")
        cells = f"<td class='feat'>{esc(r['text'])}</td><td>{badge}</td>"
        for n in names:
            a = res["assessment"][n].get(r["id"], {"status": "unknown", "evidence": ""})
            ev = f"<span class='ev'>{esc(a['evidence'])}</span>" if a["evidence"] else ""
            cells += f"<td>{pill(STATUS_LABEL[a['status']], STATUS_CLASS[a['status']])}{ev}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<div class='tbl-wrap'><table class='cmp'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def build_dataframe(res) -> pd.DataFrame:
    data = {"Feature": [r["feature"] for r in res["feature_rows"]]}
    for n in res["names"]:
        data[n] = [r["values"].get(n, "-") for r in res["feature_rows"]]
    return pd.DataFrame(data)


def build_markdown_report(res) -> str:
    def cell(x):
        return str(x).replace("|", "/").replace("\n", " ")

    out = ["# Product Comparison Report", ""]
    out += [f"**Verdict:** {res['headline']}", "", res["explanation"], ""]
    out += ["## Scores", "", "| Product | Score | Must-haves met | Status |", "|---|---|---|---|"]
    for n in res["ranked_all"]:
        s = res["scores"][n]
        status = "Recommended" if n == res["winner"] else ("Eligible" if s["eligible"] else "Disqualified")
        out.append(f"| {cell(n)} | {s['score']}/100 | {s['must_met']}/{s['must_total']} | {status} |")
    out += ["", "## Feature comparison", ""]
    out.append("| Feature | " + " | ".join(cell(n) for n in res["names"]) + " |")
    out.append("|---|" + "---|" * len(res["names"]))
    for r in res["feature_rows"]:
        out.append(f"| {cell(r['feature'])} | " + " | ".join(cell(r["values"].get(n, "-")) for n in res["names"]) + " |")
    out += ["", "## Trade-offs", ""]
    for t in res["tradeoffs"]:
        out.append(f"- **{t['title']}**: {t['explanation']}")
    if res["watch_outs"]:
        out += ["", "## Watch-outs", ""] + [f"- {w}" for w in res["watch_outs"]]
    out += ["", "_Generated by AI from the supplied documents - please verify critical details._"]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input(
        "Groq API key",
        value=get_secret_key(),
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com. The key is only kept in memory for this session.",
    )

    models = []
    if api_key.strip():
        if st.session_state.get("models_for_key") != api_key:
            with st.spinner("Loading available Groq models ..."):
                st.session_state["models"] = fetch_models(api_key)
                st.session_state["models_for_key"] = api_key
        models = st.session_state.get("models", [])
    model_options = models if models else PREFERRED_MODELS
    model = st.selectbox(
        "AI model",
        model_options,
        index=0,
        help="Loaded live from your Groq account, so only currently available models are shown.",
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05, help="Lower = more factual and consistent.")
    max_chars = st.slider("Max characters read per document", 2000, 20000, 6000, 500,
                          help="Lower this if you hit rate-limit or size errors.")
    strict = st.toggle("Strict constraints", value=True,
                       help="ON: products that fail a must-have or exceed the budget are disqualified from the recommendation.")

    with st.expander("How it works"):
        st.markdown(
            "1. Upload 2+ spec documents (PDF, DOCX, TXT, CSV, XLSX, JSON)\n"
            "2. Describe your needs and budget\n"
            "3. AI extracts specs and checks each requirement\n"
            "4. Python enforces your must-haves and ranks the products\n"
            "5. You get tables, trade-offs and a recommendation"
        )
    if st.button("🗑️ Clear results"):
        st.session_state.pop("result", None)
        st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>⚖️ AI Product Comparison Tool</h1>
  <p>Upload product specifications, tell us what you need, and get a clear comparison, honest trade-offs and a recommendation that respects your constraints.</p>
  <div class="steps"><span class="step">1 · Upload specs</span><span class="step">2 · Set requirements</span><span class="step">3 · Compare &amp; decide</span></div>
</div>
""",
    unsafe_allow_html=True,
)

tab_setup, tab_compare, tab_reco = st.tabs(["① Setup", "② Comparison", "③ Recommendation"])

# --------------------------------------------------------------------------
# Tab 1 - Setup
# --------------------------------------------------------------------------
with tab_setup:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("<div class='section-title'>📄 Product documents</div>"
                    "<div class='section-sub'>Upload one file per product (at least two).</div>", unsafe_allow_html=True)
        uploads = st.file_uploader(
            "Upload specification files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads:
            st.caption(f"{len(uploads)} file(s) ready: " + ", ".join(u.name for u in uploads))

        with st.expander("➕ Or paste a product's specs as text"):
            pasted_name = st.text_input("Product name (for pasted text)", placeholder="e.g. Acme Laptop X1")
            pasted_text = st.text_area("Specifications", height=150, placeholder="Paste specs, features, price ...")

    with right:
        st.markdown("<div class='section-title'>🎯 Your requirements</div>"
                    "<div class='section-sub'>The more specific you are, the better the comparison.</div>", unsafe_allow_html=True)
        use_case = st.text_input("What will you use it for?", placeholder="e.g. Video editing and travel")
        must = st.text_area("Must-haves (deal-breakers)", height=110,
                            placeholder="One per line, e.g.\nAt least 16 GB RAM\nWeighs under 1.5 kg")
        nice = st.text_area("Nice-to-haves", height=90,
                            placeholder="One per line, e.g.\nBacklit keyboard\nLong warranty")
        budget = st.number_input("Maximum budget (0 = no limit)", min_value=0.0, value=0.0, step=50.0, format="%.2f")

    run = st.button("🔍 Compare products", type="primary")

    if run:
        problems = []
        if not api_key.strip():
            problems.append("Please enter your Groq API key in the sidebar.")
        if not (use_case.strip() or must.strip() or nice.strip()):
            problems.append("Please describe at least one requirement (use case, must-have or nice-to-have).")

        docs, notes = [], []
        for f in uploads or []:
            try:
                text = read_uploaded_file(f).strip()
                if not text:
                    raise ValueError("no text could be extracted")
                docs.append((f.name, text[:max_chars]))
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Skipped **{f.name}**: {exc}")
        if pasted_text.strip():
            docs.append(((pasted_name.strip() or "Pasted product") , pasted_text.strip()[:max_chars]))

        for n in notes:
            st.warning(n)
        if len(docs) < 2:
            problems.append("Please provide at least two readable product documents to compare.")
        if len(docs) > 8:
            problems.append("Please compare at most 8 products at a time.")

        if problems:
            for p in problems:
                st.error(p)
        else:
            bar = st.progress(0.0, text="Starting ...")
            try:
                client = make_client(api_key)
                result = run_analysis(
                    client, model, temperature, docs,
                    {"use_case": use_case.strip(), "must": must.strip(), "nice": nice.strip(), "budget": float(budget)},
                    strict,
                    lambda frac, msg: bar.progress(min(max(frac, 0.0), 1.0), text=msg),
                )
                st.session_state["result"] = result
                bar.empty()
                st.success("Analysis complete. Open the **② Comparison** and **③ Recommendation** tabs above.")
            except Exception as exc:  # noqa: BLE001
                bar.empty()
                st.error(friendly_error(exc) if not isinstance(exc, ValueError) else str(exc))

# --------------------------------------------------------------------------
# Tab 2 - Comparison
# --------------------------------------------------------------------------
res = st.session_state.get("result")

with tab_compare:
    if not res:
        st.info("No results yet. Upload documents and click **Compare products** in the Setup tab.")
    else:
        st.markdown("<div class='section-title'>Scoreboard</div>"
                    "<div class='section-sub'>Must-haves count double. Disqualified products broke a hard constraint.</div>",
                    unsafe_allow_html=True)
        render_score_cards(res)

        st.markdown("<div class='section-title'>Requirement fit</div>"
                    "<div class='section-sub'>How each product performs against each of your requirements.</div>",
                    unsafe_allow_html=True)
        st.markdown(fit_matrix_html(res), unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Feature comparison</div>"
                    "<div class='section-sub'>Green cells mark the stronger value for that feature.</div>",
                    unsafe_allow_html=True)
        st.markdown(feature_table_html(res), unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download comparison table (CSV)",
            data=build_dataframe(res).to_csv(index=False).encode("utf-8"),
            file_name="product_comparison.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------
# Tab 3 - Recommendation
# --------------------------------------------------------------------------
with tab_reco:
    if not res:
        st.info("No results yet. Run a comparison first.")
    else:
        if res["winner"]:
            st.markdown(
                f"<div class='winner'><div class='tag'>🏆 Recommended</div>"
                f"<div class='title'>{esc(res['winner'])}</div>"
                f"<p><b>{esc(res['headline'])}</b><br>{esc(res['explanation'])}</p></div>",
                unsafe_allow_html=True,
            )
            if len(res["eligible"]) > 1:
                runner = res["eligible"][1]
                st.markdown(
                    f"<div class='card'><h3>🥈 Runner-up: {esc(runner)}</h3>"
                    f"<p class='sub'>Score {res['scores'][runner]['score']}/100 · "
                    f"must-haves met {res['scores'][runner]['must_met']}/{res['scores'][runner]['must_total']}</p></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div class='noresult'><div class='title'>⚠️ No product meets every hard constraint</div>"
                f"<p><b>{esc(res['headline'])}</b><br>{esc(res['explanation'])}</p></div>",
                unsafe_allow_html=True,
            )

        if res["best_for"]:
            st.markdown("<div class='section-title'>Who each product suits</div>", unsafe_allow_html=True)
            for n in res["ranked_all"]:
                if n in res["best_for"]:
                    st.markdown(f"<div class='card'><h3>{esc(n)}</h3><p>{esc(res['best_for'][n])}</p></div>",
                                unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Trade-offs explained</div>"
                    "<div class='section-sub'>What you gain and what you give up with each choice.</div>",
                    unsafe_allow_html=True)
        if res["tradeoffs"]:
            for t in res["tradeoffs"]:
                fav = pill(f"Favors {t['favors']}", "brand") if t["favors"] else ""
                st.markdown(
                    f"<div class='trade'><div class='t'>{esc(t['title'])} {fav}</div>"
                    f"<div class='d'>{esc(t['explanation'])}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No notable trade-offs were identified.")

        if res["watch_outs"]:
            st.markdown("<div class='section-title'>Watch-outs</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'>" + "".join(f"<p>• {esc(w)}</p>" for w in res["watch_outs"]) + "</div>",
                        unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download full report (Markdown)",
            data=build_markdown_report(res).encode("utf-8"),
            file_name="product_comparison_report.md",
            mime="text/markdown",
        )
        st.caption("AI-generated from your documents. Always verify critical specifications before buying.")
