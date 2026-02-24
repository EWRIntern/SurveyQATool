from __future__ import annotations
\
import os
import re
import json
import shutil
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import create_engine, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, sessionmaker, Session
from datetime import datetime

from docx import Document as DocxDocument
from odf.opendocument import load as odt_load
from odf import text as odt_text, teletype

# Optional OpenAI
from openai import OpenAI

APP_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(APP_ROOT, "data"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))
DB_PATH = os.getenv("APP_DB_PATH", os.path.join(DATA_DIR, "app.db"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CANONICAL_SIM_THRESHOLD = float(os.getenv("CANONICAL_SIM_THRESHOLD", "0.86"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# DB
# -------------------------
class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), default="default")
    title: Mapped[str] = mapped_column(String(512))
    year: Mapped[int] = mapped_column(Integer)
    audience: Mapped[str] = mapped_column(String(32))  # member|voter
    filename: Mapped[str] = mapped_column(String(512))
    storage_path: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Canonical(Base):
    __tablename__ = "canonical_questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), default="default")
    audience: Mapped[str] = mapped_column(String(32))  # member|voter (strict isolation)
    label: Mapped[str] = mapped_column(String(512))
    appearance_count: Mapped[int] = mapped_column(Integer, default=0)

class Question(Base):
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    tenant_id: Mapped[str] = mapped_column(String(128), default="default")
    canonical_id: Mapped[Optional[int]] = mapped_column(ForeignKey("canonical_questions.id"), nullable=True)

    question_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    question_text: Mapped[str] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(64))

    answer_options_json: Mapped[str] = mapped_column(Text, default="[]")
    matrix_rows_json: Mapped[str] = mapped_column(Text, default="[]")
    matrix_cols_json: Mapped[str] = mapped_column(Text, default="[]")

    topics_json: Mapped[str] = mapped_column(Text, default="[]")

    source_locator: Mapped[str] = mapped_column(String(512))
    source_quote: Mapped[str] = mapped_column(Text)

    audience: Mapped[str] = mapped_column(String(32))
    year: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(512))

class Embedding(Base):
    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"))
    tenant_id: Mapped[str] = mapped_column(String(128), default="default")
    vector_json: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(128))

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# Utils
# -------------------------
def clean_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def jdumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

def jloads(s: str, default):
    try:
        return json.loads(s)
    except Exception:
        return default

def cosine(a, b) -> float:
    import math
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def has_openai() -> bool:
    return bool(OPENAI_API_KEY)

def openai_client() -> OpenAI:
    return OpenAI()

# -------------------------
# Extraction (DOCX + ODT)
# -------------------------

# -------------------------
# Robust DOCX paragraph extraction (includes textboxes + auto-numbering)
# -------------------------
QUESTION_START_RE = re.compile(r"^\s*(?P<code>\d+|[A-Z])\s*[\.\)\:]\s+(?P<text>.+\S)\s*$")

# Index-based question start (ONLY these create new questions)
QUESTION_INDEX_RE = re.compile(r"^\s*(?P<code>\d+|[A-Z])\s*[\.)\:]\s+(?P<text>.+\S)\s*$")
# Lowercase letters are usually matrix rows
MATRIX_ROW_INDEX_RE = re.compile(r"^\s*(?:\[\s*\])?\s*(?P<label>[a-z])\s*\.\s+")
# Option lines: "Very satisfied<TAB>1" or "Very satisfied  1"
OPTION_RE = re.compile(r"^\s*.+?(?:\t+|\s{2,})\d{1,4}\s*$")
# Attached option codes like "Yes1"
ATTACHED_OPTION_RE = re.compile(r"^\s*.+?\D\d{1,2}\s*$")
MATRIX_ROW_RE = re.compile(r"^\s*(?:\[\s*\])?\s*(?P<label>[a-z])\.\s+(?P<rest>.+)$")
# tab or multi-space separated option + numeric code (Very satisfied\t1)
OPTION_LINE_RE = re.compile(r"^\s*(?P<text>.+?)\s*(?:\t+|\s{2,})\s*(?P<code>\d{1,4})\s*$")
# attached numeric code (Yes1, No2)
ATTACHED_CODE_RE = re.compile(r"^\s*(?P<text>.*?\D)\s*(?P<code>\d{1,4})\s*$")
# matrix row fallback like "Joe Biden1234589"
MATRIX_ROW_FALLBACK_RE = re.compile(r"^\s*(?P<text>.+?)(?P<digits>\d{4,})\s*$")
# option prefix like "1) Very satisfied"
OPTION_PREFIX_RE = re.compile(r"^\s*(?P<code>\d+|[A-Za-z])[\)\.\-]\s+(?P<text>\S.+)$")

def docx_all_paragraph_text(path: str) -> List[str]:
    """
    Extract paragraph-level text from DOCX XML directly (includes text boxes/shapes),
    and reconstruct Word auto-numbering prefixes (1., 2., A., B.) when numbering is
    stored as metadata (w:numPr) rather than typed into the text.
    """
    import zipfile
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    # numId -> abstractNumId; abstractNumId -> {ilvl: {numFmt, lvlText}}
    num_to_abs = {}
    abs_lvls = defaultdict(dict)

    def parse_numbering(z: zipfile.ZipFile):
        if "word/numbering.xml" not in z.namelist():
            return
        root = ET.fromstring(z.read("word/numbering.xml"))

        for absn in root.findall(".//w:abstractNum", ns):
            abs_id = absn.get(f"{{{ns['w']}}}abstractNumId")
            if abs_id is None:
                continue
            for lvl in absn.findall("./w:lvl", ns):
                ilvl = lvl.get(f"{{{ns['w']}}}ilvl")
                if ilvl is None:
                    continue
                fmt_el = lvl.find("./w:numFmt", ns)
                txt_el = lvl.find("./w:lvlText", ns)
                numFmt = fmt_el.get(f"{{{ns['w']}}}val") if fmt_el is not None else None
                lvlText = txt_el.get(f"{{{ns['w']}}}val") if txt_el is not None else None
                abs_lvls[str(abs_id)][str(ilvl)] = {"numFmt": numFmt, "lvlText": lvlText}

        for num in root.findall(".//w:num", ns):
            numId = num.get(f"{{{ns['w']}}}numId")
            abs_el = num.find("./w:abstractNumId", ns)
            abs_id = abs_el.get(f"{{{ns['w']}}}val") if abs_el is not None else None
            if numId and abs_id:
                num_to_abs[str(numId)] = str(abs_id)

    def format_counter(n: int, numFmt: str | None) -> str:
        if not numFmt or numFmt == "decimal":
            return str(n)
        if numFmt in ("upperLetter", "upper-letter"):
            n0 = max(1, n)
            out = ""
            while n0 > 0:
                n0, rem = divmod(n0 - 1, 26)
                out = chr(65 + rem) + out
            return out
        if numFmt in ("lowerLetter", "lower-letter"):
            n0 = max(1, n)
            out = ""
            while n0 > 0:
                n0, rem = divmod(n0 - 1, 26)
                out = chr(97 + rem) + out
            return out
        return str(n)

    def apply_lvl_text(counter_str: str, lvlText: str | None) -> str:
        if not lvlText:
            return counter_str + "."
        return re.sub(r"%\d+", counter_str, lvlText)

    def paras_from_xml(xml_bytes: bytes, counters) -> List[str]:
        root = ET.fromstring(xml_bytes)
        out: List[str] = []
        for p in root.findall(".//w:p", ns):
            texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
            s = clean_ws("".join(texts)) if texts else ""

            # numbering prefix
            numPr = p.find("./w:pPr/w:numPr", ns)
            prefix = ""
            if numPr is not None:
                numId_el = numPr.find("./w:numId", ns)
                ilvl_el = numPr.find("./w:ilvl", ns)
                numId = numId_el.get(f"{{{ns['w']}}}val") if numId_el is not None else None
                ilvl = ilvl_el.get(f"{{{ns['w']}}}val") if ilvl_el is not None else None
                if numId is not None and ilvl is not None:
                    key = (str(numId), str(ilvl))
                    counters[key] += 1
                    abs_id = num_to_abs.get(str(numId))
                    fmt = None
                    lvlText = None
                    if abs_id is not None:
                        d = abs_lvls.get(abs_id, {}).get(str(ilvl), {})
                        fmt = d.get("numFmt")
                        lvlText = d.get("lvlText")
                    counter_str = format_counter(counters[key], fmt)
                    prefix = apply_lvl_text(counter_str, lvlText)
                    prefix = clean_ws(prefix)

            if s:
                line = (prefix + " " + s).strip() if prefix else s
                out.append(line)
        return out

    paras: List[str] = []
    with zipfile.ZipFile(path, "r") as z:
        parse_numbering(z)
        counters = defaultdict(int)

        if "word/document.xml" in z.namelist():
            paras.extend(paras_from_xml(z.read("word/document.xml"), counters))
        for name in z.namelist():
            if name.startswith("word/header") or name.startswith("word/footer"):
                paras.extend(paras_from_xml(z.read(name), counters))

    # de-dupe adjacent duplicates
    dedup: List[str] = []
    for p in paras:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    return dedup


def extract_answer_options(lines: List[str]) -> List[str]:
    """
    Options like:
      Very satisfied<TAB>1
      Very satisfied  1
      Yes1
      1) Very satisfied
    """
    opts: List[str] = []
    for ln in lines:
        ln = clean_ws(ln)
        if not ln:
            continue

        # skip matrix rows
        if MATRIX_ROW_RE.match(ln):
            continue
        mf = MATRIX_ROW_FALLBACK_RE.match(ln)
        if mf and len(mf.group("digits")) >= 4:
            continue

        m = OPTION_LINE_RE.match(ln)
        if m:
            opts.append(f"{m.group('text').strip()}\t{m.group('code').strip()}")
            continue

        m2 = OPTION_PREFIX_RE.match(ln)
        if m2:
            opts.append(f"{m2.group('text').strip()}\t{m2.group('code').strip()}")
            continue

        m3 = ATTACHED_CODE_RE.match(ln)
        if m3:
            text = m3.group("text").strip()
            code = m3.group("code").strip()
            # Keep only short codes (common options) or explicit REF/DK lines
            if len(code) <= 2 or re.search(r"\b(ref|dk|term|terminate)\b", ln, re.IGNORECASE):
                opts.append(f"{text}\t{code}")
            continue

    # de-dupe
    seen=set()
    out=[]
    for o in opts:
        k=o.lower()
        if k not in seen:
            seen.add(k)
            out.append(o)
    return out


def extract_answer_options(lines: List[str]) -> List[str]:
    opts = []
    for ln in lines:
        ln = ln.strip()
        if re.match(r"^(\d+|[A-Za-z])[\)\.\-]\s+\S+", ln):
            opts.append(re.sub(r"^(\d+|[A-Za-z])[\)\.\-]\s+", "", ln).strip())
        elif re.match(r"^(Strongly|Somewhat|Neither|Very|Not sure|Don'?t know)\b", ln, re.IGNORECASE):
            opts.append(ln)
    # de-dupe
    seen = set()
    out = []
    for o in opts:
        k = o.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(o)
    return out


@dataclass
class ExtractedQuestion:
    question_text: str
    question_type: str
    question_code: Optional[str]
    section: Optional[str]
    answer_options: List[str]
    matrix_rows: List[str]
    matrix_cols: List[str]
    source_locator: str
    source_quote: str

def docx_extract(path: str) -> List[ExtractedQuestion]:
    """
    STRICT index-based questionnaire extraction.

    A new question starts ONLY when the first line begins with:
      - digits: 1. / 2) / 5:
      - uppercase letters: A. / B) / C:

    Lowercase a./b. are treated as matrix rows (NOT new questions).
    """
    paras = docx_all_paragraph_text(path)

    def is_new_question_line(line: str) -> bool:
        line = clean_ws(line)
        if not line:
            return False
        if MATRIX_ROW_INDEX_RE.match(line):
            return False
        if OPTION_RE.match(line) or ATTACHED_OPTION_RE.match(line):
            return False
        return bool(QUESTION_INDEX_RE.match(line))

    questions: List[ExtractedQuestion] = []
    buf: List[str] = []
    buf_start = 0
    current_section = None

    def flush(end_idx: int):
        nonlocal buf, buf_start, current_section
        if not buf:
            return
        block_lines = [clean_ws(x) for x in buf if clean_ws(x)]
        buf = []
        if not block_lines:
            return

        first = block_lines[0]

        if first.isupper() and len(block_lines) == 1 and 6 <= len(first) <= 80:
            current_section = first
            return

        m = QUESTION_INDEX_RE.match(first)
        if not m:
            return  # strict mode: discard blocks that don't start with index

        qcode = m.group("code").strip()
        qtext = m.group("text").strip()

        rest = block_lines[1:]
        opts = extract_answer_options(rest)

        matrix_rows: List[str] = []
        for ln in rest:
            ln = clean_ws(ln)
            mm = MATRIX_ROW_RE.match(ln)
            if mm:
                matrix_rows.append(mm.group("rest").strip())
                continue
            mf = MATRIX_ROW_FALLBACK_RE.match(ln)
            if mf and len(mf.group("digits")) >= 4:
                matrix_rows.append(mf.group("text").strip())

        qtype = "open"
        if matrix_rows:
            qtype = "matrix"
        elif opts:
            qtype = "single"
        if any("select all" in l.lower() for l in block_lines):
            qtype = "multi"

        questions.append(
            ExtractedQuestion(
                question_text=qtext,
                question_type=qtype,
                question_code=qcode,
                section=current_section,
                answer_options=opts,
                matrix_rows=matrix_rows,
                matrix_cols=[],
                source_locator=f"docx:para:{buf_start}-{end_idx}",
                source_quote="\n".join(block_lines)[:900],
            )
        )

    for i, p in enumerate(paras):
        p = clean_ws(p)
        if not p:
            continue

        if is_new_question_line(p):
            if buf:
                flush(i - 1)
            buf_start = i
            buf = [p]
        else:
            if buf:
                buf.append(p)
            else:
                continue

    if buf:
        flush(len(paras) - 1)

    return questions


def odt_extract(path: str) -> List[ExtractedQuestion]:
    doc = odt_load(path)
    paras = doc.getElementsByType(odt_text.P)
    lines = [clean_ws(teletype.extractText(p)) for p in paras]
    lines = [l for l in lines if l]

    questions: List[ExtractedQuestion] = []
    buf: List[str] = []
    buf_start = 0
    current_section = None

    def flush(end_idx: int):
        nonlocal buf, buf_start, current_section
        if not buf:
            return
        block = "\n".join(buf).strip()
        buf = []
        parts = block.split("\n")
        first = parts[0].strip()

        if first.isupper() and len(parts) == 1 and 6 <= len(first) <= 80:
            current_section = first
            return

        m = QCODE_RE.match(first)
        qcode = m.group(1) if m else None
        rest = parts[1:]
        opts = extract_answer_options(rest)

        qtype = "open"
        if opts:
            qtype = "single"
        if any("select all" in l.lower() for l in parts):
            qtype = "multi"
        if re.search(r"\b0\s*to\s*10\b|\bscale\b|numeric entry|enter a number", block, re.IGNORECASE):
            qtype = "numeric"

        questions.append(
            ExtractedQuestion(
                question_text=first,
                question_type=qtype,
                question_code=qcode,
                section=current_section,
                answer_options=opts,
                matrix_rows=[],
                matrix_cols=[],
                source_locator=f"odt:line:{buf_start}-{end_idx}",
                source_quote=block[:700],
            )
        )

    for i, ln in enumerate(lines):
        is_new_q = bool(QCODE_RE.match(ln)) or (ln.endswith("?") and len(ln) > 15)
        if is_new_q and buf:
            flush(i - 1)
            buf_start = i
        buf.append(ln)
    flush(len(lines) - 1)
    return questions

def extract_questions(path: str, filename: str) -> List[ExtractedQuestion]:
    f = filename.lower()
    if f.endswith(".docx"):
        return docx_extract(path)
    if f.endswith(".odt"):
        return odt_extract(path)
    raise ValueError("Unsupported file type. Use .docx or .odt for the prototype.")

# -------------------------
# Embeddings + Canonicals
# -------------------------
def question_to_embed_text(qtext: str, opts, rows, cols) -> str:
    parts = [qtext.strip()]
    if opts:
        parts.append("OPTIONS: " + " | ".join(opts[:60]))
    if rows:
        parts.append("ROWS: " + " | ".join(rows[:120]))
    if cols:
        parts.append("COLS: " + " | ".join(cols[:60]))
    return clean_ws("\n".join(parts))

def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if not has_openai():
        return None
    client = openai_client()
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def assign_to_canonical(db: Session, q: Question, vec: List[float]):
    # Strict isolation by audience
    canonicals = db.query(Canonical).filter(Canonical.tenant_id==q.tenant_id, Canonical.audience==q.audience).all()
    best_cid = None
    best_sim = -1.0

    for c in canonicals:
        rep_q = db.query(Question).filter(Question.canonical_id==c.id).order_by(Question.year.asc(), Question.id.asc()).first()
        if not rep_q:
            continue
        rep_e = db.query(Embedding).filter(Embedding.question_id==rep_q.id).first()
        if not rep_e:
            continue
        rep_vec = jloads(rep_e.vector_json, [])
        if not rep_vec:
            continue
        sim = cosine(vec, rep_vec)
        if sim > best_sim:
            best_sim = sim
            best_cid = c.id

    if best_cid is not None and best_sim >= CANONICAL_SIM_THRESHOLD:
        q.canonical_id = best_cid
        c = db.query(Canonical).filter(Canonical.id==best_cid).first()
        c.appearance_count += 1
        db.add_all([q, c])
        db.commit()
        return

    c = Canonical(
        tenant_id=q.tenant_id,
        audience=q.audience,
        label=q.question_text[:220],
        appearance_count=1
    )
    db.add(c); db.commit(); db.refresh(c)
    q.canonical_id = c.id
    db.add(q); db.commit()

def ingest_document(db: Session, doc: Document) -> Dict[str, Any]:
    extracted = extract_questions(doc.storage_path, doc.filename)
    q_rows: List[Question] = []

    for ex in extracted:
        q = Question(
            doc_id=doc.id,
            tenant_id=doc.tenant_id,
            canonical_id=None,
            question_code=ex.question_code,
            section=ex.section,
            question_text=ex.question_text,
            question_type=ex.question_type,
            answer_options_json=jdumps(ex.answer_options),
            matrix_rows_json=jdumps(ex.matrix_rows),
            matrix_cols_json=jdumps(ex.matrix_cols),
            topics_json=jdumps([]),
            source_locator=ex.source_locator,
            source_quote=ex.source_quote,
            audience=doc.audience,
            year=doc.year,
            title=doc.title,
        )
        db.add(q); q_rows.append(q)

    db.commit()
    for q in q_rows: db.refresh(q)

    embeddings_enabled = False
    if has_openai() and q_rows:
        texts = []
        for q in q_rows:
            opts = jloads(q.answer_options_json, [])
            rows = jloads(q.matrix_rows_json, [])
            cols = jloads(q.matrix_cols_json, [])
            texts.append(question_to_embed_text(q.question_text, opts, rows, cols))

        vectors = embed_texts(texts)
        if vectors:
            embeddings_enabled = True
            for q, vec in zip(q_rows, vectors):
                e = Embedding(question_id=q.id, tenant_id=q.tenant_id, vector_json=jdumps(vec), model=OPENAI_EMBED_MODEL)
                db.add(e)
            db.commit()

            for q, vec in zip(q_rows, vectors):
                assign_to_canonical(db, q, vec)
    else:
        # No embeddings: canonical per question (still demo-OK)
        for q in q_rows:
            c = Canonical(tenant_id=q.tenant_id, audience=q.audience, label=q.question_text[:220], appearance_count=1)
            db.add(c); db.commit(); db.refresh(c)
            q.canonical_id = c.id
            db.add(q)
        db.commit()

    return {"questions_extracted": len(q_rows), "embeddings_enabled": embeddings_enabled, "canonical_threshold": CANONICAL_SIM_THRESHOLD}

# -------------------------
# Search (keyword + optional semantic)
# -------------------------
def serialize_question(q: Question) -> Dict[str, Any]:
    return {
        "question_id": q.id,
        "canonical_id": q.canonical_id,
        "survey_title": q.title,
        "year": q.year,
        "audience": q.audience,
        "section": q.section,
        "question_code": q.question_code,
        "question_type": q.question_type,
        "question_text": q.question_text,
        "answer_options": jloads(q.answer_options_json, []),
        "matrix_rows": jloads(q.matrix_rows_json, []),
        "matrix_cols": jloads(q.matrix_cols_json, []),
        "source_locator": q.source_locator,
        "source_quote": q.source_quote,
    }

def keyword_search(db: Session, tenant_id: str, query: str, limit: int = 80) -> List[Question]:
    like = f"%{query}%"
    return (db.query(Question)
        .filter(Question.tenant_id==tenant_id)
        .filter(
            (Question.question_text.ilike(like)) |
            (Question.title.ilike(like)) |
            (Question.answer_options_json.ilike(like)) |
            (Question.matrix_rows_json.ilike(like)) |
            (Question.matrix_cols_json.ilike(like))
        )
        .limit(limit)
        .all()
    )

def semantic_search(db: Session, tenant_id: str, query: str, limit: int = 80) -> List[Question]:
    if not has_openai():
        return []
    qvecs = embed_texts([query])
    if not qvecs:
        return []
    qvec = qvecs[0]
    embs = db.query(Embedding).filter(Embedding.tenant_id==tenant_id).all()
    scored = []
    for e in embs:
        vec = jloads(e.vector_json, [])
        if not vec: 
            continue
        scored.append((cosine(qvec, vec), e.question_id))
    scored.sort(reverse=True, key=lambda x: x[0])
    top_ids = [qid for _, qid in scored[:limit]]
    if not top_ids:
        return []
    qs = db.query(Question).filter(Question.id.in_(top_ids)).all()
    by_id = {q.id: q for q in qs}
    return [by_id[i] for i in top_ids if i in by_id]

def hybrid_search(db: Session, tenant_id: str, query: str, audience: Optional[str], year_from: Optional[int], year_to: Optional[int], limit: int=20):
    kw = keyword_search(db, tenant_id, query, limit=120)
    sem = semantic_search(db, tenant_id, query, limit=120)

    scores: Dict[int, float] = {}
    for i, q in enumerate(kw):
        scores[q.id] = scores.get(q.id, 0.0) + 1.0/(1+i)
    for i, q in enumerate(sem):
        scores[q.id] = scores.get(q.id, 0.0) + 2.0/(1+i)

    def ok(q: Question) -> bool:
        if audience and q.audience != audience:
            return False
        if year_from is not None and q.year < year_from:
            return False
        if year_to is not None and q.year > year_to:
            return False
        return True

    cands = [q for q in set(kw+sem) if ok(q)]
    cands.sort(key=lambda q: scores.get(q.id, 0.0), reverse=True)
    return [serialize_question(q) for q in cands[:limit]]

def canonical_timeline(db: Session, canonical_id: int, audience: str):
    qs = (db.query(Question)
          .filter(Question.canonical_id==canonical_id, Question.audience==audience)
          .order_by(Question.year.asc(), Question.id.asc())
          .all())
    return [serialize_question(q) for q in qs]

def compare_years(db: Session, tenant_id: str, query: str, year_a: int, year_b: int, audience: str, limit: int=25):
    hits = hybrid_search(db, tenant_id, query, audience=audience, year_from=None, year_to=None, limit=120)
    canon_ids = []
    seen = set()
    for h in hits:
        cid = h.get("canonical_id")
        if cid and cid not in seen:
            seen.add(cid); canon_ids.append(cid)

    out = []
    for cid in canon_ids:
        qa = db.query(Question).filter(Question.tenant_id==tenant_id, Question.audience==audience, Question.canonical_id==cid, Question.year==year_a).first()
        qb = db.query(Question).filter(Question.tenant_id==tenant_id, Question.audience==audience, Question.canonical_id==cid, Question.year==year_b).first()
        if qa or qb:
            out.append({
                "canonical_id": cid,
                "audience": audience,
                "year_a": year_a,
                "year_b": year_b,
                "question_a": serialize_question(qa) if qa else None,
                "question_b": serialize_question(qb) if qb else None,
            })
    out.sort(key=lambda r: (r["question_a"] is None, r["question_b"] is None))
    return out[:limit]

# -------------------------
# Generation (grounded)
# -------------------------
def generate_answer(user_question: str, evidence: List[Dict[str, Any]], audience: Optional[str], year_from: Optional[int], year_to: Optional[int]) -> Dict[str, Any]:
    if not has_openai():
        raise RuntimeError("OPENAI_API_KEY not set; generation disabled.")
    client = openai_client()

    packed = []
    for e in evidence[:18]:
        packed.append({
            "question_id": e["question_id"],
            "canonical_id": e.get("canonical_id"),
            "survey_title": e["survey_title"],
            "year": e["year"],
            "audience": e["audience"],
            "question_code": e.get("question_code"),
            "question_type": e.get("question_type"),
            "question_text": e["question_text"],
            "answer_options": e.get("answer_options", []),
            "matrix_rows": e.get("matrix_rows", []),
            "matrix_cols": e.get("matrix_cols", []),
            "source_locator": e.get("source_locator"),
        })

    system = (
        "You are a survey questionnaire librarian.\n"
        "Use ONLY the EVIDENCE provided. Do not invent surveys, years, wording, options, or locations.\n"
        "Return VALID JSON with keys: answer (string), citations (array of evidence items you used), notes (optional string)."
    )

    payload = {
        "question": user_question,
        "filters": {"audience": audience, "year_from": year_from, "year_to": year_to},
        "EVIDENCE": packed
    }

    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
        temperature=0.2
    )
    txt = resp.choices[0].message.content or ""
    try:
        data = json.loads(txt)
        if "answer" not in data:
            raise ValueError("missing answer")
        if "citations" not in data:
            data["citations"] = packed
        return data
    except Exception:
        return {"answer": txt.strip(), "citations": packed, "notes":"Model did not return valid JSON; returning raw output."}

# -------------------------
# FastAPI + UI
# -------------------------
app = FastAPI(title="Survey Q&A Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {"ok": True, "embeddings": has_openai(), "generation": has_openai()}

@app.post("/api/upload")
def api_upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    year: int = Form(...),
    audience: str = Form(...),
    tenant_id: str = Form("default"),
    db: Session = Depends(get_db),
):
    audience = audience.strip().lower()
    if audience not in ("member", "voter"):
        raise HTTPException(400, "audience must be member or voter")
    if not (file.filename.lower().endswith(".docx") or file.filename.lower().endswith(".odt")):
        raise HTTPException(400, "Prototype supports .docx and .odt only")

    dest_path = os.path.join(UPLOAD_DIR, f"{tenant_id}__{year}__{file.filename}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = Document(tenant_id=tenant_id, title=title, year=year, audience=audience, filename=file.filename, storage_path=dest_path)
    db.add(doc); db.commit(); db.refresh(doc)

    summary = ingest_document(db, doc)
    return {"document_id": doc.id, "ingest_summary": summary}

@app.get("/api/documents")
def api_documents(tenant_id: str="default", db: Session=Depends(get_db)):
    docs = db.query(Document).filter(Document.tenant_id==tenant_id).order_by(Document.year.desc()).all()
    return [{"id":d.id,"title":d.title,"year":d.year,"audience":d.audience,"filename":d.filename} for d in docs]

@app.get("/api/search")
def api_search(
    q: str,
    tenant_id: str="default",
    audience: Optional[str]=None,
    year_from: Optional[int]=None,
    year_to: Optional[int]=None,
    limit: int=20,
    db: Session=Depends(get_db),
):
    if audience:
        audience = audience.lower()
        if audience not in ("member","voter"):
            raise HTTPException(400,"audience must be member or voter")
    return hybrid_search(db, tenant_id, q, audience, year_from, year_to, limit)

@app.get("/api/question/{question_id}")
def api_question(question_id: int, db: Session=Depends(get_db)):
    q = db.query(Question).filter(Question.id==question_id).first()
    if not q:
        raise HTTPException(404, "question not found")
    data = serialize_question(q)
    if q.canonical_id:
        data["timeline"] = canonical_timeline(db, q.canonical_id, audience=q.audience)
    return data

@app.get("/api/compare")
def api_compare(
    q: str,
    year_a: int,
    year_b: int,
    tenant_id: str="default",
    audience: Optional[str]=None,
    limit: int=25,
    db: Session=Depends(get_db),
):
    # Always audience-isolated. If no audience passed, return both buckets.
    if audience:
        audience = audience.lower()
        if audience not in ("member","voter"):
            raise HTTPException(400,"audience must be member or voter")
        return compare_years(db, tenant_id, q, year_a, year_b, audience, limit)

    return {
        "member": compare_years(db, tenant_id, q, year_a, year_b, "member", limit),
        "voter": compare_years(db, tenant_id, q, year_a, year_b, "voter", limit),
    }

@app.get("/api/answer")
def api_answer(
    q: str,
    tenant_id: str="default",
    audience: Optional[str]=None,
    year_from: Optional[int]=None,
    year_to: Optional[int]=None,
    k: int=10,
    db: Session=Depends(get_db),
):
    if audience:
        audience = audience.lower()
        if audience not in ("member","voter"):
            raise HTTPException(400,"audience must be member or voter")

    evidence = hybrid_search(db, tenant_id, q, audience, year_from, year_to, limit=k)
    try:
        gen = generate_answer(q, evidence, audience, year_from, year_to)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    return {"query": q, "evidence_count": len(evidence), "answer": gen.get("answer"), "citations": gen.get("citations", []), "notes": gen.get("notes")}
