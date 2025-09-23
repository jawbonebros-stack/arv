# Foresight ARV (Replit-ready) — FastAPI + Jinja + WebSockets + SQLite
# Minimal MVP implementing: trials, commit–reveal, collaborative descriptors, judging, aggregation.
# NOTE: Experimental/entertainment only. Not financial advice.

import json
import hashlib
import secrets
import random
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, Form, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi import Depends, HTTPException, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.hash import bcrypt
from itsdangerous import TimestampSigner, BadSignature
import os
from dotenv import load_dotenv

# AI suggestions module
try:
    from ai_suggestions import (
        suggest_trial_configuration, 
        suggest_target_selection, 
        suggest_timing_optimization,
        analyze_trial_viability,
        AI_ENABLED
    )
except ImportError:
    AI_ENABLED = False

# Optional embeddings backends
# Temporarily disable OpenAI due to proxy compatibility issue
OpenAI = None

# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None
try:
    import torch
    import base64
    from PIL import Image
    import io
    # Try open_clip or clip
    try:
        import open_clip
        _HAS_OPEN_CLIP = True
        _HAS_CLIP = False
    except Exception:
        _HAS_OPEN_CLIP = False
        try:
            import clip as clip_lib
            _HAS_CLIP = True
        except Exception:
            _HAS_CLIP = False
except Exception:
    torch = None
    open_clip = None
    _HAS_OPEN_CLIP = False
    _HAS_CLIP = False
    Image = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
load_dotenv()

from sqlmodel import Field, SQLModel, create_engine, Session, select

# ---------------------------- DB Models ----------------------------

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    password_hash: str
    role: str = "viewer"  # admin | analyst | viewer | judge
    skill_score: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_admin(self):
        return self.role == "admin"

class Target(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uri: str
    modality: str = "image"
    tags: Optional[str] = None  # comma-separated
    image_embed_json: Optional[str] = None  # cached CLIP embedding as JSON list
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Trial(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    domain: str  # stocks | sports | lottery
    title: str   # human-friendly label
    event_spec_json: str  # JSON string
    market_type: str = "binary"
    result_time_utc: datetime
    status: str = "draft"  # draft | locked | open | settled | live
    decision_rule_json: str = "{}"
    live_start_utc: Optional[datetime] = None
    live_end_utc: Optional[datetime] = None
    draft_seconds: Optional[int] = 300
    created_by: Optional[int] = None  # User ID who created this trial
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TrialOutcome(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    label: str      # "A" / "B" or "Yes" / "No"
    implied_prob: Optional[float] = None

class TrialTarget(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    outcome_id: int
    target_id: int

class Commit(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    sha256_hash: str
    salt_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SessionRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    viewer_name: str
    descriptors_json: str = "[]"
    self_conf: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ConsensusDescriptor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    text: str
    category: Optional[str] = "general"  # colours, tactile, energy, smell, sound, visual, general
    upvotes: int = 0
    downvotes: int = 0
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Judgment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    judge_name: str
    score_a: float
    score_b: float
    score_c: Optional[float] = None  # For three-outcome sports trials
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Aggregate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    score_a: float
    score_b: float
    score_c: Optional[float] = None  # For three-outcome sports trials
    margin: float
    p: float
    ev: Optional[float] = None
    decision: str = "abstain"
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Result(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    outcome_label_won: str  # "A" or "B"
    settled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_uri: Optional[str] = None
    mapping_json: Optional[str] = None
    salt: Optional[str] = None

# ---------------------------- App Setup ----------------------------

app = FastAPI(title="Foresight ARV (Demo)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Add custom Jinja2 filters
import json
def from_json_filter(value):
    try:
        return json.loads(value) if value else {}
    except:
        return {}

templates.env.filters['from_json'] = from_json_filter

DB_URL = os.getenv("DATABASE_URL", "sqlite:///arv.db")
engine = create_engine(DB_URL)
SQLModel.metadata.create_all(engine)

# In-memory websockets per trial for descriptor collaboration
channels: Dict[int, List[WebSocket]] = {}

# ---------------------------- Auth Helpers ----------------------------

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")
signer = TimestampSigner(SESSION_SECRET)

def set_session(resp, user_id: int):
    token = signer.sign(str(user_id)).decode("utf-8")
    resp.set_cookie("session", token, httponly=True, samesite="lax")

def get_current_user(s: Session, session: Optional[str] = Cookie(default=None)):
    if not session:
        return None
    try:
        raw = signer.unsign(session, max_age=60*60*24*7).decode("utf-8")
        uid = int(raw)
        return s.get(User, uid)
    except BadSignature:
        return None

def db():
    with Session(engine) as s:
        yield s

def require_user(s: Session = Depends(db), session: Optional[str] = Cookie(default=None)):
    u = get_current_user(s, session)
    if not u:
        raise HTTPException(401, "Login required")
    return u

def require_admin():
    def _require_admin(s: Session = Depends(db), session: Optional[str] = Cookie(default=None)):
        u = get_current_user(s, session)
        if not u or u.role not in ["admin"]:
            raise HTTPException(403, "Forbidden: insufficient role")
        return u
    return _require_admin

def require_judge():
    def _require_judge(s: Session = Depends(db), session: Optional[str] = Cookie(default=None)):
        u = get_current_user(s, session)
        if not u or u.role not in ["judge", "admin"]:
            raise HTTPException(403, "Forbidden: insufficient role")
        return u
    return _require_judge

# ---------------------------- Helpers ----------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def commit_mapping(mapping: dict, salt: str) -> (str, str):
    data = json.dumps(mapping, sort_keys=True).encode("utf-8")
    h = sha256_bytes(salt.encode("utf-8") + data)
    return h, sha256_bytes(salt.encode("utf-8"))

def logistic(p: float) -> float:
    return 1/(1 + pow(2.718281828, -p))

def p_from_delta(delta: float, k: float = 0.8) -> float:
    return 1/(1 + pow(2.718281828, -k*delta))

# ---------------------------- Embeddings Backend ----------------------------

CLIP_ENABLE = os.getenv("CLIP_ENABLE", "0") == "1"
CLIP_WEIGHT = float(os.getenv("CLIP_WEIGHT", "0.5"))

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "auto")
OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def embed_texts(texts: list[str]) -> list[list[float]]:
    prov = EMBEDDINGS_PROVIDER
    if prov == "auto":
        prov = "openai" if (OpenAI and os.getenv("OPENAI_API_KEY")) else "tfidf"
    if prov == "openai" and OpenAI and os.getenv("OPENAI_API_KEY"):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    if TfidfVectorizer is None:
        return [[hashlib.md5(t.encode()).digest()[0] / 255.0] for t in texts]
    vec = TfidfVectorizer(stop_words="english").fit_transform(texts)
    return vec.toarray()

def cosine(a, b):
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb + 1e-9)

def embed_similarity(a_text: str, b_text: str) -> float:
    embs = embed_texts([a_text, b_text])
    return cosine(embs[0], embs[1])

# ---------------------------- CLIP Helpers (optional) ----------------------------
_CLIP_INIT = False
_CLIP_MODEL = None
_CLIP_PREPROC = None
_CLIP_DEVICE = "cpu"

def _init_clip():
    global _CLIP_INIT, _CLIP_MODEL, _CLIP_PREPROC, _CLIP_DEVICE
    if _CLIP_INIT or not CLIP_ENABLE or torch is None:
        _CLIP_INIT = True
        return
    _CLIP_DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"
    try:
        if _HAS_OPEN_CLIP:
            _CLIP_MODEL, _, _CLIP_PREPROC = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        elif _HAS_CLIP:
            _CLIP_MODEL, _CLIP_PREPROC = clip_lib.load('ViT-B/32', device=_CLIP_DEVICE, jit=False)
        else:
            pass
        if _CLIP_MODEL is not None:
            _CLIP_MODEL = _CLIP_MODEL.to(_CLIP_DEVICE)
    except Exception:
        _CLIP_MODEL = None
    _CLIP_INIT = True

def clip_image_embed_from_url(url: str):
    if not CLIP_ENABLE or torch is None:
        return None
    _init_clip()
    if _CLIP_MODEL is None or _CLIP_PREPROC is None:
        return None
    import requests
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if _HAS_OPEN_CLIP:
            image = _CLIP_PREPROC(img).unsqueeze(0).to(_CLIP_DEVICE)
            with torch.no_grad():
                emb = _CLIP_MODEL.encode_image(image)
        else:
            image = _CLIP_PREPROC(img).unsqueeze(0).to(_CLIP_DEVICE)
            with torch.no_grad():
                emb = _CLIP_MODEL.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().tolist()
    except Exception:
        return None

def clip_text_embed(text: str):
    if not CLIP_ENABLE or torch is None:
        return None
    _init_clip()
    if _CLIP_MODEL is None:
        return None
    try:
        if _HAS_OPEN_CLIP:
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            tokens = tokenizer([text])
            with torch.no_grad():
                emb = _CLIP_MODEL.encode_text(tokens.to(_CLIP_DEVICE))
        else:
            tokens = clip_lib.tokenize([text]).to(_CLIP_DEVICE)
            with torch.no_grad():
                emb = _CLIP_MODEL.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().tolist()
    except Exception:
        return None

def cosine_list(a, b):
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb + 1e-9)

# ---------------------------- Seed Targets ----------------------------

SAMPLE_TARGETS = [
    ("https://images.unsplash.com/photo-1523978591478-c753949ff840", "water, flowing, river, blue, wet, nature, waterfall"),
    ("https://images.unsplash.com/photo-1482192505345-5655af888cc4", "city, buildings, urban, skyscraper, night, lights, glass"),
    ("https://images.unsplash.com/photo-1500530855697-b586d89ba3ee", "desert, sand, dune, hot, dry, yellow, barren"),
    ("https://images.unsplash.com/photo-1499084732479-de2c02d45fc4", "concert, crowd, stage, loud, music, lights, energy"),
    ("https://images.unsplash.com/photo-1526178612274-3e45ce1e1f87", "mountain, snow, cold, peak, rock, sky, crisp"),
    ("https://images.unsplash.com/photo-1519681393784-d120267933ba", "books, library, shelves, paper, quiet, wood, study"),
]

def ensure_sample_targets(session: Session):
    if session.exec(select(Target)).first() is None:
        for uri, tags in SAMPLE_TARGETS:
            session.add(Target(uri=uri, tags=tags))
        session.commit()

def ensure_admin_user(session: Session):
    admin = session.exec(select(User).where(User.role == "admin")).first()
    if admin is None:
        admin_user = User(
            name="admin",
            email="admin@example.com",
            password_hash=bcrypt.hash("admin123"),
            role="admin"
        )
        session.add(admin_user)
        session.commit()

# ---------------------------- Routes ----------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, s: Session = Depends(db)):
    user = get_current_user(s, request.cookies.get("session"))
    ensure_sample_targets(s)
    ensure_admin_user(s)
    
    trials = s.exec(select(Trial).order_by(Trial.created_at.desc()).limit(10)).all()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "trials": trials
    })

# ==================== USER MANAGEMENT ROUTES ====================

@app.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request, s: Session = Depends(db)):
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    users = s.exec(select(User).order_by(User.created_at.desc())).all()
    
    # Calculate user statistics
    user_stats = {
        "total": len(users),
        "admins": len([u for u in users if u.role == "admin"]),
        "active_recently": len(users),  # Simplified - would need last_active field
        "trial_creators": len([u for u in users if u.role in ["admin", "analyst"]])
    }
    
    return templates.TemplateResponse("admin_users.html", {
        "request": request, 
        "user": user,
        "users": users,
        "user_stats": user_stats
    })

@app.post("/admin/users/create")
def create_user(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    skill_score: float = Form(0.0),
    s: Session = Depends(db)
):
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    # Check if email already exists
    existing_user = s.exec(select(User).where(User.email == email)).first()
    if existing_user:
        raise HTTPException(400, "Email already exists")
    
    # Validate role
    valid_roles = ["admin", "judge", "analyst", "viewer"]
    if role not in valid_roles:
        raise HTTPException(400, "Invalid role")
    
    # Create new user
    password_hash = bcrypt.hash(password)
    new_user = User(
        name=name,
        email=email,
        password_hash=password_hash,
        role=role,
        skill_score=skill_score
    )
    
    s.add(new_user)
    s.commit()
    
    return RedirectResponse(url="/admin/users", status_code=303)

@app.get("/admin/users/{user_id}")
def get_user_details(user_id: int, request: Request, s: Session = Depends(db)):
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    target_user = s.get(User, user_id)
    if not target_user:
        raise HTTPException(404, "User not found")
    
    return {
        "id": target_user.id,
        "name": target_user.name,
        "email": target_user.email,
        "role": target_user.role,
        "skill_score": target_user.skill_score,
        "created_at": target_user.created_at.isoformat()
    }

@app.post("/admin/users/{user_id}/edit")
def edit_user(
    user_id: int,
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    role: str = Form(...),
    skill_score: float = Form(...),
    new_password: str = Form(""),
    s: Session = Depends(db)
):
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    target_user = s.get(User, user_id)
    if not target_user:
        raise HTTPException(404, "User not found")
    
    # Check if email is taken by another user
    existing_user = s.exec(select(User).where(User.email == email, User.id != user_id)).first()
    if existing_user:
        raise HTTPException(400, "Email already exists")
    
    # Validate role
    valid_roles = ["admin", "judge", "analyst", "viewer"]
    if role not in valid_roles:
        raise HTTPException(400, "Invalid role")
    
    # Update user fields
    target_user.name = name
    target_user.email = email
    target_user.role = role
    target_user.skill_score = skill_score
    
    # Update password if provided
    if new_password.strip():
        target_user.password_hash = bcrypt.hash(new_password)
    
    s.add(target_user)
    s.commit()
    
    return RedirectResponse(url="/admin/users", status_code=303)

@app.post("/admin/users/{user_id}/delete")
def delete_user(user_id: int, request: Request, s: Session = Depends(db)):
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    target_user = s.get(User, user_id)
    if not target_user:
        raise HTTPException(404, "User not found")
    
    # Prevent deleting self
    if target_user.id == user.id:
        raise HTTPException(400, "Cannot delete your own account")
    
    # Delete user
    s.delete(target_user)
    s.commit()
    
    return {"success": True}

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login_post(request: Request, email: str = Form(), password: str = Form(), s: Session = Depends(db)):
    user = s.exec(select(User).where(User.email == email)).first()
    if not user or not bcrypt.verify(password, user.password_hash):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid credentials"
        })
    
    resp = RedirectResponse("/", status_code=302)
    set_session(resp, user.id)
    return resp

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register_post(request: Request, name: str = Form(), email: str = Form(), 
                 password: str = Form(), s: Session = Depends(db)):
    existing = s.exec(select(User).where(User.email == email)).first()
    if existing:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Email already registered"
        })
    
    user = User(
        name=name,
        email=email,
        password_hash=bcrypt.hash(password),
        role="viewer"
    )
    s.add(user)
    s.commit()
    
    resp = RedirectResponse("/", status_code=302)
    set_session(resp, user.id)
    return resp

@app.get("/logout")
def logout():
    resp = RedirectResponse("/", status_code=302)
    resp.delete_cookie("session")
    return resp

@app.get("/trials", response_class=HTMLResponse)
def trials_page(request: Request, s: Session = Depends(db)):
    user = get_current_user(s, request.cookies.get("session"))
    trials = s.exec(select(Trial).order_by(Trial.created_at.desc())).all()
    return templates.TemplateResponse("trials.html", {
        "request": request,
        "user": user,
        "trials": trials
    })

@app.get("/trials/create")
def trials_create_redirect():
    """Redirect /trials/create to the trial wizard"""
    return RedirectResponse("/admin/wizard", status_code=302)

@app.get("/trials/{trial_id}", response_class=HTMLResponse)
def trial_detail(request: Request, trial_id: int, s: Session = Depends(db)):
    user = get_current_user(s, request.cookies.get("session"))
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id)).all()
    descriptors = s.exec(select(ConsensusDescriptor).where(ConsensusDescriptor.trial_id == trial_id)).all()
    judgments = s.exec(select(Judgment).where(Judgment.trial_id == trial_id)).all()
    aggregate = s.exec(select(Aggregate).where(Aggregate.trial_id == trial_id)).first()
    
    # Get targets for each outcome
    targets_by_outcome = {}
    for outcome in outcomes:
        trial_targets = s.exec(
            select(TrialTarget, Target)
            .where(TrialTarget.trial_id == trial_id)
            .where(TrialTarget.outcome_id == outcome.id)
            .join(Target, TrialTarget.target_id == Target.id)
        ).all()
        targets_by_outcome[outcome.id] = [tt[1] for tt in trial_targets]
    
    return templates.TemplateResponse("trial_detail.html", {
        "request": request,
        "user": user,
        "trial": trial,
        "outcomes": outcomes,
        "descriptors": descriptors,
        "judgments": judgments,
        "aggregate": aggregate,
        "targets_by_outcome": targets_by_outcome
    })

@app.get("/targets", response_class=HTMLResponse)
def targets_page(request: Request, s: Session = Depends(db)):
    user = require_user(s, request.cookies.get("session"))
    targets = s.exec(select(Target).order_by(Target.created_at.desc())).all()
    return templates.TemplateResponse("targets.html", {
        "request": request,
        "user": user,
        "targets": targets
    })

@app.post("/targets/upload", response_model=None)
def upload_target(request: Request,
                 uri: str = Form(),
                 tags: str = Form(default=""),
                 s: Session = Depends(db),
                 user: User = Depends(require_user)):
    target = Target(uri=uri, tags=tags, modality="image")
    
    # Try to generate CLIP embedding
    if CLIP_ENABLE:
        embed = clip_image_embed_from_url(uri)
        if embed:
            target.image_embed_json = json.dumps(embed)
    
    s.add(target)
    s.commit()
    return RedirectResponse("/targets", status_code=302)

@app.delete("/targets/{target_id}", response_model=None)
def delete_target(target_id: int, s: Session = Depends(db), user: User = Depends(require_admin())):
    target = s.get(Target, target_id)
    if not target:
        raise HTTPException(404, "Target not found")
    
    # Check if target is used in any trials
    trial_targets = s.exec(select(TrialTarget).where(TrialTarget.target_id == target_id)).first()
    if trial_targets:
        raise HTTPException(400, "Cannot delete target: it is being used in trials")
    
    s.delete(target)
    s.commit()
    return {"status": "deleted"}

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, s: Session = Depends(db)):
    user = require_admin()(s, request.cookies.get("session"))
    trials = s.exec(select(Trial).order_by(Trial.created_at.desc())).all()
    targets = s.exec(select(Target)).all()
    users = s.exec(select(User)).all()
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "user": user,
        "trials": trials,
        "targets": targets,
        "users": users
    })

@app.get("/admin/wizard", response_class=HTMLResponse)
def trial_wizard(request: Request, s: Session = Depends(db)):
    user = require_user(s, request.cookies.get("session"))  # Changed from require_admin to require_user
    targets = s.exec(select(Target)).all()
    
    # Convert targets to serializable format
    targets_dict = [{"id": t.id, "tags": t.tags, "uri": t.uri} for t in targets]
    
    return templates.TemplateResponse("trial_wizard.html", {
        "request": request,
        "user": user,
        "targets": targets_dict,
        "ai_enabled": AI_ENABLED
    })

# AI Suggestions API endpoints
@app.post("/api/suggest_trial_config")
async def api_suggest_trial_config(
    title: str = Form(),
    domain: str = Form(), 
    description: str = Form(default=""),
    s: Session = Depends(db),
    user: User = Depends(require_admin())
):
    """Get AI suggestions for trial configuration."""
    if not AI_ENABLED:
        return {"error": "AI suggestions require OpenAI API key", "success": False}
    
    suggestions = suggest_trial_configuration(title, domain, description)
    return suggestions

@app.post("/api/suggest_targets")
async def api_suggest_targets(
    domain: str = Form(),
    outcomes: str = Form(),  # JSON string of outcome names
    s: Session = Depends(db),
    user: User = Depends(require_admin())
):
    """Get AI suggestions for target selection."""
    if not AI_ENABLED:
        return {"error": "AI suggestions require OpenAI API key", "success": False}
    
    try:
        outcomes_list = json.loads(outcomes)
        targets = s.exec(select(Target)).all()
        targets_dict = [{"id": t.id, "tags": t.tags, "uri": t.uri} for t in targets]
        
        suggestions = suggest_target_selection(domain, outcomes_list, targets_dict)
        return suggestions
    except Exception as e:
        return {"error": f"Failed to process request: {str(e)}", "success": False}

@app.post("/api/analyze_timing") 
async def api_analyze_timing(
    title: str = Form(),
    domain: str = Form(),
    proposed_time: str = Form(),
    s: Session = Depends(db),
    user: User = Depends(require_admin())
):
    """Get AI analysis of trial timing."""
    if not AI_ENABLED:
        return {"error": "AI suggestions require OpenAI API key", "success": False}
    
    suggestions = suggest_timing_optimization(title, domain, proposed_time)
    return suggestions

@app.post("/api/analyze_viability")
async def api_analyze_viability(
    title: str = Form(),
    domain: str = Form(),
    description: str = Form(default=""),
    outcomes: str = Form(),  # JSON string
    s: Session = Depends(db),
    user: User = Depends(require_admin())
):
    """Get AI analysis of trial viability."""
    if not AI_ENABLED:
        return {"error": "AI suggestions require OpenAI API key", "success": False}
    
    try:
        outcomes_list = json.loads(outcomes)
        suggestions = analyze_trial_viability(title, domain, description, outcomes_list)
        return suggestions
    except Exception as e:
        return {"error": f"Failed to process request: {str(e)}", "success": False}

def auto_select_distinctive_targets(s: Session, num_targets: int, exclude_recent: bool = True) -> list[Target]:
    """
    Automatically select distinctive targets for trial assignment.
    Uses smart selection to ensure maximum visual distinctiveness.
    """
    import random
    from collections import Counter
    
    # Get all available targets
    all_targets = s.exec(select(Target)).all()
    if len(all_targets) < num_targets:
        # If not enough targets, return what we have
        return random.sample(all_targets, min(len(all_targets), num_targets))
    
    # If requested, exclude recently used targets (last 10 trials)
    excluded_targets = set()
    if exclude_recent:
        recent_trials = s.exec(
            select(Trial)
            .order_by(Trial.created_at.desc())
            .limit(10)
        ).all()
        
        for trial in recent_trials:
            trial_targets = s.exec(
                select(TrialTarget).where(TrialTarget.trial_id == trial.id)
            ).all()
            for tt in trial_targets:
                excluded_targets.add(tt.target_id)
    
    # Filter out excluded targets
    available_targets = [t for t in all_targets if t.id not in excluded_targets]
    if len(available_targets) < num_targets:
        # If not enough non-recent targets, use all targets
        available_targets = all_targets
    
    # Smart selection algorithm: prioritize diversity
    selected = []
    
    # Parse tags for each target
    target_tags = {}
    for target in available_targets:
        if target.tags:
            target_tags[target.id] = set(tag.strip().lower() for tag in target.tags.split(','))
        else:
            target_tags[target.id] = set()
    
    # Select first target randomly
    first_target = random.choice(available_targets)
    selected.append(first_target)
    available_targets.remove(first_target)
    
    # For remaining targets, select those with maximum distinction
    while len(selected) < num_targets and available_targets:
        best_target = None
        max_distinctiveness = -1
        
        for candidate in available_targets:
            # Calculate distinctiveness as minimal tag overlap with selected targets
            distinctiveness = 0
            candidate_tags = target_tags.get(candidate.id, set())
            
            for selected_target in selected:
                selected_tags = target_tags.get(selected_target.id, set())
                # Higher score for less overlap (more distinctive)
                if candidate_tags and selected_tags:
                    overlap = len(candidate_tags & selected_tags)
                    max_possible = max(len(candidate_tags), len(selected_tags))
                    distinctiveness += (max_possible - overlap) / max_possible if max_possible > 0 else 1
                else:
                    # No tags data - treat as neutral
                    distinctiveness += 0.5
            
            if distinctiveness > max_distinctiveness:
                max_distinctiveness = distinctiveness
                best_target = candidate
        
        if best_target:
            selected.append(best_target)
            available_targets.remove(best_target)
        else:
            # Fallback: select randomly
            selected.append(random.choice(available_targets))
            available_targets.remove(selected[-1])
    
    return selected


@app.post("/admin/create_trial", response_model=None)
def create_trial(request: Request,
                title: str = Form(),
                domain: str = Form(),
                event_spec: str = Form(),
                result_time: str = Form(),
                target_a_id: Optional[int] = Form(default=None),
                target_b_id: Optional[int] = Form(default=None),
                target_c_id: Optional[int] = Form(default=None),
                outcome_a_name: str = Form(default=""),
                outcome_b_name: str = Form(default=""),
                outcome_c_name: str = Form(default=""),
                lottery_balls: Optional[int] = Form(default=None),
                auto_select_targets: bool = Form(default=True),
                s: Session = Depends(db),
                user: User = Depends(require_user)):
    
    # Parse datetime
    result_time_utc = datetime.fromisoformat(result_time.replace('Z', '+00:00'))
    
    # Create trial
    trial = Trial(
        title=title,
        domain=domain,
        event_spec_json=event_spec,
        result_time_utc=result_time_utc,
        status="draft",
        created_by=user.id
    )
    s.add(trial)
    s.flush()
    
    # Determine number of targets needed and auto-select if enabled
    num_targets_needed = 3 if domain == "sports" else (lottery_balls if domain == "lottery" and lottery_balls else 2)
    
    if auto_select_targets:
        selected_targets = auto_select_distinctive_targets(s, num_targets_needed)
        # Override manual selections with auto-selected targets
        if len(selected_targets) >= 1:
            target_a_id = selected_targets[0].id
        if len(selected_targets) >= 2:
            target_b_id = selected_targets[1].id
        if len(selected_targets) >= 3:
            target_c_id = selected_targets[2].id
    
    # Create outcomes based on domain
    if domain == "sports":
        # Sports trials have custom-named outcomes
        outcome_a_label = outcome_a_name if outcome_a_name else "Win"
        outcome_b_label = outcome_b_name if outcome_b_name else "Lose"
        outcome_c_label = outcome_c_name if outcome_c_name else "Draw"
        
        outcome_a = TrialOutcome(trial_id=trial.id, label=outcome_a_label)
        outcome_b = TrialOutcome(trial_id=trial.id, label=outcome_b_label) 
        outcome_c = TrialOutcome(trial_id=trial.id, label=outcome_c_label)
        s.add(outcome_a)
        s.add(outcome_b)
        s.add(outcome_c)
        s.flush()
        
        # Link targets (now auto-selected)
        if target_a_id:
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_a.id, target_id=target_a_id))
        if target_b_id:
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_b.id, target_id=target_b_id))
        if target_c_id:
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_c.id, target_id=target_c_id))
    elif domain == "lottery":
        # Lottery trials have variable number of balls/outcomes
        if not lottery_balls or lottery_balls < 2:
            lottery_balls = 5  # Default to 5 balls
            
        # Get available targets for auto-assignment
        available_targets = s.exec(select(Target)).all()
        if len(available_targets) < lottery_balls:
            raise HTTPException(400, f"Need at least {lottery_balls} targets available, but only {len(available_targets)} found")
        
        # Create outcomes for each ball number
        outcomes = []
        for i in range(lottery_balls):
            ball_number = i + 1
            outcome = TrialOutcome(trial_id=trial.id, label=f"Ball {ball_number}")
            s.add(outcome)
            outcomes.append(outcome)
        s.flush()
        
        # Auto-assign targets to outcomes (randomly shuffle for variety)
        import random
        shuffled_targets = random.sample(available_targets, lottery_balls)
        for outcome, target in zip(outcomes, shuffled_targets):
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome.id, target_id=target.id))
    else:
        # Binary outcomes for other domains (stocks, etc.)
        if not target_a_id or not target_b_id:
            raise HTTPException(400, "Target A and Target B are required for non-lottery domains")
            
        outcome_a = TrialOutcome(trial_id=trial.id, label="A")
        outcome_b = TrialOutcome(trial_id=trial.id, label="B")
        s.add(outcome_a)
        s.add(outcome_b)
        s.flush()
        
        # Link targets
        s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_a.id, target_id=target_a_id))
        s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_b.id, target_id=target_b_id))
    
    s.commit()
    return RedirectResponse("/admin", status_code=302)

@app.get("/trials/{trial_id}/edit", response_class=HTMLResponse)
def edit_trial_form(trial_id: int, request: Request, s: Session = Depends(db)):
    user = require_user(s, request.cookies.get("session"))
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Only allow editing draft trials
    if trial.status != "draft":
        raise HTTPException(400, "Only draft trials can be edited")
    
    # Get trial data
    outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id)).all()
    targets = s.exec(select(Target)).all()
    
    # Get current target assignments
    trial_targets = s.exec(
        select(TrialTarget, Target)
        .where(TrialTarget.trial_id == trial_id)
        .join(Target, TrialTarget.target_id == Target.id)
    ).all()
    
    # Organize targets by outcome
    targets_by_outcome = {}
    for tt, target in trial_targets:
        outcome = s.get(TrialOutcome, tt.outcome_id)
        if outcome:
            targets_by_outcome[outcome.label] = target
    
    # Convert targets to serializable format
    targets_dict = [{"id": t.id, "tags": t.tags, "uri": t.uri} for t in targets]
    
    return templates.TemplateResponse("edit_trial.html", {
        "request": request,
        "user": user,
        "trial": trial,
        "outcomes": outcomes,
        "targets": targets_dict,
        "targets_by_outcome": targets_by_outcome,
        "ai_enabled": AI_ENABLED
    })

@app.post("/trials/{trial_id}/edit", response_model=None)
def update_trial(trial_id: int,
                request: Request,
                title: str = Form(),
                domain: str = Form(),
                event_spec: str = Form(),
                result_time: str = Form(),
                target_a_id: Optional[int] = Form(default=None),
                target_b_id: Optional[int] = Form(default=None),
                target_c_id: Optional[int] = Form(default=None),
                outcome_a_name: str = Form(default=""),
                outcome_b_name: str = Form(default=""),
                outcome_c_name: str = Form(default=""),
                lottery_balls: Optional[int] = Form(default=None),
                s: Session = Depends(db),
                user: User = Depends(require_user)):
    
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Only allow editing draft trials
    if trial.status != "draft":
        raise HTTPException(400, "Only draft trials can be edited")
    
    # Parse datetime
    result_time_utc = datetime.fromisoformat(result_time.replace('Z', '+00:00'))
    
    # Update trial basics
    trial.title = title
    trial.domain = domain
    trial.event_spec_json = event_spec
    trial.result_time_utc = result_time_utc
    
    # Delete existing outcomes and trial targets
    existing_outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id)).all()
    existing_trial_targets = s.exec(select(TrialTarget).where(TrialTarget.trial_id == trial_id)).all()
    
    for tt in existing_trial_targets:
        s.delete(tt)
    for outcome in existing_outcomes:
        s.delete(outcome)
    s.flush()
    
    # Recreate outcomes based on domain (same logic as create_trial)
    if domain == "sports":
        # Sports trials have custom-named outcomes
        outcome_a_label = outcome_a_name if outcome_a_name else "Win"
        outcome_b_label = outcome_b_name if outcome_b_name else "Lose"
        outcome_c_label = outcome_c_name if outcome_c_name else "Draw"
        
        outcome_a = TrialOutcome(trial_id=trial.id, label=outcome_a_label)
        outcome_b = TrialOutcome(trial_id=trial.id, label=outcome_b_label) 
        outcome_c = TrialOutcome(trial_id=trial.id, label=outcome_c_label)
        s.add(outcome_a)
        s.add(outcome_b)
        s.add(outcome_c)
        s.flush()
        
        # Link targets
        if target_a_id:
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_a.id, target_id=target_a_id))
        if target_b_id:
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_b.id, target_id=target_b_id))
        if target_c_id:
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_c.id, target_id=target_c_id))
            
    elif domain == "lottery":
        # Lottery trials have variable number of balls/outcomes
        if not lottery_balls or lottery_balls < 2:
            lottery_balls = 5  # Default to 5 balls
            
        # Get available targets for auto-assignment
        available_targets = s.exec(select(Target)).all()
        if len(available_targets) < lottery_balls:
            raise HTTPException(400, f"Need at least {lottery_balls} targets available, but only {len(available_targets)} found")
        
        # Create outcomes for each ball number
        outcomes = []
        for i in range(lottery_balls):
            ball_number = i + 1
            outcome = TrialOutcome(trial_id=trial.id, label=f"Ball {ball_number}")
            s.add(outcome)
            outcomes.append(outcome)
        s.flush()
        
        # Auto-assign targets to outcomes (randomly shuffle for variety)
        import random
        shuffled_targets = random.sample(available_targets, lottery_balls)
        for outcome, target in zip(outcomes, shuffled_targets):
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome.id, target_id=target.id))
            
    else:
        # Binary outcomes for other domains (stocks, etc.)
        if not target_a_id or not target_b_id:
            raise HTTPException(400, "Target A and Target B are required for non-lottery domains")
            
        outcome_a = TrialOutcome(trial_id=trial.id, label="A")
        outcome_b = TrialOutcome(trial_id=trial.id, label="B")
        s.add(outcome_a)
        s.add(outcome_b)
        s.flush()
        
        # Link targets
        s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_a.id, target_id=target_a_id))
        s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_b.id, target_id=target_b_id))
    
    s.commit()
    return RedirectResponse(f"/trials/{trial_id}", status_code=302)

@app.post("/trials/{trial_id}/start")
def start_trial(trial_id: int,
               s: Session = Depends(db),
               user: User = Depends(require_admin())):
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    trial.status = "open"
    trial.live_start_utc = datetime.now(timezone.utc)
    trial.live_end_utc = datetime.now(timezone.utc) + timedelta(seconds=trial.draft_seconds or 300)
    s.commit()
    
    return {"status": "started"}

@app.post("/trials/{trial_id}/judge", response_model=None)
async def submit_judgment(trial_id: int,
                   request: Request,
                   notes: str = Form(default=""),
                   s: Session = Depends(db),
                   user: User = Depends(require_judge())):
    
    # Get trial and its outcomes
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Get outcomes for this trial to understand the scoring structure
    outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id)).all()
    
    # Parse form data for dynamic scoring
    form_data = await request.form()
    
    # Extract scores based on domain type
    if trial.domain == "lottery":
        # For lottery, scores are named like "score_ball_1", "score_ball_2", etc.
        scores = {}
        for key, value in form_data.items():
            if key.startswith("score_"):
                try:
                    scores[key] = float(value)
                except ValueError:
                    continue
        
        # Map scores to the first few score fields (a, b, c, etc.) - for now just store first 3
        score_a = scores.get("score_ball_1", 0.0)
        score_b = scores.get("score_ball_2", 0.0) 
        score_c = scores.get("score_ball_3", None)
    else:
        # For sports and other domains, use standard score_a, score_b, score_c
        score_a = float(form_data.get("score_a", 0))
        score_b = float(form_data.get("score_b", 0))
        score_c = float(form_data.get("score_c", 0)) if form_data.get("score_c") else None
    
    # Check if judge already submitted
    existing = s.exec(
        select(Judgment)
        .where(Judgment.trial_id == trial_id)
        .where(Judgment.judge_name == user.name)
    ).first()
    
    if existing:
        existing.score_a = score_a
        existing.score_b = score_b
        if trial.domain in ["sports", "lottery"]:
            existing.score_c = score_c
        existing.notes = notes
    else:
        judgment = Judgment(
            trial_id=trial_id,
            judge_name=user.name,
            score_a=score_a,
            score_b=score_b,
            score_c=score_c if trial.domain in ["sports", "lottery"] else None,
            notes=notes
        )
        s.add(judgment)
    
    s.commit()
    return RedirectResponse(f"/trials/{trial_id}", status_code=302)

@app.post("/trials/{trial_id}/descriptors", response_model=None)
def add_descriptor(trial_id: int,
                  text: str = Form(),
                  category: str = Form(default="general"),
                  s: Session = Depends(db),
                  user: User = Depends(require_user)):
    
    descriptor = ConsensusDescriptor(
        trial_id=trial_id,
        text=text,
        category=category,
        author=user.name
    )
    s.add(descriptor)
    s.commit()
    
    # Broadcast to WebSocket channels
    if trial_id in channels:
        message = {
            "type": "descriptor_added",
            "descriptor": {
                "id": descriptor.id,
                "text": descriptor.text,
                "category": descriptor.category,
                "author": descriptor.author,
                "upvotes": descriptor.upvotes,
                "downvotes": descriptor.downvotes
            }
        }
        for ws in channels[trial_id]:
            try:
                import asyncio
                asyncio.create_task(ws.send_json(message))
            except:
                pass
    
    return RedirectResponse(f"/trials/{trial_id}", status_code=302)

@app.websocket("/ws/trial/{trial_id}")
async def websocket_endpoint(websocket: WebSocket, trial_id: int):
    await websocket.accept()
    
    if trial_id not in channels:
        channels[trial_id] = []
    channels[trial_id].append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            # Echo to all connected clients for this trial
            for ws in channels[trial_id]:
                if ws != websocket:
                    try:
                        await ws.send_json(data)
                    except:
                        channels[trial_id].remove(ws)
    except WebSocketDisconnect:
        if websocket in channels[trial_id]:
            channels[trial_id].remove(websocket)

@app.post("/admin/load_additional_targets")
async def load_additional_targets(s: Session = Depends(db), user: User = Depends(require_admin())):
    """Load 100 additional stock photo targets into the database."""
    try:
        # Import and run the additional targets script
        from add_stock_targets import ADDITIONAL_TARGETS
        
        added_count = 0
        skipped_count = 0
        
        for target_data in ADDITIONAL_TARGETS:
            # Check if target already exists
            existing = s.exec(select(Target).where(
                (Target.tags == target_data["tags"]) | 
                (Target.uri == target_data["uri"])
            )).first()
            
            if existing:
                skipped_count += 1
                continue
            
            # Create new target
            target = Target(
                uri=target_data["uri"],
                tags=target_data["tags"],
                modality="image"
            )
            s.add(target)
            added_count += 1
        
        s.commit()
        
        # Get total count
        total_targets = len(s.exec(select(Target)).all())
        
        return {
            "loaded": added_count,
            "skipped": skipped_count,
            "total": total_targets,
            "message": f"Successfully loaded {added_count} additional stock targets"
        }
        
    except Exception as e:
        return {"error": f"Failed to load additional targets: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
