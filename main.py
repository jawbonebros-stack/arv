# Foresight ARV (Replit-ready) — FastAPI + Jinja + WebSockets + SQLite
# Minimal MVP implementing: trials, commit–reveal, collaborative descriptors, judging, aggregation.
# NOTE: Experimental/entertainment only. Not financial advice.

import json
import hashlib
import secrets
import random
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from functools import lru_cache

from fastapi import FastAPI, Request, Form, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi import Depends, HTTPException, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from passlib.hash import bcrypt
from itsdangerous import TimestampSigner, BadSignature
import os
from dotenv import load_dotenv
import stripe

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

# AI analysis module
try:
    from ai_analysis import AIAnalysisEngine, save_analysis_result, get_analysis_result
    AI_ANALYSIS_ENABLED = True
    print("✓ AI analysis enabled with AI analysis module")
except Exception as e:
    print(f"AI analysis module failed to load: {e}")
    AI_ANALYSIS_ENABLED = False

# Optional embeddings backends
try:
    from openai import OpenAI
    # Initialize OpenAI client
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0) if OPENAI_API_KEY else None
except Exception as e:
    print(f"OpenAI client initialization failed: {e}")
    OpenAI = None
    openai_client = None
# CLIP features now enabled with dependencies installed
torch = None
open_clip = None
clip_lib = None
_HAS_OPEN_CLIP = False
_HAS_CLIP = False

try:
    import base64
    import io
    from PIL import Image
except ImportError:
    Image = None
    
# CLIP dependencies now available - enable functionality
try:
    import torch
    # Try open_clip or clip
    try:
        import open_clip
        _HAS_OPEN_CLIP = True
        _HAS_CLIP = False
    except ImportError:
        _HAS_OPEN_CLIP = False
        try:
            import clip as clip_lib
            _HAS_CLIP = True
        except ImportError:
            _HAS_CLIP = False
except ImportError:
    torch = None
    open_clip = None
    _HAS_OPEN_CLIP = False
    _HAS_CLIP = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None

try:
    import imagehash
    from PIL import Image as PILImage
    _HAS_IMAGEHASH = True
except ImportError:
    imagehash = None
    PILImage = None
    _HAS_IMAGEHASH = False
    cosine_similarity = None
load_dotenv()

from sqlmodel import Field, SQLModel, create_engine, Session, select
from sqlalchemy import func, text

# ---------------------------- DB Models ----------------------------

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    password_hash: str
    role: str = "viewer"  # admin | analyst | viewer | judge
    skill_score: float = 0.0
    total_points: int = 0
    total_predictions: int = 0
    correct_predictions: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Pricing and credit system
    tasking_credits: int = 3  # Start with 3 free taskings
    individual_taskings_used: int = 0  # Track usage of individual taskings
    total_spent: float = 0.0  # Track total money spent on credits
    
    # Referral system
    referral_code: Optional[str] = Field(default=None, unique=True)  # User's unique referral code
    referred_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id")  # Who referred this user
    total_referrals: int = 0  # How many users this person has referred
    referral_credits_earned: int = 0  # Credits earned from referrals
    accuracy_credits_earned: int = 0  # Credits earned from accurate predictions
    
    @property
    def is_admin(self):
        return self.role == "admin"
    
    @property
    def accuracy_percentage(self):
        if self.total_predictions == 0:
            return 0.0
        return (self.correct_predictions / self.total_predictions) * 100
    
    @property
    def can_create_individual_tasking(self):
        """Check if user can create an individual tasking (has credits or is admin)"""
        return self.is_admin or self.tasking_credits > 0
    
    def use_individual_tasking_credit(self):
        """Use one credit for individual tasking creation"""
        if self.tasking_credits > 0:
            self.tasking_credits -= 1
            self.individual_taskings_used += 1
            return True
        return False
    
    def add_tasking_credits(self, credits: int, cost: float):
        """Add purchased credits to user account"""
        self.tasking_credits += credits
        self.total_spent += cost
    
    def reward_group_join_credit(self):
        """Reward creator with 1 credit when someone joins their group tasking"""
        self.tasking_credits += 1
    
    def generate_referral_code(self):
        """Generate a unique referral code for this user"""
        import secrets
        import string
        if not self.referral_code:
            # Generate 8-character alphanumeric code
            self.referral_code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            return self.referral_code
        return self.referral_code
    
    def reward_referral_credit(self, credits: int = 5):
        """Reward user with credits for successful referral"""
        self.tasking_credits += credits
        self.referral_credits_earned += credits
        self.total_referrals += 1
    
    def reward_accuracy_credit(self, credits: int = 1):
        """Reward user with credits for accurate prediction"""
        self.tasking_credits += credits
        self.accuracy_credits_earned += credits

class CreditPurchase(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    package_type: str  # "starter_10", "value_25", "bulk_50"
    credits_purchased: int
    cost: float
    stripe_session_id: Optional[str] = None
    payment_status: str = "pending"  # pending, completed, failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

class UserFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    workflow: str = Field(index=True)  # trial_creation, arv_session, trial_detail, etc.
    context: Optional[str] = None  # Additional context about the workflow step
    overall_rating: Optional[int] = Field(default=None, ge=1, le=5)
    ease_rating: Optional[int] = Field(default=None, ge=1, le=5)
    feedback_text: Optional[str] = None
    suggestions: Optional[str] = None
    issues: Optional[str] = None  # JSON array of issue types
    contact_ok: bool = Field(default=False)
    page_url: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # For anonymous feedback
    session_id: Optional[str] = None

class Target(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uri: str
    modality: str = "image"
    tags: Optional[str] = None  # comma-separated
    image_embed_json: Optional[str] = None  # cached CLIP embedding as JSON list
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

def generate_unique_target_number():
    """Generate a unique 7-digit target number for taskings"""
    return random.randint(1000000, 9999999)

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
    
    # Unique 7-digit target number for each tasking
    target_number: str = Field(default_factory=lambda: str(generate_unique_target_number()))
    
    # Group tasking fields
    is_group_tasking: bool = False  # True for group taskings
    is_open_for_joining: bool = False  # True if others can join this group
    max_participants: Optional[int] = None  # Max number of participants (None = unlimited)
    participant_count: int = 1  # Start with 1 (the creator)

class TrialParticipant(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int = Field(foreign_key="trial.id")
    user_id: int = Field(foreign_key="user.id")
    joined_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_creator: bool = False  # True for the trial creator
    credited_creator: bool = False  # True if this join already credited the creator

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
    target_id: Optional[int] = None  # For target-specific descriptors (lottery balls)
    outcome_id: Optional[int] = None  # For outcome-specific descriptors (lottery balls)
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

class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    user_id: int
    outcome_id: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_correct: Optional[bool] = None  # Set when trial is settled
    points_awarded: int = 0

class HomepageContent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    section: str = Field(index=True)  # hero, benefits, earning, etc.
    key: str = Field(index=True)      # title, subtitle, etc.
    content: str                      # The actual content
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Result(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trial_id: int
    outcome_label_won: str  # "A" or "B"
    outcome_id_won: Optional[int] = None  # The winning outcome ID
    settled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_uri: Optional[str] = None
    mapping_json: Optional[str] = None
    salt: Optional[str] = None
    admin_settled: bool = False  # Track if admin manually settled this trial

class NotificationSettings(SQLModel, table=True):
    """Admin-configurable notification settings"""
    id: Optional[int] = Field(default=None, primary_key=True)
    setting_key: str = Field(index=True)  # email_notifications_enabled, activity_feed_enabled, etc.
    setting_value: str  # JSON string for complex values
    description: Optional[str] = None
    updated_by: Optional[int] = Field(default=None, foreign_key="user.id")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserNotificationPreferences(SQLModel, table=True):
    """User preferences for different types of notifications"""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    email_task_conclusions: bool = True  # Email when participated tasks conclude
    email_task_updates: bool = True     # Email for other task updates
    browser_notifications: bool = True   # Browser push notifications
    activity_feed_enabled: bool = True  # Show activity feed on dashboard
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NotificationLog(SQLModel, table=True):
    """Track sent notifications to prevent duplicates"""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    trial_id: int = Field(foreign_key="trial.id", index=True)
    notification_type: str = Field(index=True)  # email_conclusion, browser_update, etc.
    status: str = "sent"  # sent, failed, pending
    details: Optional[str] = None  # JSON with additional info
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserActivity(SQLModel, table=True):
    """Track user activity to determine what's 'new' for them"""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True, unique=True)
    last_dashboard_visit: Optional[datetime] = None
    last_trial_check: Optional[datetime] = None
    last_notification_check: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ---------------------------- App Setup ----------------------------

app = FastAPI(title="ARVLab")

# Add compression middleware to reduce response sizes
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add optimized CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static files with cache headers
class CachedStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        # Cache static assets for 1 year
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        response.headers["Expires"] = "Thu, 31 Dec 2037 23:55:55 GMT"
        return response

app.mount("/static", CachedStaticFiles(directory="static"), name="static")
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

# Improve database connection handling for PostgreSQL
if DB_URL.startswith("postgresql"):
    # Add connection pooling and SSL settings for PostgreSQL
    engine = create_engine(
        DB_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "sslmode": "require",
            "connect_timeout": 10,
            "application_name": "onsenses_app"
        }
    )
else:
    # SQLite fallback
    engine = create_engine(DB_URL)

SQLModel.metadata.create_all(engine)

# Add database indexes for performance optimization
with engine.begin() as conn:
    try:
        # Index for trial queries
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_trial_status ON trial(status)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_trial_created_at ON trial(created_at DESC)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_trial_result_time ON trial(result_time_utc DESC)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_trial_domain ON trial(domain)'))
        
        # Index for user queries (quoted table name for PostgreSQL)
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_user_email ON "user"(email)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_user_name ON "user"(name)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_user_referral_code ON "user"(referral_code)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_user_total_predictions ON "user"(total_predictions)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_user_total_points ON "user"(total_points DESC)'))
        
        # Index for prediction queries
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_prediction_trial_id ON prediction(trial_id)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_prediction_user_id ON prediction(user_id)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_prediction_is_correct ON prediction(is_correct)'))
    except Exception as e:
        print(f"Note: Some indexes may already exist or be unnecessary: {e}")

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
        # Add retry logic for database operations
        for attempt in range(3):
            try:
                return s.get(User, uid)
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"Database error after 3 attempts: {e}")
                    return None
                print(f"Database connection attempt {attempt + 1} failed, retrying...")
                time.sleep(0.5)
    except BadSignature:
        return None

def db():
    with Session(engine) as s:
        yield s

def require_user(s: Session = Depends(db), session: Optional[str] = Cookie(default=None)):
    u = get_current_user(s, session)
    if not u:
        raise HTTPException(status_code=401, detail="Login required")
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

def commit_mapping(mapping: dict, salt: str) -> tuple[str, str]:
    data = json.dumps(mapping, sort_keys=True).encode("utf-8")
    h = sha256_bytes(salt.encode("utf-8") + data)
    return h, sha256_bytes(salt.encode("utf-8"))

def logistic(p: float) -> float:
    return 1/(1 + pow(2.718281828, -p))

def p_from_delta(delta: float, k: float = 0.8) -> float:
    return 1/(1 + pow(2.718281828, -k*delta))

# ---------------------------- Embeddings Backend ----------------------------

CLIP_ENABLE = True  # Enabled with installed dependencies
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
    
    try:
        # Handle both local file paths and HTTP URLs
        if url.startswith('http'):
            # External URL - fetch via HTTP (restricted to trusted sources)
            import requests
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        else:
            # Local file path - resolve safely within static directory
            import os
            # Remove leading "/" if present and sanitize path
            local_path = url.lstrip('/')
            
            # Security: Prevent path traversal attacks
            if '..' in local_path or os.path.isabs(local_path):
                print(f"Invalid file path (security): {local_path}")
                return None
                
            # Constrain to static directory only
            base_dir = os.path.join(os.getcwd(), "static")
            if not local_path.startswith('static/'):
                local_path = os.path.join('static', local_path)
            
            absolute_path = os.path.join(os.getcwd(), local_path)
            
            # Ensure path is within static directory
            if not os.path.commonpath([absolute_path, base_dir]) == base_dir:
                print(f"Path outside allowed directory: {absolute_path}")
                return None
            
            if not os.path.exists(absolute_path):
                print(f"Local image file not found: {absolute_path}")
                return None
                
            img = Image.open(absolute_path).convert("RGB")
        
        # Process image with CLIP
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
    except Exception as e:
        print(f"Failed to process image {url}: {e}")
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

# ---------------------------- Image Similarity Detection ----------------------------

def get_image_hash(image_path: str):
    """Generate perceptual hash for an image using dhash algorithm"""
    if not _HAS_IMAGEHASH:
        return None
    
    try:
        # Handle both local paths and URLs
        if image_path.startswith('http'):
            import requests
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            image = PILImage.open(io.BytesIO(response.content))
        else:
            # Local file path
            full_path = image_path if image_path.startswith('/') else f"/{image_path}"
            image = PILImage.open(full_path)
        
        # Generate perceptual hash using dhash (difference hash)
        hash_value = imagehash.dhash(image, hash_size=16)  # 16x16 = 256 bits for better precision
        return str(hash_value)
    except Exception as e:
        print(f"Error generating hash for {image_path}: {e}")
        return None

def calculate_image_similarity(target1: Target, target2: Target) -> float:
    """Calculate similarity between two target images using available methods"""
    
    # Method 1: Use CLIP embeddings if available
    if target1.image_embed_json and target2.image_embed_json:
        try:
            import json
            embed1 = json.loads(target1.image_embed_json)
            embed2 = json.loads(target2.image_embed_json)
            similarity = cosine_list(embed1, embed2)
            return similarity
        except Exception as e:
            print(f"Error using CLIP embeddings: {e}")
    
    # Method 2: Generate CLIP embeddings on-the-fly if CLIP is available
    if CLIP_ENABLE and (_HAS_OPEN_CLIP or _HAS_CLIP):
        try:
            embed1 = clip_image_embed_from_url(target1.uri if target1.uri.startswith('http') else f"/{target1.uri}")
            embed2 = clip_image_embed_from_url(target2.uri if target2.uri.startswith('http') else f"/{target2.uri}")
            
            if embed1 and embed2:
                similarity = cosine_list(embed1, embed2)
                return similarity
        except Exception as e:
            print(f"Error generating CLIP embeddings: {e}")
    
    # Method 3: Fallback to perceptual hashing
    if _HAS_IMAGEHASH:
        try:
            hash1 = get_image_hash(target1.uri)
            hash2 = get_image_hash(target2.uri)
            
            if hash1 and hash2:
                # Convert hash strings back to imagehash objects for comparison
                hash1_obj = imagehash.hex_to_hash(hash1)
                hash2_obj = imagehash.hex_to_hash(hash2)
                
                # Calculate hamming distance (lower = more similar)
                hamming_distance = hash1_obj - hash2_obj
                
                # Convert to similarity score (0-1, where 1 = identical)
                max_distance = len(hash1) * 4  # Each hex character represents 4 bits
                similarity = 1.0 - (hamming_distance / max_distance)
                return similarity
        except Exception as e:
            print(f"Error using perceptual hashing: {e}")
    
    # Method 4: Basic tag similarity as last resort
    if target1.tags and target2.tags:
        tags1 = set(target1.tags.lower().split(','))
        tags2 = set(target2.tags.lower().split(','))
        
        # Calculate Jaccard similarity
        intersection = len(tags1.intersection(tags2))
        union = len(tags1.union(tags2))
        
        return intersection / union if union > 0 else 0.0
    
    return 0.0  # No similarity method available

def find_similar_targets(session: Session, similarity_threshold: float = 0.85):
    """Find groups of similar target images"""
    targets = session.exec(select(Target)).all()
    similar_groups = []
    processed_targets = set()
    
    for i, target1 in enumerate(targets):
        if target1.id in processed_targets:
            continue
            
        similar_to_target1 = [target1]
        processed_targets.add(target1.id)
        
        for j, target2 in enumerate(targets[i+1:], start=i+1):
            if target2.id in processed_targets:
                continue
                
            similarity = calculate_image_similarity(target1, target2)
            
            if similarity >= similarity_threshold:
                similar_to_target1.append(target2)
                processed_targets.add(target2.id)
        
        # Only add groups with more than 1 image (i.e., actual duplicates/similar)
        if len(similar_to_target1) > 1:
            # Sort by creation date (newest first)
            similar_to_target1.sort(key=lambda x: x.created_at, reverse=True)
            similar_groups.append({
                'targets': similar_to_target1,
                'similarity_scores': [calculate_image_similarity(target1, t) for t in similar_to_target1[1:]]
            })
    
    return similar_groups

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

def ensure_homepage_content(session: Session):
    """Initialize default homepage content if none exists"""
    content_exists = session.exec(select(HomepageContent)).first()
    if content_exists is None:
        default_content = [
            # Hero section
            ("hero", "main_title", "Turn Your Skills Into Real Income"),
            ("hero", "subtitle", "Master Associative Remote Viewing (ARV) and generate income through stock predictions, sports betting, and lottery systems. Join our community of successful predictors making money from intuitive skills."),
            ("hero", "testimonial_author", "ProViewer_2023"),
            ("hero", "testimonial_text", "Made $2,400 last month with 73% accuracy on stock predictions"),
            
            # Benefits section  
            ("benefits", "benefit1_title", "Proven Income Generation"),
            ("benefits", "benefit2_title", "Scientific Method Training"),
            ("benefits", "benefit3_title", "Supportive Community"),
            
            # Earning section
            ("earning", "title", "Real Earning Potential"),
            ("earning", "subtitle", "Our top predictors consistently earn income through accurate predictions across multiple domains."),
            
            # Leaderboard section
            ("leaderboard", "title", "Top Performers"),
            ("leaderboard", "subtitle", "See how our community is performing")
        ]
        
        for section, key, content in default_content:
            homepage_content = HomepageContent(
                section=section,
                key=key,
                content=content
            )
            session.add(homepage_content)
        
        session.commit()

# ---------------------------- Routes ----------------------------

@app.get("/health")
@app.head("/health")
def health_check():
    """Simple health check endpoint for deployment monitoring"""
    return {"status": "ok", "service": "onsenses-arv"}

@app.api_route("/api", methods=["GET", "HEAD"])
async def api_health():
    """API health endpoint to stop 404 spam in logs"""
    return {"ok": True}

# ==================== SEO AND AI OPTIMIZATION ROUTES ====================

@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    """Robots.txt file for search engine and AI crawler guidance"""
    robots_content = """User-agent: *
Allow: /
Allow: /trials
Allow: /dashboard
Allow: /referrals
Allow: /api/public/*

# Disallow admin and private areas
Disallow: /admin
Disallow: /api/admin/*
Disallow: /account/*
Disallow: /purchase-credits
Disallow: /trials/*/edit
Disallow: /auth/*

# Allow AI services and search engines
User-agent: GPTBot
Allow: /
Allow: /api/public/*
Disallow: /admin
Disallow: /account/*

User-agent: Google-Extended
Allow: /
Allow: /api/public/*

User-agent: CCBot
Allow: /
Allow: /api/public/*

User-agent: Claude-Web
Allow: /
Allow: /api/public/*

User-agent: anthropic-ai
Allow: /
Allow: /api/public/*

User-agent: ChatGPT-User
Allow: /
Allow: /api/public/*

# Crawl delay for polite crawling
Crawl-delay: 1

# Sitemap location
Sitemap: https://arvlab.xyz/sitemap.xml"""
    return robots_content

@app.get("/sitemap.xml", response_class=Response)
def sitemap_xml(s: Session = Depends(db)):
    """Generate XML sitemap for all public pages"""
    from datetime import datetime
    
    # Get public trials
    public_trials = s.exec(
        select(Trial)
        .where(Trial.status.in_(["open", "live", "settled"]))
        .order_by(Trial.created_at.desc())
        .limit(1000)
    ).all()
    
    # Build sitemap XML
    sitemap_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    sitemap_content += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    
    # Homepage
    sitemap_content += '  <url>\n'
    sitemap_content += '    <loc>https://arvlab.xyz/</loc>\n'
    sitemap_content += f'    <lastmod>{datetime.now().strftime("%Y-%m-%d")}</lastmod>\n'
    sitemap_content += '    <changefreq>daily</changefreq>\n'
    sitemap_content += '    <priority>1.0</priority>\n'
    sitemap_content += '  </url>\n'
    
    # Trials page
    sitemap_content += '  <url>\n'
    sitemap_content += '    <loc>https://arvlab.xyz/trials</loc>\n'
    sitemap_content += f'    <lastmod>{datetime.now().strftime("%Y-%m-%d")}</lastmod>\n'
    sitemap_content += '    <changefreq>hourly</changefreq>\n'
    sitemap_content += '    <priority>0.9</priority>\n'
    sitemap_content += '  </url>\n'
    
    # Individual trial pages
    for trial in public_trials:
        sitemap_content += '  <url>\n'
        sitemap_content += f'    <loc>https://arvlab.xyz/trials/{trial.id}</loc>\n'
        sitemap_content += f'    <lastmod>{trial.created_at.strftime("%Y-%m-%d")}</lastmod>\n'
        sitemap_content += '    <changefreq>daily</changefreq>\n'
        sitemap_content += '    <priority>0.8</priority>\n'
        sitemap_content += '  </url>\n'
    
    # Public API endpoints for AI
    sitemap_content += '  <url>\n'
    sitemap_content += '    <loc>https://arvlab.xyz/api/public/leaderboard</loc>\n'
    sitemap_content += f'    <lastmod>{datetime.now().strftime("%Y-%m-%d")}</lastmod>\n'
    sitemap_content += '    <changefreq>daily</changefreq>\n'
    sitemap_content += '    <priority>0.7</priority>\n'
    sitemap_content += '  </url>\n'
    
    sitemap_content += '  <url>\n'
    sitemap_content += '    <loc>https://arvlab.xyz/api/public/stats</loc>\n'
    sitemap_content += f'    <lastmod>{datetime.now().strftime("%Y-%m-%d")}</lastmod>\n'
    sitemap_content += '    <changefreq>daily</changefreq>\n'
    sitemap_content += '    <priority>0.7</priority>\n'
    sitemap_content += '  </url>\n'
    
    sitemap_content += '</urlset>'
    
    return Response(content=sitemap_content, media_type="application/xml")

# ==================== PUBLIC API ENDPOINTS FOR AI SERVICES ====================

@lru_cache(maxsize=32)
def _get_cached_homepage_data():
    """Cached homepage data to improve load times"""
    with Session(engine) as s:
        # Get recent trials and leaderboard in optimized queries
        trials = s.exec(
            select(Trial)
            .where(
                Trial.status.in_(["open", "live", "settled"]),
                Trial.is_group_tasking == True
            )
            .order_by(Trial.created_at.desc())
            .limit(10)
        ).all()
        
        # Get leaderboard - only users with activity
        leaderboard = s.exec(
            select(User)
            .where(User.total_predictions > 0)
            .order_by(User.total_points.desc(), User.correct_predictions.desc())
            .limit(10)
        ).all()
        
        # Get homepage content
        homepage_content_raw = s.exec(select(HomepageContent)).all()
        homepage_content = {}
        for item in homepage_content_raw:
            if item.section not in homepage_content:
                homepage_content[item.section] = {}
            homepage_content[item.section][item.key] = item.content
        
        return {
            'trials': trials,
            'leaderboard': leaderboard, 
            'homepage_content': homepage_content
        }

@lru_cache(maxsize=128)
def _get_cached_stats():
    """Cached version of stats calculation"""
    with Session(engine) as s:
        # Get overall platform stats
        total_users = s.exec(select(func.count(User.id))).first()
        total_trials = s.exec(select(func.count(Trial.id))).first()
        total_predictions = s.exec(select(func.count(Prediction.id))).first()
        settled_trials = s.exec(select(func.count(Trial.id)).where(Trial.status == "settled")).first()
        
        # Get accuracy stats
        correct_predictions = s.exec(
            select(func.count(Prediction.id)).where(Prediction.is_correct == True)
        ).first()
        
        platform_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        # Domain statistics
        domain_stats = {}
        for domain in ["stocks", "sports", "lottery", "binary"]:
            domain_trials = s.exec(
                select(func.count(Trial.id)).where(Trial.domain == domain)
            ).first()
            domain_predictions = s.exec(
                select(func.count(Prediction.id))
                .join(Trial, Prediction.trial_id == Trial.id)
                .where(Trial.domain == domain)
            ).first()
            domain_correct = s.exec(
                select(func.count(Prediction.id))
                .join(Trial, Prediction.trial_id == Trial.id)
                .where(Trial.domain == domain, Prediction.is_correct == True)
            ).first()
            
            domain_accuracy = (domain_correct / domain_predictions * 100) if domain_predictions > 0 else 0
            
            domain_stats[domain] = {
                "trials": domain_trials or 0,
                "predictions": domain_predictions or 0,
                "accuracy": round(domain_accuracy, 1)
            }
        
        return {
            "platform": "ARVLab",
            "description": "Scientific platform for Associative Remote Viewing research",
            "url": "https://arvlab.xyz",
            "stats": {
                "total_users": total_users or 0,
                "total_trials": total_trials or 0,
                "total_predictions": total_predictions or 0,
                "settled_trials": settled_trials or 0,
                "platform_accuracy": round(platform_accuracy, 1),
                "domains": domain_stats
            },
            "research_focus": [
                "Associative Remote Viewing",
                "Consciousness Studies", 
                "Prediction Research",
                "Precognition Testing",
                "Collaborative Research"
            ]
        }

@app.get("/api/public/stats")
def get_public_stats(response: Response):
    """Public platform statistics for AI services and indexing"""
    
    # Set cache headers for API responses
    response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes cache
    response.headers["ETag"] = f'"stats-{int(datetime.now().timestamp() // 300)}"'
    
    stats_data = _get_cached_stats()
    stats_data["last_updated"] = datetime.now().isoformat()
    return stats_data

@app.get("/api/public/leaderboard")
def get_public_leaderboard(limit: int = 20, s: Session = Depends(db)):
    """Public leaderboard data for AI services"""
    
    # Get top performers
    leaderboard_users = s.exec(
        select(User)
        .where(User.total_predictions >= 5)  # Minimum activity requirement
        .order_by(User.total_points.desc(), User.correct_predictions.desc())
        .limit(min(limit, 50))  # Cap at 50 for performance
    ).all()
    
    leaderboard_data = []
    for rank, user in enumerate(leaderboard_users, 1):
        accuracy = (user.correct_predictions / user.total_predictions * 100) if user.total_predictions > 0 else 0
        
        leaderboard_data.append({
            "rank": rank,
            "username": user.name,
            "total_predictions": user.total_predictions,
            "correct_predictions": user.correct_predictions,
            "accuracy": round(accuracy, 1),
            "skill_score": user.total_points,
            "member_since": user.created_at.strftime("%Y-%m-%d") if user.created_at else None
        })
    
    return {
        "platform": "ARVLab",
        "leaderboard": leaderboard_data,
        "description": "Top performers in remote viewing prediction accuracy",
        "minimum_predictions": 5,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/public/trials/recent")
def get_recent_public_trials(limit: int = 10, s: Session = Depends(db)):
    """Recent public trials for AI indexing"""
    
    # Get recent settled trials with prediction counts in one query
    recent_trials_query = """
    SELECT t.id, t.title, t.domain, t.market_type, t.status, 
           t.created_at, t.result_time_utc,
           COUNT(p.id) as total_predictions,
           COUNT(CASE WHEN p.is_correct = true THEN 1 END) as correct_predictions
    FROM trial t
    LEFT JOIN prediction p ON t.id = p.trial_id
    WHERE t.status = 'settled'
    GROUP BY t.id, t.title, t.domain, t.market_type, t.status, t.created_at, t.result_time_utc
    ORDER BY t.result_time_utc DESC
    LIMIT :limit
    """
    
    from sqlalchemy import text
    result = s.exec(text(recent_trials_query), {"limit": min(limit, 20)})
    
    trials_data = []
    for row in result:
        trial_accuracy = (row.correct_predictions / row.total_predictions * 100) if row.total_predictions > 0 else 0
        
        trials_data.append({
            "id": row.id,
            "title": row.title,
            "domain": row.domain,
            "market_type": row.market_type,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "result_time": row.result_time_utc.isoformat() if row.result_time_utc else None,
            "participants": row.total_predictions,
            "accuracy": round(trial_accuracy, 1),
            "url": f"https://arvlab.xyz/trials/{row.id}"
        })
    
    return {
        "platform": "ARVLab",
        "recent_trials": trials_data,
        "description": "Recently completed prediction trials with results",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def index(request: Request, response: Response, s: Session = Depends(db)):
    # Set aggressive caching for homepage
    response.headers["Cache-Control"] = "public, max-age=180"  # 3 minutes cache
    response.headers["ETag"] = f'"homepage-{int(datetime.now().timestamp() // 180)}"'
    
    # Get user authentication (only operation that needs real-time data)
    user = get_current_user(s, request.cookies.get("session"))
    
    # Get cached homepage data
    cached_data = _get_cached_homepage_data()
    
    # Helper function for templates to get content with fallback
    def get_content(section: str, key: str, default: str = "") -> str:
        return cached_data['homepage_content'].get(section, {}).get(key, default)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "trials": cached_data['trials'],
        "leaderboard": cached_data['leaderboard'],
        "homepage_content": cached_data['homepage_content'],
        "get_content": get_content
    })

@app.get("/leaderboard", response_class=HTMLResponse)
@app.get("/top-performers", response_class=HTMLResponse)
def leaderboard_page(request: Request, s: Session = Depends(db)):
    """Dedicated Top Performers / Leaderboard page with detailed rankings"""
    
    user = get_current_user(s, request.cookies.get("session"))
    
    # Get comprehensive leaderboard data (more than homepage shows)
    leaderboard_users = s.exec(
        select(User)
        .where(User.total_predictions > 0)
        .order_by(User.total_points.desc(), User.correct_predictions.desc())
        .limit(50)  # Show top 50 instead of just 10
    ).all()
    
    # Get additional stats for the page
    total_users = s.exec(select(User).where(User.total_predictions > 0)).all()
    total_predictions = sum(user.total_predictions for user in total_users)
    avg_accuracy = sum(user.accuracy_percentage for user in total_users) / len(total_users) if total_users else 0
    
    # Calculate user's rank if logged in
    user_rank = None
    if user and user.total_predictions > 0:
        better_users = s.exec(
            select(User)
            .where(
                User.total_predictions > 0,
                (User.total_points > user.total_points) | 
                ((User.total_points == user.total_points) & (User.correct_predictions > user.correct_predictions))
            )
        ).all()
        user_rank = len(better_users) + 1
    
    leaderboard_stats = {
        "total_active_users": len(total_users),
        "total_predictions": total_predictions,
        "average_accuracy": round(avg_accuracy, 1),
        "user_rank": user_rank
    }
    
    return templates.TemplateResponse("leaderboard.html", {
        "request": request,
        "user": user,
        "leaderboard": leaderboard_users,
        "stats": leaderboard_stats
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
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    skill_score: float = Form(0.0),
    custom_name: str = Form(""),
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
    
    # Use custom name if provided, otherwise auto-generate Reddit-style username
    if custom_name.strip():
        # Check if custom name already exists
        existing_name = s.exec(select(User).where(User.name == custom_name.strip())).first()
        if existing_name:
            raise HTTPException(400, "Username already exists")
        name = custom_name.strip()
    else:
        name = generate_reddit_style_username(s)
    
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

# ---------------------------- Admin Duplicate Management Routes ----------------------------

@app.get("/admin/duplicates", response_class=HTMLResponse)
def admin_duplicates_page(request: Request, s: Session = Depends(db)):
    """Admin page for managing duplicate/similar target images"""
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    # Get similarity threshold from query params (default 0.85)
    similarity_threshold = float(request.query_params.get("threshold", "0.85"))
    
    # Find similar target groups
    similar_groups = find_similar_targets(s, similarity_threshold)
    
    # Calculate stats
    total_targets = s.exec(select(Target)).count()
    duplicate_targets = sum(len(group['targets']) for group in similar_groups)
    duplicate_groups = len(similar_groups)
    
    stats = {
        "total_targets": total_targets,
        "duplicate_targets": duplicate_targets, 
        "duplicate_groups": duplicate_groups,
        "similarity_threshold": similarity_threshold
    }
    
    return templates.TemplateResponse("admin_duplicates.html", {
        "request": request,
        "user": user,
        "similar_groups": similar_groups,
        "stats": stats
    })

@app.post("/admin/duplicates/scan")
def admin_scan_duplicates(
    request: Request,
    threshold: float = Form(0.85),
    s: Session = Depends(db)
):
    """API endpoint to scan for duplicates with custom threshold"""
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    try:
        # Find similar groups with the specified threshold
        similar_groups = find_similar_targets(s, threshold)
        
        # Calculate stats
        total_targets = s.exec(select(Target)).count()
        duplicate_targets = sum(len(group['targets']) for group in similar_groups)
        
        return {
            "success": True,
            "groups_found": len(similar_groups),
            "total_duplicates": duplicate_targets,
            "total_targets": total_targets,
            "redirect": f"/admin/duplicates?threshold={threshold}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/admin/targets/{target_id}/delete")
def admin_delete_target(
    target_id: int,
    request: Request,
    s: Session = Depends(db)
):
    """Delete a target image (admin only)"""
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    # Get the target
    target = s.get(Target, target_id)
    if not target:
        raise HTTPException(404, "Target not found")
    
    try:
        # Check if target is used in any active trials
        used_in_trials = s.exec(
            select(TrialTarget.trial_id).where(TrialTarget.target_id == target_id)
        ).all()
        
        if used_in_trials:
            trial_ids = [str(tid[0]) for tid in used_in_trials]
            return {
                "success": False, 
                "error": f"Target is used in trials: {', '.join(trial_ids)}. Cannot delete.",
                "used_in_trials": trial_ids
            }
        
        # Delete the target
        s.delete(target)
        s.commit()
        
        return {"success": True, "message": "Target deleted successfully"}
        
    except Exception as e:
        s.rollback()
        return {"success": False, "error": str(e)}

@app.post("/admin/targets/{target_id}/update")
def admin_update_target(
    target_id: int,
    request: Request,
    tags: str = Form(),
    s: Session = Depends(db)
):
    """Update target tags (admin only)"""
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    # Get the target
    target = s.get(Target, target_id)
    if not target:
        raise HTTPException(404, "Target not found")
    
    try:
        # Update tags
        target.tags = tags.strip()
        s.commit()
        
        return {"success": True, "message": "Target updated successfully"}
        
    except Exception as e:
        s.rollback()
        return {"success": False, "error": str(e)}

# ---------------------------- Bulk Target Management Routes ----------------------------

@app.post("/admin/targets/bulk-delete")
def admin_bulk_delete_targets(
    request: Request,
    data: dict,
    s: Session = Depends(db)
):
    """Bulk delete multiple targets (admin only)"""
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    target_ids = data.get("target_ids", [])
    if not target_ids:
        return {"success": False, "error": "No target IDs provided"}
    
    try:
        deleted_count = 0
        failed_targets = []
        
        for target_id in target_ids:
            target = s.get(Target, target_id)
            if not target:
                failed_targets.append({"id": target_id, "reason": "Target not found"})
                continue
                
            # Check if target is used in any trials
            used_in_trials = s.exec(
                select(TrialTarget.trial_id).where(TrialTarget.target_id == target_id)
            ).first()
            
            if used_in_trials:
                failed_targets.append({"id": target_id, "reason": f"Used in trial {used_in_trials[0]}"})
                continue
            
            # Delete the target
            s.delete(target)
            deleted_count += 1
        
        s.commit()
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "failed_targets": failed_targets
        }
        
    except Exception as e:
        s.rollback()
        return {"success": False, "error": str(e)}

@app.post("/admin/targets/bulk-edit-tags")
def admin_bulk_edit_tags(
    request: Request,
    data: dict,
    s: Session = Depends(db)
):
    """Bulk edit tags for multiple targets (admin only)"""
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    target_ids = data.get("target_ids", [])
    tags = data.get("tags", "").strip()
    action = data.get("action", "replace")  # replace, append, remove
    
    if not target_ids:
        return {"success": False, "error": "No target IDs provided"}
    
    try:
        updated_count = 0
        
        for target_id in target_ids:
            target = s.get(Target, target_id)
            if not target:
                continue
                
            if action == "replace":
                target.tags = tags
            elif action == "append":
                existing_tags = set(target.tags.split(",") if target.tags else [])
                new_tags = set(tag.strip() for tag in tags.split(",") if tag.strip())
                all_tags = existing_tags.union(new_tags)
                target.tags = ", ".join(sorted(all_tags))
            elif action == "remove":
                if target.tags:
                    existing_tags = set(target.tags.split(","))
                    remove_tags = set(tag.strip() for tag in tags.split(",") if tag.strip())
                    remaining_tags = existing_tags - remove_tags
                    target.tags = ", ".join(sorted(remaining_tags))
            
            updated_count += 1
        
        s.commit()
        
        return {
            "success": True,
            "updated_count": updated_count
        }
        
    except Exception as e:
        s.rollback()
        return {"success": False, "error": str(e)}

@app.post("/admin/targets/bulk-generate-embeddings")
def admin_bulk_generate_embeddings(
    request: Request,
    data: dict,
    s: Session = Depends(db)
):
    """Bulk generate embeddings for multiple targets (admin only)"""
    # Get current user and verify admin access
    user = get_current_user(s, request.cookies.get("session"))
    if not user or user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    target_ids = data.get("target_ids", [])
    if not target_ids:
        return {"success": False, "error": "No target IDs provided"}
    
    if not CLIP_ENABLE:
        return {"success": False, "error": "CLIP embeddings are not enabled"}
    
    try:
        processed_count = 0
        
        for target_id in target_ids:
            target = s.get(Target, target_id)
            if not target:
                continue
                
            # Skip if embedding already exists
            if target.image_embed_json:
                processed_count += 1
                continue
                
            # Generate embedding
            try:
                embed = clip_image_embed_from_url(target.uri if target.uri.startswith('http') else f"/{target.uri}")
                if embed:
                    import json
                    target.image_embed_json = json.dumps(embed)
                    processed_count += 1
            except Exception as e:
                print(f"Error generating embedding for target {target_id}: {e}")
                continue
        
        s.commit()
        
        return {
            "success": True,
            "processed_count": processed_count
        }
        
    except Exception as e:
        s.rollback()
        return {"success": False, "error": str(e)}

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

def generate_reddit_style_username(session: Session) -> str:
    """Generate a unique Reddit-style username like 'ProViewer_2023' or 'QuantumSeer_42'"""
    prefixes = [
        "Pro", "Quantum", "Mystic", "Psychic", "Remote", "Astral", "Cosmic", "Digital",
        "Future", "Mind", "Vision", "Crystal", "Neural", "Zen", "Alpha", "Ultra",
        "Meta", "Hyper", "Cyber", "Nexus", "Prime", "Elite", "Shadow", "Phoenix"
    ]
    
    suffixes = [
        "Viewer", "Seer", "Vision", "Mind", "Eye", "Sight", "Sense", "Oracle",
        "Prophet", "Reader", "Scanner", "Probe", "Detector", "Hunter", "Seeker",
        "Explorer", "Tracker", "Watcher", "Observer", "Gazer", "Finder", "Scout"
    ]
    
    # Try to generate a unique username
    max_attempts = 50
    for attempt in range(max_attempts):
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        number = random.randint(1, 9999)
        username = f"{prefix}{suffix}_{number}"
        
        # Check if username already exists
        existing = session.exec(select(User).where(User.name == username)).first()
        if not existing:
            return username
    
    # Fallback if all attempts fail
    return f"ARVUser_{random.randint(10000, 99999)}"

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, ref: Optional[str] = None):
    return templates.TemplateResponse("register.html", {
        "request": request,
        "referral_code": ref
    })

@app.post("/register")
def register_post(request: Request, 
                 email: str = Form(), 
                 password: str = Form(), 
                 referral_code: Optional[str] = Form(default=None),
                 s: Session = Depends(db)):
    existing = s.exec(select(User).where(User.email == email)).first()
    if existing:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Email already registered"
        })
    
    # Check referral code validity
    referrer = None
    if referral_code:
        referrer = s.exec(select(User).where(User.referral_code == referral_code.upper())).first()
        if not referrer:
            return templates.TemplateResponse("register.html", {
                "request": request,
                "error": "Invalid referral code"
            })
    
    # Auto-generate Reddit-style username
    reddit_username = generate_reddit_style_username(s)
    
    # Create new user with bonus credits if referred
    bonus_credits = 5 if referrer else 0
    user = User(
        name=reddit_username,
        email=email,
        password_hash=bcrypt.hash(password),
        role="viewer",
        tasking_credits=3 + bonus_credits,  # 3 base + 5 if referred
        referred_by_user_id=referrer.id if referrer else None
    )
    
    # Generate unique referral code for new user
    user.generate_referral_code()
    s.add(user)
    s.flush()  # Get user ID before rewarding referrer
    
    # Reward referrer with credits
    if referrer:
        referrer.reward_referral_credit(5)
        s.add(referrer)
    
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
    trials = s.exec(
        select(Trial)
        .where(Trial.is_group_tasking == True)
        .order_by(Trial.created_at.desc())
    ).all()
    return templates.TemplateResponse("trials.html", {
        "request": request,
        "user": user,
        "trials": trials
    })

@app.get("/trials/create")
def trials_create_redirect():
    """Redirect /trials/create to the trial wizard"""
    return RedirectResponse("/wizard", status_code=302)

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
    
    # Get trial result if settled
    result = s.exec(select(Result).where(Result.trial_id == trial_id)).first()
    winning_outcome_id = result.outcome_id_won if result else None
    
    # Get targets for each outcome
    targets_by_outcome = {}
    for outcome in outcomes:
        # For settled trials, only show targets for the winning outcome to prevent RV displacement
        if trial.status == "settled" and winning_outcome_id and outcome.id != winning_outcome_id:
            targets_by_outcome[outcome.id] = []
            continue
            
        trial_targets = s.exec(
            select(TrialTarget, Target)
            .where(TrialTarget.trial_id == trial_id)
            .where(TrialTarget.outcome_id == outcome.id)
            .join(Target, TrialTarget.target_id == Target.id)
        ).all()
        targets_by_outcome[outcome.id] = [tt[1] for tt in trial_targets]
    
    # Get target-specific descriptors for lottery trials
    target_descriptors = {}
    if trial.domain == 'lottery':
        try:
            target_descriptors_query = s.exec(
                select(ConsensusDescriptor)
                .where(ConsensusDescriptor.trial_id == trial_id)
                .where(ConsensusDescriptor.target_id != None)
            ).all()
            
            for desc in target_descriptors_query:
                if desc.outcome_id and desc.target_id:
                    key = f"{desc.outcome_id}_{desc.target_id}"
                    if key not in target_descriptors:
                        target_descriptors[key] = []
                    target_descriptors[key].append(desc)
        except Exception as e:
            print(f"Error loading target descriptors: {e}")
            target_descriptors = {}
    
    # Convert descriptors to dictionaries for JSON serialization
    descriptors_dict = []
    for desc in descriptors:
        descriptors_dict.append({
            "id": desc.id,
            "text": desc.text,
            "category": desc.category,
            "author": desc.author,
            "upvotes": desc.upvotes,
            "downvotes": desc.downvotes,
            "created_at": desc.created_at.isoformat() if desc.created_at else None
        })

    return templates.TemplateResponse("trial_detail.html", {
        "request": request,
        "user": user,
        "trial": trial,
        "outcomes": outcomes,
        "descriptors": descriptors,
        "descriptors_json": descriptors_dict,
        "judgments": judgments,
        "aggregate": aggregate,
        "targets_by_outcome": targets_by_outcome,
        "target_descriptors": target_descriptors,
        "result": result,
        "winning_outcome_id": winning_outcome_id
    })

@app.get("/targets", response_class=HTMLResponse)
def targets_page(request: Request, user: User = Depends(require_user), s: Session = Depends(db)):
    # Restrict access to admin users only
    if user.role != "admin":
        raise HTTPException(403, "Forbidden: Target library access restricted to administrators")
    targets = s.exec(select(Target).order_by(Target.created_at.desc())).all()
    return templates.TemplateResponse("targets.html", {
        "request": request,
        "user": user,
        "targets": targets
    })

@app.post("/targets/upload", response_model=None)
async def upload_target(request: Request,
                       uri: str = Form(default=""),
                       tags: str = Form(default=""),
                       file: Optional[UploadFile] = File(default=None),
                       s: Session = Depends(db),
                       user: User = Depends(require_admin())):
    
    # Check if this is a bulk upload request
    is_bulk_upload = request.headers.get("x-bulk-upload") == "true"
    
    try:
        # Handle file upload
        if file and file.filename:
            # Validate file type
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'}
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                error_msg = f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
                if is_bulk_upload:
                    return JSONResponse({"success": False, "error": error_msg}, status_code=400)
                else:
                    raise HTTPException(400, error_msg)
            
            # Create filename and save to static directory
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = f"static/{unique_filename}"
            
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Use the uploaded file path as URI (without 'static/' prefix since it's added in templates)
            target_uri = unique_filename
        elif uri:
            # Use provided URL
            target_uri = uri
        else:
            error_msg = "Either provide an image URL or upload a file"
            if is_bulk_upload:
                return JSONResponse({"success": False, "error": error_msg}, status_code=400)
            else:
                raise HTTPException(400, error_msg)
        
        target = Target(uri=target_uri, tags=tags, modality="image")
        
        # Try to generate CLIP embedding
        if CLIP_ENABLE:
            embed = clip_image_embed_from_url(target_uri if uri else f"/{target_uri}")
            if embed:
                target.image_embed_json = json.dumps(embed)
        
        s.add(target)
        s.commit()
        
        # Return JSON for bulk uploads, redirect for form submissions
        if is_bulk_upload:
            return {"success": True, "target_id": target.id, "uri": target_uri}
        else:
            return RedirectResponse("/targets", status_code=302)
            
    except Exception as e:
        # Handle database or file system errors
        error_msg = f"Upload failed: {str(e)}"
        if is_bulk_upload:
            return JSONResponse({"success": False, "error": error_msg}, status_code=500)
        else:
            raise HTTPException(500, error_msg)

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

def calculate_dashboard_data(s: Session, user: User) -> Dict[str, Any]:
    """Calculate comprehensive dashboard data for individual user"""
    from datetime import datetime, timedelta, timezone
    import json
    
    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)
    seven_days_ago = now - timedelta(days=7)
    
    try:
        # Get user's predictions and sessions with error handling
        user_predictions = s.exec(select(Prediction).where(Prediction.user_id == user.id)).all()
        user_sessions = s.exec(select(SessionRow).where(SessionRow.viewer_name == user.name)).all()
        
        # Get trials the user participated in
        user_trial_ids = set(p.trial_id for p in user_predictions)
        user_trials = s.exec(select(Trial).where(Trial.id.in_(list(user_trial_ids)))).all() if user_trial_ids else []
        
        # Get trials the user created (only group tasks)
        user_created_trials = s.exec(
            select(Trial)
            .where(Trial.created_by == user.id, Trial.is_group_tasking == True)
            .order_by(Trial.created_at.desc())
        ).all()
    except Exception as e:
        print(f"Error fetching user data for dashboard: {e}")
        # Return minimal dashboard data on error
        return {
            "performance": {
                "accuracy_rate": 0,
                "accuracy_trend": 0,
                "skill_score": 0,
                "domain_breakdown": {},
                "recent_timeline": []
            },
            "streaks": {"current_streak": 0, "longest_streak": 0},
            "engagement": {
                "total_sessions": 0, "sessions_this_week": 0, "sessions_this_month": 0,
                "avg_sessions_per_week": 0, "consistency_score": 0, "active_days_month": 0,
                "days_since_first_session": 0
            },
            "comparative": {
                "rank": 1, "total_users": 1, "percentile": 0, "accuracy_vs_community": 0,
                "sessions_last_30": 0, "predictions_last_30": 0
            },
            "achievements": {"earned_badges": [], "next_badges": []},
            "recent_sessions": [],
            "created_tasks": {"stats": {}, "recent": []},
            "notifications": {"unseen_count": 0, "activity_feed": []},
            "insights": []
        }
    
    # Performance Metrics
    total_predictions = len(user_predictions)
    correct_predictions = len([p for p in user_predictions if p.is_correct])
    accuracy_rate = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Recent performance trend (last 30 vs previous 30 days)
    sixty_days_ago = now - timedelta(days=60)
    recent_predictions = [p for p in user_predictions if p.created_at >= thirty_days_ago]
    previous_predictions = [p for p in user_predictions if sixty_days_ago <= p.created_at < thirty_days_ago]
    
    recent_correct = len([p for p in recent_predictions if p.is_correct])
    previous_correct = len([p for p in previous_predictions if p.is_correct])
    
    recent_accuracy = (recent_correct / len(recent_predictions) * 100) if recent_predictions else 0
    previous_accuracy = (previous_correct / len(previous_predictions) * 100) if previous_predictions else 0
    accuracy_trend = round(recent_accuracy - previous_accuracy, 1)
    
    # Domain breakdown
    domain_breakdown = {}
    for domain in ["stocks", "sports", "lottery", "other"]:
        domain_predictions = []
        for trial in user_trials:
            if trial.domain == domain:
                domain_predictions.extend([p for p in user_predictions if p.trial_id == trial.id])
        
        domain_correct = len([p for p in domain_predictions if p.is_correct])
        domain_total = len(domain_predictions)
        domain_accuracy = (domain_correct / domain_total * 100) if domain_total > 0 else 0
        
        domain_breakdown[domain] = {
            "accuracy": round(domain_accuracy, 1),
            "predictions": domain_total
        }
    
    # Streak calculation
    current_streak = 0
    longest_streak = 0
    temp_streak = 0
    
    sorted_predictions = sorted(user_predictions, key=lambda x: x.created_at, reverse=True)
    for pred in sorted_predictions:
        if pred.is_correct is None:
            continue
        if pred.is_correct:
            temp_streak += 1
            if current_streak == 0:  # First prediction from most recent
                current_streak = temp_streak
        else:
            longest_streak = max(longest_streak, temp_streak)
            temp_streak = 0
            if current_streak == 0:  # First prediction was incorrect
                current_streak = 0
                break
    longest_streak = max(longest_streak, temp_streak)
    
    # Session analytics
    total_sessions = len(user_sessions)
    sessions_this_week = len([s for s in user_sessions if s.created_at >= seven_days_ago])
    sessions_this_month = len([s for s in user_sessions if s.created_at >= thirty_days_ago])
    
    # Calculate consistency (days with sessions out of total active days)
    if user_sessions:
        first_session = min(user_sessions, key=lambda x: x.created_at).created_at
        days_since_first = (now - first_session).days + 1
        session_dates = set((s.created_at.date() for s in user_sessions))
        consistency_score = round((len(session_dates) / days_since_first * 100), 1) if days_since_first > 0 else 0
    else:
        days_since_first = 0
        consistency_score = 0
    
    # Get all users for ranking with error handling
    try:
        all_users = s.exec(select(User)).all()
        sorted_users = sorted(all_users, key=lambda u: u.accuracy_percentage, reverse=True)
        user_rank = next((i+1 for i, u in enumerate(sorted_users) if u.id == user.id), len(sorted_users))
        
        # Community comparison
        community_avg_accuracy = sum(u.accuracy_percentage for u in all_users) / len(all_users) if all_users else 0
        percentile = round((len(all_users) - user_rank + 1) / len(all_users) * 100, 1) if all_users else 0
    except Exception as e:
        print(f"Error calculating user rankings: {e}")
        all_users = [user]  # Fallback to just current user
        user_rank = 1
        community_avg_accuracy = 0
        percentile = 100
    
    # Achievements calculation
    earned_badges = []
    next_badges = []
    
    # Accuracy milestones
    if accuracy_rate >= 70:
        earned_badges.append({
            "name": "Expert Viewer", 
            "description": "70%+ accuracy rate",
            "icon": "target",
            "earned_date": "Recently"
        })
    elif accuracy_rate >= 60:
        earned_badges.append({
            "name": "Skilled Practitioner", 
            "description": "60%+ accuracy rate",
            "icon": "award",
            "earned_date": "Recently"
        })
        next_badges.append({
            "name": "Expert Viewer",
            "description": "Reach 70% accuracy",
            "icon": "target",
            "progress": round((accuracy_rate / 70) * 100, 1)
        })
    elif accuracy_rate >= 50:
        earned_badges.append({
            "name": "Rising Star", 
            "description": "50%+ accuracy rate",
            "icon": "star",
            "earned_date": "Recently"
        })
        next_badges.append({
            "name": "Skilled Practitioner",
            "description": "Reach 60% accuracy", 
            "icon": "award",
            "progress": round((accuracy_rate / 60) * 100, 1)
        })
    else:
        next_badges.append({
            "name": "Rising Star",
            "description": "Reach 50% accuracy",
            "icon": "star", 
            "progress": round((accuracy_rate / 50) * 100, 1)
        })
    
    # Session milestones
    if total_sessions >= 100:
        earned_badges.append({
            "name": "Century Club", 
            "description": "100+ sessions completed",
            "icon": "check-circle",
            "earned_date": "Recently"
        })
    elif total_sessions >= 50:
        earned_badges.append({
            "name": "Dedicated Viewer", 
            "description": "50+ sessions completed",
            "icon": "zap",
            "earned_date": "Recently"
        })
        next_badges.append({
            "name": "Century Club",
            "description": "Complete 100 sessions",
            "icon": "check-circle",
            "progress": round((total_sessions / 100) * 100, 1)
        })
    elif total_sessions >= 10:
        earned_badges.append({
            "name": "Active Participant", 
            "description": "10+ sessions completed",
            "icon": "trending-up",
            "earned_date": "Recently"
        })
        next_badges.append({
            "name": "Dedicated Viewer",
            "description": "Complete 50 sessions",
            "icon": "zap",
            "progress": round((total_sessions / 50) * 100, 1)
        })
    else:
        next_badges.append({
            "name": "Active Participant",
            "description": "Complete 10 sessions",
            "icon": "trending-up",
            "progress": round((total_sessions / 10) * 100, 1)
        })
    
    # Recent sessions for display
    recent_sessions = []
    for pred in sorted_predictions[:10]:  # Last 10 predictions
        trial = next((t for t in user_trials if t.id == pred.trial_id), None)
        if trial:
            recent_sessions.append({
                "trial_id": trial.id,
                "trial_title": trial.title,
                "date": pred.created_at.strftime("%b %d"),
                "is_correct": pred.is_correct,
                "domain": trial.domain,
                "status": trial.status
            })
    
    # User's created tasks analysis
    created_tasks_stats = {}
    created_tasks_recent = []
    
    if user_created_trials:
        # Status breakdown
        status_counts = {"draft": 0, "open": 0, "live": 0, "settled": 0}
        for trial in user_created_trials:
            status_counts[trial.status] = status_counts.get(trial.status, 0) + 1
        
        # Participation analysis for group tasks
        group_tasks = [t for t in user_created_trials if t.is_group_tasking]
        total_participants = 0
        for trial in group_tasks:
            # Get participant count for this trial
            participants = s.exec(select(TrialParticipant).where(TrialParticipant.trial_id == trial.id)).all()
            total_participants += len(participants)
        
        created_tasks_stats = {
            "total_created": len(user_created_trials),
            "status_breakdown": status_counts,
            "group_tasks": len(group_tasks),
            "individual_tasks": len(user_created_trials) - len(group_tasks),
            "total_participants": total_participants,
            "avg_participants_per_group": round(total_participants / len(group_tasks), 1) if group_tasks else 0
        }
        
        # Recent created tasks (last 10)
        for trial in user_created_trials[:10]:
            # Get participant count for this specific trial
            participants = s.exec(select(TrialParticipant).where(TrialParticipant.trial_id == trial.id)).all()
            participant_count = len(participants)
            
            # Get prediction count for this trial
            trial_predictions = s.exec(select(Prediction).where(Prediction.trial_id == trial.id)).all()
            prediction_count = len(trial_predictions)
            
            created_tasks_recent.append({
                "trial_id": trial.id,
                "title": trial.title,
                "domain": trial.domain,
                "status": trial.status,
                "created_date": trial.created_at.strftime("%b %d"),
                "is_group": trial.is_group_tasking,
                "participant_count": participant_count,
                "prediction_count": prediction_count,
                "result_time": trial.result_time_utc.strftime("%b %d") if trial.result_time_utc else None
            })
    
    # Generate insights
    insights = []
    if accuracy_rate > community_avg_accuracy:
        insights.append({
            "title": "Above Average Performance",
            "description": f"Your {accuracy_rate:.1f}% accuracy is {accuracy_rate - community_avg_accuracy:.1f}% above community average",
            "icon": "trending-up",
            "color": "76, 175, 80"  # Green
        })
    
    if longest_streak >= 5:
        insights.append({
            "title": "Strong Streak Performance", 
            "description": f"Your longest streak of {longest_streak} shows consistent capability",
            "icon": "zap",
            "color": "255, 193, 7"  # Yellow
        })
    
    if sessions_this_week < 2 and total_sessions > 5:
        insights.append({
            "title": "Practice Recommendation",
            "description": "Regular practice improves ARV performance. Try for 2-3 sessions per week",
            "icon": "calendar",
            "color": "33, 150, 243"  # Blue
        })
    
    # Recent timeline (last 4 weeks) with error handling
    recent_timeline = []
    try:
        for i in range(4):
            week_start = now - timedelta(days=(i+1)*7)
            week_end = now - timedelta(days=i*7)
            
            # Filter predictions with timezone awareness
            week_predictions = []
            for p in user_predictions:
                pred_time = p.created_at
                if pred_time.tzinfo is None:
                    pred_time = pred_time.replace(tzinfo=timezone.utc)
                if week_start <= pred_time < week_end:
                    week_predictions.append(p)
            
            week_correct = len([p for p in week_predictions if p.is_correct])
            week_accuracy = (week_correct / len(week_predictions) * 100) if week_predictions else 0
            
            recent_timeline.append({
                "period": f"Week {i+1}",
                "accuracy": round(week_accuracy, 1),
                "sessions": len(week_predictions)
            })
        
        recent_timeline.reverse()  # Show oldest to newest
    except Exception as e:
        print(f"Error calculating timeline: {e}")
        recent_timeline = [{"period": "Week 1", "accuracy": 0, "sessions": 0}]
    
    # Notification and activity data with better error handling
    try:
        from notification_service import get_notification_service
        notification_service = get_notification_service(s)
        
        # Get unseen concluded tasks for "New Results" badges
        unseen_tasks = notification_service.get_unseen_concluded_tasks(user.id)
        unseen_task_ids = {task["trial_id"] for task in unseen_tasks}
        
        # Update recent sessions with "new" status
        for session in recent_sessions:
            session["is_new_result"] = session["trial_id"] in unseen_task_ids
        
        # Get activity feed data
        activity_feed = notification_service.create_activity_feed_data(user.id, limit=8)
        
        # Count unseen items
        unseen_count = len(unseen_tasks)
        
    except (ImportError, AttributeError, RuntimeError, Exception) as e:
        # Fallback if notification service is not available or has errors
        print(f"Notification service error in dashboard: {e}")
        activity_feed = []
        unseen_count = 0
        for session in recent_sessions:
            session["is_new_result"] = False
    
    return {
        "performance": {
            "accuracy_rate": round(accuracy_rate, 1),
            "accuracy_trend": accuracy_trend,
            "skill_score": round(user.skill_score, 2),
            "domain_breakdown": domain_breakdown,
            "recent_timeline": recent_timeline
        },
        "streaks": {
            "current_streak": current_streak,
            "longest_streak": longest_streak
        },
        "engagement": {
            "total_sessions": total_sessions,
            "sessions_this_week": sessions_this_week,
            "sessions_this_month": sessions_this_month,
            "avg_sessions_per_week": round(total_sessions / max(days_since_first / 7, 1), 1) if days_since_first > 0 else 0,
            "consistency_score": consistency_score,
            "active_days_month": len(set(s.created_at.date() for s in user_sessions if s.created_at >= thirty_days_ago)),
            "days_since_first_session": days_since_first
        },
        "comparative": {
            "rank": user_rank,
            "total_users": len(all_users),
            "percentile": percentile,
            "accuracy_vs_community": round(accuracy_rate - community_avg_accuracy, 1),
            "sessions_last_30": sessions_this_month,
            "predictions_last_30": len(recent_predictions)
        },
        "achievements": {
            "earned_badges": earned_badges,
            "next_badges": next_badges[:3]  # Show top 3 next achievements
        },
        "recent_sessions": recent_sessions,
        "created_tasks": {
            "stats": created_tasks_stats,
            "recent": created_tasks_recent
        },
        "notifications": {
            "unseen_count": unseen_count,
            "activity_feed": activity_feed if isinstance(activity_feed, list) else []
        },
        "insights": insights[:3]  # Show top 3 insights
    }

def calculate_analytics(s: Session) -> Dict[str, Any]:
    """Calculate comprehensive analytics for the admin dashboard"""
    now = datetime.now(timezone.utc)
    
    # Basic counts
    users = s.exec(select(User)).all()
    trials = s.exec(select(Trial)).all()
    predictions = s.exec(select(Prediction)).all()
    sessions = s.exec(select(SessionRow)).all()
    descriptors = s.exec(select(ConsensusDescriptor)).all()
    
    # User Activity Analytics
    total_users = len(users)
    active_users_last_7_days = 0
    active_users_last_30_days = 0
    
    # Calculate active users based on prediction activity
    for user in users:
        user_predictions = [p for p in predictions if p.user_id == user.id]
        if user_predictions:
            last_prediction = max(user_predictions, key=lambda x: x.created_at).created_at
            days_since_last = (now - last_prediction).days
            if days_since_last <= 7:
                active_users_last_7_days += 1
            if days_since_last <= 30:
                active_users_last_30_days += 1
    
    # Task Completion Metrics
    total_trials = len(trials)
    completed_trials = len([t for t in trials if t.status == "settled"])
    active_trials = len([t for t in trials if t.status in ["open", "live"]])
    draft_trials = len([t for t in trials if t.status == "draft"])
    
    # Prediction Performance Analytics
    total_predictions = len(predictions)
    correct_predictions = len([p for p in predictions if p.is_correct])
    accuracy_rate = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Domain-specific performance
    domain_stats = {}
    for domain in ["stocks", "sports", "lottery", "other"]:
        domain_trials = [t for t in trials if t.domain == domain]
        domain_predictions = []
        for trial in domain_trials:
            domain_predictions.extend([p for p in predictions if p.trial_id == trial.id])
        
        domain_correct = len([p for p in domain_predictions if p.is_correct])
        domain_total = len(domain_predictions)
        domain_accuracy = (domain_correct / domain_total * 100) if domain_total > 0 else 0
        
        domain_stats[domain] = {
            "trials": len(domain_trials),
            "predictions": domain_total,
            "accuracy": round(domain_accuracy, 1)
        }
    
    # Session Analytics
    total_sessions = len(sessions)
    avg_descriptors_per_session = 0
    if sessions:
        total_descriptors_count = 0
        for session in sessions:
            try:
                descriptors_data = json.loads(session.descriptors_json)
                total_descriptors_count += len(descriptors_data)
            except:
                pass
        avg_descriptors_per_session = total_descriptors_count / len(sessions) if sessions else 0
    
    # Top Performers
    top_users = sorted(users, key=lambda u: u.accuracy_percentage, reverse=True)[:5]
    top_performers = []
    for user in top_users:
        if user.total_predictions > 0:
            top_performers.append({
                "name": user.name,
                "accuracy": round(user.accuracy_percentage, 1),
                "total_predictions": user.total_predictions,
                "skill_score": round(user.skill_score, 2)
            })
    
    # Recent Activity (last 30 days)
    thirty_days_ago = now - timedelta(days=30)
    
    # Handle timezone-aware comparison safely
    recent_trials = []
    for t in trials:
        trial_time = t.created_at
        if trial_time.tzinfo is None:
            trial_time = trial_time.replace(tzinfo=timezone.utc)
        if trial_time >= thirty_days_ago:
            recent_trials.append(t)
    
    recent_predictions = []
    for p in predictions:
        pred_time = p.created_at
        if pred_time.tzinfo is None:
            pred_time = pred_time.replace(tzinfo=timezone.utc)
        if pred_time >= thirty_days_ago:
            recent_predictions.append(p)
    
    recent_sessions = []
    for s in sessions:
        session_time = s.created_at
        if session_time.tzinfo is None:
            session_time = session_time.replace(tzinfo=timezone.utc)
        if session_time >= thirty_days_ago:
            recent_sessions.append(s)
    
    # Engagement metrics - handle timezone for new users
    new_users_30_days = []
    for u in users:
        user_time = u.created_at
        if user_time.tzinfo is None:
            user_time = user_time.replace(tzinfo=timezone.utc)
        if user_time >= thirty_days_ago:
            new_users_30_days.append(u)
    
    user_retention = {
        "new_users_30_days": len(new_users_30_days),
        "returning_users": active_users_last_30_days,
        "retention_rate": round((active_users_last_30_days / total_users * 100) if total_users > 0 else 0, 1)
    }
    
    # Collaborative metrics
    collaboration_stats = {
        "total_descriptors": len(descriptors),
        "avg_votes_per_descriptor": round(sum(d.upvotes + d.downvotes for d in descriptors) / len(descriptors), 1) if descriptors else 0,
        "collaborative_trials": len(set(d.trial_id for d in descriptors))
    }
    
    return {
        "user_activity": {
            "total_users": total_users,
            "active_7_days": active_users_last_7_days,
            "active_30_days": active_users_last_30_days,
            "retention": user_retention
        },
        "trial_metrics": {
            "total_trials": total_trials,
            "completed_trials": completed_trials,
            "active_trials": active_trials,
            "draft_trials": draft_trials,
            "completion_rate": round((completed_trials / total_trials * 100) if total_trials > 0 else 0, 1)
        },
        "prediction_performance": {
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy_rate": round(accuracy_rate, 1),
            "domain_stats": domain_stats
        },
        "session_analytics": {
            "total_sessions": total_sessions,
            "avg_descriptors_per_session": round(avg_descriptors_per_session, 1),
            "recent_sessions_30_days": len(recent_sessions)
        },
        "top_performers": top_performers,
        "collaboration": collaboration_stats,
        "recent_activity": {
            "trials_created_30_days": len(recent_trials),
            "predictions_made_30_days": len(recent_predictions),
            "sessions_completed_30_days": len(recent_sessions)
        }
    }

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, s: Session = Depends(db)):
    user = require_admin()(s, request.cookies.get("session"))
    trials = s.exec(select(Trial).order_by(Trial.created_at.desc())).all()
    targets = s.exec(select(Target)).all()
    users = s.exec(select(User)).all()
    
    # Calculate comprehensive analytics
    analytics = calculate_analytics(s)
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "user": user,
        "trials": trials,
        "targets": targets,
        "users": users,
        "analytics": analytics
    })

# Homepage editor routes
@app.get("/admin/homepage", response_class=HTMLResponse)
def homepage_editor(request: Request, s: Session = Depends(db)):
    user = require_admin()(s, request.cookies.get("session"))
    
    # Get all homepage content from database
    homepage_content_raw = s.exec(select(HomepageContent)).all()
    content = {}
    
    for item in homepage_content_raw:
        if item.section not in content:
            content[item.section] = {}
        content[item.section][item.key] = {"value": item.content, "id": item.id}
    
    return templates.TemplateResponse("admin/homepage_editor.html", {
        "request": request,
        "user": user,
        "content": content
    })

@app.post("/admin/homepage/update")
def update_homepage_content(
    request: Request,
    section: str = Form(...),
    key: str = Form(...), 
    content_value: str = Form(...),
    s: Session = Depends(db)
):
    user = require_admin()(s, request.cookies.get("session"))
    
    # Check if content exists
    existing = s.exec(
        select(HomepageContent)
        .where(HomepageContent.section == section)
        .where(HomepageContent.key == key)
    ).first()
    
    if existing:
        # Update existing content
        existing.content = content_value
        existing.updated_at = datetime.now(timezone.utc)
        s.add(existing)
    else:
        # Create new content
        new_content = HomepageContent(
            section=section,
            key=key,
            content=content_value
        )
        s.add(new_content)
    
    s.commit()
    
    return RedirectResponse(url="/admin/homepage?updated=success", status_code=303)

@app.get("/dashboard", response_class=HTMLResponse)
def user_dashboard(request: Request, s: Session = Depends(db)):
    # Check if user is logged in, redirect to login if not
    user = get_current_user(s, request.cookies.get("session"))
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    
    try:
        # Calculate comprehensive dashboard data for the user
        dashboard_data = calculate_dashboard_data(s, user)
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": user,
            "dashboard_data": dashboard_data
        })
    except Exception as e:
        print(f"Dashboard error for user {user.id}: {e}")
        # Return error template instead of crashing
        return templates.TemplateResponse("error.html", {
            "request": request,
            "user": user,
            "error_title": "Dashboard Error",
            "error_message": "There was an error loading your dashboard. Please try again.",
            "error_details": f"Technical details: {str(e)}"
        })

@app.get("/wizard", response_class=HTMLResponse)
def trial_wizard(request: Request, s: Session = Depends(db)):
    # Check if user is logged in, redirect to login if not
    user = get_current_user(s, request.cookies.get("session"))
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    
    targets = s.exec(select(Target)).all()
    
    # Convert targets to serializable format
    targets_dict = [{"id": t.id, "tags": t.tags, "uri": t.uri} for t in targets]
    
    return templates.TemplateResponse("trial_wizard.html", {
        "request": request,
        "user": user,
        "targets": targets_dict,
        "ai_enabled": AI_ENABLED
    })

@app.get("/admin/wizard", response_class=HTMLResponse)
def admin_trial_wizard(request: Request, user: User = Depends(require_user), s: Session = Depends(db)):
    targets = s.exec(select(Target)).all()
    
    # Convert targets to serializable format
    targets_dict = [{"id": t.id, "tags": t.tags, "uri": t.uri} for t in targets]
    
    return templates.TemplateResponse("trial_wizard.html", {
        "request": request,
        "user": user,
        "targets": targets_dict,
        "ai_enabled": AI_ENABLED
    })

@app.get("/admin/lottery/wizard", response_class=HTMLResponse)
def lottery_wizard(request: Request, user: User = Depends(require_user), s: Session = Depends(db)):
    if user.role != "admin":
        raise HTTPException(403, "Forbidden: Admin access required")
    
    today = datetime.now(timezone.utc).date().isoformat()
    
    return templates.TemplateResponse("lottery_wizard.html", {
        "request": request,
        "user": user,
        "today": today
    })

@app.post("/admin/lottery/create", response_model=None)
def create_lottery_trial(request: Request,
                        title: str = Form(),
                        ball_count: int = Form(),
                        ball_range: str = Form(),
                        result_date: str = Form(),
                        result_time: str = Form(),
                        draft_seconds: int = Form(default=300),
                        target_strategy: str = Form(default="auto"),
                        auto_start: bool = Form(default=False),
                        s: Session = Depends(db),
                        user: User = Depends(require_admin())):
    
    try:
        # Parse result datetime
        result_datetime_str = f"{result_date} {result_time}"
        result_time_utc = datetime.strptime(result_datetime_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        
        # Parse ball range
        range_parts = ball_range.split('-')
        min_ball = int(range_parts[0])
        max_ball = int(range_parts[1])
        
        # Generate random ball numbers
        import random
        ball_numbers = random.sample(range(min_ball, max_ball + 1), ball_count)
        ball_numbers.sort()
        
        # Create event spec
        event_spec = {
            "ball_count": ball_count,
            "ball_range": ball_range,
            "ball_numbers": ball_numbers,
            "type": "lottery"
        }
        
        # Create trial
        trial = Trial(
            domain="lottery",
            title=title,
            event_spec_json=json.dumps(event_spec),
            market_type="lottery",
            result_time_utc=result_time_utc,
            status="open" if auto_start else "draft",
            draft_seconds=draft_seconds,
            created_by=user.id
        )
        
        s.add(trial)
        s.commit()
        s.refresh(trial)
        
        # Create outcomes (one for each ball)
        for i, ball_num in enumerate(ball_numbers, 1):
            outcome = TrialOutcome(
                trial_id=trial.id,
                label=f"Ball {i}: {ball_num}"
            )
            s.add(outcome)
        
        s.commit()
        
        # Auto-assign targets if requested
        if target_strategy == "auto":
            outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial.id)).all()
            available_targets = s.exec(select(Target)).all()
            
            if len(available_targets) >= len(outcomes):
                # Use intelligent target selection for maximum distinctiveness
                selected_targets = []
                if AI_ENABLED and 'suggest_target_selection' in globals():
                    try:
                        target_selection = suggest_target_selection(
                            domain="lottery",
                            outcomes=len(outcomes),
                            available_targets=available_targets
                        )
                        selected_targets = target_selection.get('targets', [])
                    except Exception as e:
                        print(f"AI target selection failed: {e}")
                        pass
                
                # Fallback to random selection if AI fails
                if not selected_targets:
                    import random
                    selected_targets = random.sample(available_targets, len(outcomes))
                
                # Assign targets to outcomes
                for outcome, target in zip(outcomes, selected_targets):
                    trial_target = TrialTarget(
                        trial_id=trial.id,
                        outcome_id=outcome.id,
                        target_id=target.id
                    )
                    s.add(trial_target)
        
        s.commit()
        
        return RedirectResponse(f"/trials/{trial.id}", status_code=302)
        
    except Exception as e:
        print(f"Error creating lottery trial: {e}")
        raise HTTPException(500, f"Failed to create lottery trial: {str(e)}")

@app.get("/api/targets/count", response_model=None)
def get_targets_count(s: Session = Depends(db)):
    """Get count of available targets for wizard display"""
    count = len(s.exec(select(Target)).all())
    return {"count": count}

@app.get("/trials/{trial_id}/arv-wizard", response_class=HTMLResponse)
def arv_viewer_wizard(trial_id: int, request: Request, s: Session = Depends(db)):
    """ARV viewer wizard for adding sensory impressions"""
    # For testing - allow access without login
    user = None
    session_cookie = request.cookies.get("session")
    if session_cookie:
        try:
            user = require_user(s, session_cookie)
        except:
            pass
    
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    if trial.status not in ['open', 'live']:
        raise HTTPException(400, "Trial is not accepting new descriptors")
    
    # Get trial outcomes 
    outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id).order_by(TrialOutcome.id)).all()
    
    # For non-lottery tasks, only show single target (proper ARV methodology)
    # This prevents displacement to wrong targets
    if trial.domain != "lottery":
        # For non-lottery tasks, randomly select one outcome for ARV impressions
        # This maintains ARV integrity by focusing on single target
        import random
        if outcomes:
            selected_outcome = random.choice(outcomes)
            outcomes = [selected_outcome]
    
    return templates.TemplateResponse("arv_wizard_new.html", {
        "request": request,
        "user": user,
        "trial": trial,
        "outcomes": outcomes,
        "is_single_target": trial.domain != "lottery"
    })

@app.post("/api/trials/{trial_id}/arv-descriptors", response_model=None)
async def save_arv_descriptors(
    trial_id: int,
    request: Request,
    outcome_number: int = Form(),
    colours: str = Form(default=""),
    tactile: str = Form(default=""),
    energy: str = Form(default=""),
    smell: str = Form(default=""),
    sound: str = Form(default=""),
    visual: str = Form(default=""),
    s: Session = Depends(db)
):
    """Save ARV descriptors for a specific outcome in a trial"""
    
    # For testing - allow access without login, but get user if available
    user = None
    session_cookie = request.cookies.get("session")
    if session_cookie:
        try:
            user = require_user(s, session_cookie)
        except:
            pass
    
    # Fallback author name if no user
    author = user.name if user and hasattr(user, 'name') else "Anonymous"
    
    trial = s.get(Trial, trial_id)
    if not trial:
        return {"success": False, "message": "Trial not found"}
    
    if trial.status not in ['open', 'live']:
        return {"success": False, "message": "Trial is not accepting new descriptors"}
    
    # Get the specific outcome
    outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id).order_by(TrialOutcome.id)).all()
    if outcome_number < 1 or outcome_number > len(outcomes):
        return {"success": False, "message": "Invalid outcome number"}
    
    outcome = outcomes[outcome_number - 1]  # Convert to 0-based index
    
    # Save each non-empty descriptor
    descriptor_categories = [
        ('colours', colours),
        ('tactile', tactile), 
        ('energy', energy),
        ('smell', smell),
        ('sound', sound),
        ('visual', visual)
    ]
    
    saved_count = 0
    saved_descriptor_ids = []
    for category, text in descriptor_categories:
        if text.strip():  # Only save non-empty descriptors
            descriptor = ConsensusDescriptor(
                trial_id=trial_id,
                outcome_id=outcome.id,
                category=category,
                text=text.strip(),
                author=author,
                upvotes=0,
                downvotes=0
            )
            s.add(descriptor)
            s.flush()  # Get the ID before commit
            saved_descriptor_ids.append(descriptor.id)
            saved_count += 1
    
    try:
        s.commit()
        
        # Run AI analysis after successful save if analysis is enabled and user is available
        analysis_result = None
        if AI_ANALYSIS_ENABLED and user and hasattr(user, 'id'):
            try:
                from ai_analysis import AIAnalysisEngine
                engine = AIAnalysisEngine()
                print(f"Running AI analysis for user {user.id} on trial {trial_id}, outcome {outcome.id}")
                analysis_result = engine.analyze_viewing_session(trial_id, user.id, outcome.id, saved_descriptor_ids)
                if analysis_result:
                    print(f"AI analysis completed - recommended target: {analysis_result.recommended_target.target_name}")
                else:
                    print("AI analysis returned no results")
            except Exception as analysis_error:
                print(f"AI analysis failed: {analysis_error}")
                # Continue without analysis - don't break the save process
        
        response_data = {
            "success": True, 
            "message": f"Saved {saved_count} descriptors for outcome {outcome.label}",
            "descriptors_saved": saved_count,
            "outcome_label": outcome.label
        }
        
        # Add analysis results if available
        if analysis_result:
            # Convert analysis result to serializable format
            analysis_data = {
                "recommended_target": {
                    "target_id": analysis_result.recommended_target.target_id,
                    "target_name": analysis_result.recommended_target.target_name,
                    "overall_match": analysis_result.recommended_target.overall_match,
                    "category_matches": analysis_result.recommended_target.category_matches,
                    "reasoning": analysis_result.recommended_target.reasoning
                },
                "all_targets": [
                    {
                        "target_id": result.target_id,
                        "target_name": result.target_name,
                        "overall_match": result.overall_match,
                        "category_matches": result.category_matches,
                        "reasoning": result.reasoning
                    }
                    for result in analysis_result.results
                ],
                "analysis_summary": analysis_result.analysis_summary
            }
            response_data["ai_analysis"] = analysis_data
            response_data["message"] = f"Saved {saved_count} descriptors and analyzed your impressions!"
        
        return response_data
        
    except Exception as e:
        s.rollback()
        return {"success": False, "message": f"Error saving descriptors: {str(e)}"}

@app.post("/api/feedback", response_model=None)
async def submit_feedback(
    request: Request,
    workflow: str = Form(),
    context: Optional[str] = Form(default=None),
    overall_rating: Optional[int] = Form(default=None),
    ease_rating: Optional[int] = Form(default=None),
    feedback_text: Optional[str] = Form(default=None),
    suggestions: Optional[str] = Form(default=None),
    issues: Optional[str] = Form(default="[]"),
    contact_ok: bool = Form(default=False),
    page_url: Optional[str] = Form(default=None),
    user_agent: Optional[str] = Form(default=None),
    timestamp: Optional[str] = Form(default=None),
    s: Session = Depends(db)
):
    """Submit user feedback for workflow improvement"""
    
    # Get user if authenticated, allow anonymous feedback
    user_id = None
    session_cookie = request.cookies.get("session")
    if session_cookie:
        try:
            user = require_user(s, session_cookie)
            user_id = user.id
        except:
            pass
    
    # Generate session ID for anonymous users
    session_id = None
    if not user_id:
        session_id = request.cookies.get("session", f"anon_{int(time.time())}")
    
    # Validate ratings
    if overall_rating is not None and (overall_rating < 1 or overall_rating > 5):
        return {"success": False, "message": "Overall rating must be between 1 and 5"}
    if ease_rating is not None and (ease_rating < 1 or ease_rating > 5):
        return {"success": False, "message": "Ease rating must be between 1 and 5"}
    
    try:
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            workflow=workflow,
            context=context,
            overall_rating=overall_rating,
            ease_rating=ease_rating,
            feedback_text=feedback_text.strip() if feedback_text else None,
            suggestions=suggestions.strip() if suggestions else None,
            issues=issues,
            contact_ok=contact_ok,
            page_url=page_url,
            user_agent=user_agent
        )
        
        s.add(feedback)
        s.commit()
        
        return {"success": True, "message": "Thank you for your feedback!"}
        
    except Exception as e:
        s.rollback()
        return {"success": False, "message": f"Error saving feedback: {str(e)}"}

@app.get("/api/current_user")
async def api_current_user(
    request: Request,
    s: Session = Depends(db),
    session: Optional[str] = Cookie(default=None)
):
    """Get current user information"""
    user = get_current_user(s, session)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {
        "id": user.id,
        "name": user.name,
        "role": user.role
    }

@app.post("/api/analyze_viewing")
async def api_analyze_viewing(
    request: Request,
    trial_id: int = Form(...),
    user_id: int = Form(...)
):
    """API endpoint for AI analysis of viewing session"""
    try:
        if not AI_ANALYSIS_ENABLED:
            return {"success": False, "error": "AI analysis is not available"}
        
        # Check if analysis already exists
        existing_analysis = get_analysis_result(trial_id, user_id)
        if existing_analysis:
            return {
                "success": True, 
                "analysis": _format_analysis_for_api(existing_analysis),
                "cached": True
            }
        
        # Perform new analysis
        engine = AIAnalysisEngine()
        analysis = engine.analyze_viewing_session(trial_id, user_id)
        
        if not analysis:
            return {"success": False, "error": "Unable to perform analysis"}
        
        # Save analysis result
        save_analysis_result(trial_id, user_id, analysis)
        
        return {
            "success": True,
            "analysis": _format_analysis_for_api(analysis),
            "cached": False
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def _format_analysis_for_api(analysis):
    """Format analysis result for API response"""
    return {
        "recommended_target": {
            "target_id": analysis.recommended_target.target_id,
            "target_name": analysis.recommended_target.target_name,
            "overall_match": analysis.recommended_target.overall_match,
            "category_matches": analysis.recommended_target.category_matches,
            "reasoning": analysis.recommended_target.reasoning
        },
        "all_results": [
            {
                "target_id": r.target_id,
                "target_name": r.target_name,
                "overall_match": r.overall_match,
                "category_matches": r.category_matches,
                "reasoning": r.reasoning
            }
            for r in analysis.results
        ],
        "summary": analysis.analysis_summary
    }

@app.get("/trials/{trial_id}/analysis/{user_id}", response_class=HTMLResponse)
async def view_analysis_results(
    request: Request,
    trial_id: int,
    user_id: int,
    s: Session = Depends(db),
    user: User = Depends(require_user)
):
    """View AI analysis results for a viewing session"""
    
    # Get trial information
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Task not found")
    
    # Check permissions - user can view their own analysis or admin can view any
    if user.id != user_id and not user.is_admin:
        raise HTTPException(403, "Access denied")
    
    # Get analysis result
    analysis = get_analysis_result(trial_id, user_id)
    
    # Get winning target image if analysis exists
    winning_target = None
    if analysis and analysis.recommended_target:
        # Find the target with matching ID
        winning_target = s.exec(
            select(Target).where(Target.id == analysis.recommended_target.target_id)
        ).first()
    
    return templates.TemplateResponse("analysis_results.html", {
        "request": request,
        "user": user,
        "trial": trial,
        "analysis": analysis,
        "winning_target": winning_target
    })

@app.get("/admin/feedback", response_class=HTMLResponse)
async def admin_feedback_dashboard(request: Request, user: User = Depends(require_user), s: Session = Depends(db)):
    """Admin dashboard for viewing user feedback"""
    if not user.is_admin:
        raise HTTPException(403, "Admin access required")
    
    # Get feedback statistics
    total_feedback = len(s.exec(select(UserFeedback)).all())
    recent_feedback = s.exec(
        select(UserFeedback)
        .where(UserFeedback.created_at >= datetime.now(timezone.utc) - timedelta(days=7))
    ).all()
    
    all_feedback = s.exec(select(UserFeedback).order_by(UserFeedback.created_at).limit(50)).all()
    
    # Calculate averages
    ratings = s.exec(select(UserFeedback).where(UserFeedback.overall_rating != None)).all()
    avg_overall = sum(f.overall_rating for f in ratings if f.overall_rating is not None) / len(ratings) if ratings else 0
    
    ease_ratings = s.exec(select(UserFeedback).where(UserFeedback.ease_rating != None)).all()
    avg_ease = sum(f.ease_rating for f in ease_ratings if f.ease_rating is not None) / len(ease_ratings) if ease_ratings else 0
    
    feedback_stats = {
        'total_feedback': total_feedback,
        'avg_overall_rating': avg_overall,
        'avg_ease_rating': avg_ease,
        'recent_count': len(recent_feedback)
    }
    
    return templates.TemplateResponse("admin/feedback_dashboard.html", {
        "request": request,
        "user": user,
        "feedback_stats": feedback_stats,
        "recent_feedback": all_feedback
    })

@app.get("/admin/feedback/{feedback_id}")
async def get_feedback_detail(feedback_id: int, request: Request, user: User = Depends(require_user), s: Session = Depends(db)):
    """Get detailed feedback information"""
    if not user.is_admin:
        raise HTTPException(403, "Admin access required")
    
    feedback = s.get(UserFeedback, feedback_id)
    if not feedback:
        return {"success": False, "message": "Feedback not found"}
    
    # Convert to dict for JSON serialization
    feedback_dict = {
        "id": feedback.id,
        "workflow": feedback.workflow,
        "context": feedback.context,
        "overall_rating": feedback.overall_rating,
        "ease_rating": feedback.ease_rating,
        "feedback_text": feedback.feedback_text,
        "suggestions": feedback.suggestions,
        "issues": feedback.issues,
        "contact_ok": feedback.contact_ok,
        "page_url": feedback.page_url,
        "created_at": feedback.created_at.isoformat()
    }
    
    return {"success": True, "feedback": feedback_dict}

# Admin Notification Settings Routes
@app.get("/admin/notifications", response_class=HTMLResponse)
def notification_settings(request: Request, s: Session = Depends(db)):
    """Admin page for configuring notification settings"""
    user = require_admin()(s, request.cookies.get("session"))
    
    try:
        from notification_service import get_notification_service
        notification_service = get_notification_service(s)
        settings = notification_service.get_notification_settings()
        
        # Get notification statistics
        notification_logs = s.exec(select(NotificationLog)).all()
        total_sent = len([log for log in notification_logs if log.status == "sent"])
        total_failed = len([log for log in notification_logs if log.status == "failed"])
        
        # Recent notifications
        recent_notifications = s.exec(
            select(NotificationLog)
            .order_by(NotificationLog.sent_at.desc())
            .limit(20)
        ).all()
        
        notification_stats = {
            "total_sent": total_sent,
            "total_failed": total_failed,
            "success_rate": round((total_sent / (total_sent + total_failed) * 100), 1) if (total_sent + total_failed) > 0 else 0,
            "recent_count": len(recent_notifications)
        }
        
    except ImportError:
        settings = {}
        notification_stats = {}
        recent_notifications = []
    
    return templates.TemplateResponse("admin/notification_settings.html", {
        "request": request,
        "user": user,
        "settings": settings,
        "notification_stats": notification_stats,
        "recent_notifications": recent_notifications
    })

@app.post("/admin/notifications/update")
def update_notification_setting(
    request: Request,
    setting_key: str = Form(...),
    setting_value: str = Form(...),
    s: Session = Depends(db)
):
    """Update a notification setting"""
    user = require_admin()(s, request.cookies.get("session"))
    
    try:
        from notification_service import get_notification_service
        notification_service = get_notification_service(s)
        
        # Parse boolean values
        if setting_value.lower() in ['true', 'false']:
            parsed_value = setting_value.lower() == 'true'
        else:
            # Try to parse as int, fallback to string
            try:
                parsed_value = int(setting_value)
            except ValueError:
                parsed_value = setting_value
        
        notification_service.update_notification_setting(
            setting_key=setting_key,
            setting_value=parsed_value,
            updated_by_user_id=user.id
        )
        
        return RedirectResponse(url="/admin/notifications?updated=success", status_code=303)
        
    except ImportError:
        return RedirectResponse(url="/admin/notifications?error=service_unavailable", status_code=303)
    except Exception as e:
        return RedirectResponse(url=f"/admin/notifications?error={str(e)}", status_code=303)

@app.post("/notifications/mark-seen")
def mark_notifications_seen(
    request: Request,
    s: Session = Depends(db)
):
    """Mark user's notifications as seen (clears 'New Results' badges)"""
    try:
        user = require_user()(s, request.cookies.get("session"))
        # For now, just return success - the notification system can be enhanced later
        return {"success": True, "message": "Notifications marked as seen"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/current-user")
def get_current_user_api(
    request: Request,
    s: Session = Depends(db)
):
    """Get current user information for API calls"""
    try:
        user = get_current_user(s, request.cookies.get("session"))
        if user:
            return {
                "success": True,
                "id": user.id,
                "name": user.name,
                "email": user.email
            }
        else:
            return {"success": False, "error": "Not logged in"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# AI Suggestions API endpoints
@app.post("/api/suggest_trial_config")
async def api_suggest_trial_config(
    title: str = Form(),
    domain: str = Form(), 
    description: str = Form(default=""),
    s: Session = Depends(db),
    user: User = Depends(require_user)
):
    """Get AI suggestions for trial configuration."""
    if not AI_ENABLED or 'suggest_trial_configuration' not in globals():
        return {"error": "AI suggestions require OpenAI API key", "success": False}
    
    try:
        suggestions = suggest_trial_configuration(title, domain, description)
        return suggestions
    except Exception as e:
        return {"error": f"AI suggestion failed: {str(e)}", "success": False}

@app.post("/api/suggest_targets")
async def api_suggest_targets(
    domain: str = Form(),
    outcomes: str = Form(),  # JSON string of outcome names
    s: Session = Depends(db),
    user: User = Depends(require_user)
):
    """Get AI suggestions for target selection."""
    if not AI_ENABLED or 'suggest_target_selection' not in globals():
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
    user: User = Depends(require_user)
):
    """Get AI analysis of trial timing."""
    if not AI_ENABLED or 'suggest_timing_optimization' not in globals():
        return {"error": "AI suggestions require OpenAI API key", "success": False}
    
    try:
        suggestions = suggest_timing_optimization(title, domain, proposed_time)
        return suggestions
    except Exception as e:
        return {"error": f"AI timing analysis failed: {str(e)}", "success": False}

@app.post("/api/analyze_viability")
async def api_analyze_viability(
    title: str = Form(),
    domain: str = Form(),
    description: str = Form(default=""),
    outcomes: str = Form(),  # JSON string
    s: Session = Depends(db),
    user: User = Depends(require_user)
):
    """Get AI analysis of trial viability."""
    if not AI_ENABLED or 'analyze_trial_viability' not in globals():
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
    Uses enhanced algorithm to maximize visual distinctiveness for optimal ARV discrimination.
    
    Selection criteria prioritize:
    - Maximum visual contrast (animate vs inanimate, natural vs artificial, etc.)
    - Opposing sensory characteristics (bright vs dark, warm vs cool, textured vs smooth)
    - Diverse conceptual categories to enhance remote viewing impressions
    - Avoidance of recently used targets for variety
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
    
    # Enhanced selection algorithm: maximize visual distinctiveness for ARV
    selected = []
    
    # Parse tags for each target and categorize by visual characteristics
    target_characteristics = {}
    visual_categories = {
        'animate': ['animal', 'person', 'human', 'wildlife', 'bird', 'cat', 'dog', 'face', 'people'],
        'inanimate': ['object', 'tool', 'building', 'vehicle', 'machine', 'furniture', 'architecture'],
        'natural': ['landscape', 'nature', 'tree', 'mountain', 'water', 'sky', 'outdoor', 'forest'],
        'artificial': ['technology', 'electronic', 'urban', 'indoor', 'synthetic', 'manufactured'],
        'geometric': ['geometric', 'pattern', 'abstract', 'shape', 'design', 'symmetrical'],
        'organic': ['organic', 'flowing', 'curved', 'irregular', 'natural', 'biological'],
        'bright': ['bright', 'colorful', 'vivid', 'light', 'white', 'yellow', 'orange'],
        'dark': ['dark', 'black', 'shadow', 'night', 'deep', 'dim'],
        'warm': ['warm', 'red', 'orange', 'yellow', 'fire', 'sun'],
        'cool': ['cool', 'blue', 'green', 'cold', 'ice', 'water'],
        'textured': ['textured', 'rough', 'bumpy', 'grainy', 'detailed'],
        'smooth': ['smooth', 'clean', 'simple', 'minimal', 'sleek']
    }
    
    for target in available_targets:
        tags = set()
        if target.tags:
            tags = set(tag.strip().lower() for tag in target.tags.split(','))
        
        # Categorize target based on its tags
        characteristics = set()
        for category, keywords in visual_categories.items():
            if any(keyword in tag for tag in tags for keyword in keywords):
                characteristics.add(category)
        
        # If no specific characteristics found, infer from common tags
        if not characteristics:
            if any(word in ' '.join(tags) for word in ['red', 'blue', 'green', 'color']):
                characteristics.add('colorful')
            else:
                characteristics.add('neutral')
        
        target_characteristics[target.id] = characteristics
    
    # Select first target with strongest characteristics
    target_scores = []
    for target in available_targets:
        char_count = len(target_characteristics.get(target.id, set()))
        target_scores.append((char_count, target))
    
    # Sort by characteristic strength and select diverse first target
    target_scores.sort(key=lambda x: x[0], reverse=True)
    first_target = target_scores[0][1]
    selected.append(first_target)
    available_targets.remove(first_target)
    
    # For remaining targets, select those with maximum visual contrast
    while len(selected) < num_targets and available_targets:
        best_target = None
        max_distinctiveness = -1
        
        for candidate in available_targets:
            candidate_chars = target_characteristics.get(candidate.id, set())
            
            # Calculate distinctiveness as contrast with ALL selected targets
            total_distinctiveness = 0
            
            for selected_target in selected:
                selected_chars = target_characteristics.get(selected_target.id, set())
                
                # Count opposing characteristics (maximum contrast)
                opposing_pairs = [
                    ('animate', 'inanimate'),
                    ('natural', 'artificial'), 
                    ('geometric', 'organic'),
                    ('bright', 'dark'),
                    ('warm', 'cool'),
                    ('textured', 'smooth')
                ]
                
                contrast_score = 0
                for pair in opposing_pairs:
                    if (pair[0] in candidate_chars and pair[1] in selected_chars) or \
                       (pair[1] in candidate_chars and pair[0] in selected_chars):
                        contrast_score += 2  # High score for opposing characteristics
                
                # Penalize shared characteristics (reduce similarity)
                shared_chars = len(candidate_chars & selected_chars)
                contrast_score -= shared_chars
                
                # Bonus for completely different characteristic sets
                if candidate_chars and selected_chars and not (candidate_chars & selected_chars):
                    contrast_score += 1
                
                total_distinctiveness += max(0, contrast_score)
            
            # Prefer targets with more distinctive characteristics overall
            distinctiveness = total_distinctiveness + len(candidate_chars) * 0.1
            
            if distinctiveness > max_distinctiveness:
                max_distinctiveness = distinctiveness
                best_target = candidate
        
        if best_target:
            selected.append(best_target)
            available_targets.remove(best_target)
        else:
            # Fallback: select target with most unique characteristics
            remaining_chars = {}
            for target in available_targets:
                chars = target_characteristics.get(target.id, set())
                used_chars = set()
                for sel_target in selected:
                    used_chars.update(target_characteristics.get(sel_target.id, set()))
                unique_chars = len(chars - used_chars)
                remaining_chars[target] = unique_chars
            
            if remaining_chars:
                best_fallback = max(remaining_chars.items(), key=lambda x: x[1])[0]
                selected.append(best_fallback)
                available_targets.remove(best_fallback)
            else:
                # Final fallback: random selection
                selected.append(random.choice(available_targets))
                available_targets.remove(selected[-1])
    
    return selected


@app.post("/create_trial", response_model=None)
async def create_trial_user(request: Request,
                 title: str = Form(),
                 domain: str = Form(),
                 event_spec: str = Form(),
                 result_time: str = Form(),
                 draft_seconds: int = Form(default=300),
                 target_strategy: str = Form(default="auto"),
                 auto_start: bool = Form(default=False),
                 auto_select_targets: bool = Form(default=True),
                 # Sports-specific fields (dynamic outcomes)
                 sports_outcomes_count: int = Form(default=0),
                 # Legacy fields for backward compatibility
                 outcome_a_name: str = Form(default=""),
                 outcome_b_name: str = Form(default=""),
                 outcome_c_name: str = Form(default=""),
                 # Binary outcome fields
                 target_a_id: Optional[int] = Form(default=None),
                 target_b_id: Optional[int] = Form(default=None),
                 # Lottery fields
                 lottery_balls: int = Form(default=6),
                 # Pricing fields
                 tasking_type: str = Form(default="individual"),  # individual or group
                 s: Session = Depends(db),
                 user: User = Depends(require_user)):
    
    try:
        # Check pricing and credits for individual taskings
        if tasking_type == "individual":
            if not user.can_create_individual_tasking:
                # Redirect to purchase page if no credits
                return RedirectResponse(url=f"/purchase-credits?reason=no_credits&trial_title={title}", status_code=303)
            
            # Use credit for individual tasking
            user.use_individual_tasking_credit()
            s.add(user)  # Update user in session
        # Parse result datetime from ISO format (from HTML datetime-local)
        result_time_utc = datetime.fromisoformat(result_time).replace(tzinfo=timezone.utc)
        
        # Parse event_spec if it's JSON string
        try:
            event_spec_dict = json.loads(event_spec)
            event_description = event_spec_dict.get('description', event_spec)
        except (json.JSONDecodeError, TypeError):
            event_description = event_spec
        
        # Create trial (only individual taskings can be created now)
        trial = Trial(
            title=title,
            domain=domain,
            event_spec_json=json.dumps({"description": event_description}),
            result_time_utc=result_time_utc,
            status="draft",
            created_by=user.id,
            draft_seconds=draft_seconds,
            is_group_tasking=False,
            is_open_for_joining=False,
            participant_count=1
        )
        s.add(trial)
        s.flush()
        trial_id = trial.id
        
        # Add creator as first participant for individual taskings
        creator_participant = TrialParticipant(
            trial_id=trial.id,
            user_id=user.id,
            is_creator=True,
            credited_creator=False  # Creator doesn't credit themselves
        )
        s.add(creator_participant)
        
        # Create domain-specific outcomes
        if domain == "lottery":
            # Generate lottery ball outcomes
            for ball_num in range(1, lottery_balls + 1):
                outcome = TrialOutcome(trial_id=trial.id, label=f"Ball {ball_num}")
                s.add(outcome)
            s.flush()
            
            # Auto-assign targets to outcomes (randomly shuffle for variety)
            outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id)).all()
            available_targets = s.exec(select(Target)).all()
            if len(available_targets) < lottery_balls:
                raise HTTPException(400, f"Not enough targets available. Need {lottery_balls}, have {len(available_targets)}")
                
            # Auto-assign targets to outcomes (randomly shuffle for variety)
            import random
            shuffled_targets = random.sample(available_targets, lottery_balls)
            for outcome, target in zip(outcomes, shuffled_targets):
                s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome.id, target_id=target.id))
                
        elif domain == "sports":
            # Handle dynamic sports outcomes
            sports_outcomes = []
            
            # Check if we have new dynamic sports outcomes
            if sports_outcomes_count > 0:
                # Get dynamic sports outcomes from form data
                form = await request.form()
                for i in range(sports_outcomes_count):
                    outcome_name = form.get(f"sports_outcome_{i}")
                    if outcome_name and outcome_name.strip():
                        sports_outcomes.append(outcome_name.strip())
            else:
                # Fallback to legacy 3-outcome format for backward compatibility
                if outcome_a_name or outcome_b_name or outcome_c_name:
                    if outcome_a_name: sports_outcomes.append(outcome_a_name)
                    if outcome_b_name: sports_outcomes.append(outcome_b_name)
                    if outcome_c_name: sports_outcomes.append(outcome_c_name)
                else:
                    # Default outcomes if none provided
                    sports_outcomes = ["Team A", "Team B"]
                    
            # Validate sports outcomes
            if len(sports_outcomes) < 2:
                raise HTTPException(400, "Sports trials need at least 2 outcomes/teams")
            if len(sports_outcomes) > 20:
                raise HTTPException(400, "Sports trials can have maximum 20 outcomes/teams")
                
            # Create outcomes for each team/result
            created_outcomes = []
            for i, outcome_name in enumerate(sports_outcomes):
                outcome = TrialOutcome(trial_id=trial.id, label=outcome_name)
                s.add(outcome)
                created_outcomes.append(outcome)
            s.flush()
            
            # Auto-assign targets for sports outcomes
            available_targets = s.exec(select(Target)).all()
            num_outcomes = len(sports_outcomes)
            if len(available_targets) < num_outcomes:
                raise HTTPException(400, f"Not enough targets available. Need {num_outcomes}, have {len(available_targets)}")
                
            import random
            shuffled_targets = random.sample(available_targets, num_outcomes)
            for outcome, target in zip(created_outcomes, shuffled_targets):
                s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome.id, target_id=target.id))
            
        else:
            # Binary outcomes for other domains (stocks, etc.)
            if not target_a_id or not target_b_id:
                # Auto-assign targets if not specified
                available_targets = s.exec(select(Target)).all()
                if len(available_targets) < 2:
                    raise HTTPException(400, "Not enough targets available. Need at least 2 targets.")
                    
                import random
                shuffled_targets = random.sample(available_targets, 2)
                target_a_id = shuffled_targets[0].id
                target_b_id = shuffled_targets[1].id
                
            outcome_a = TrialOutcome(trial_id=trial.id, label=outcome_a_name or "A")
            outcome_b = TrialOutcome(trial_id=trial.id, label=outcome_b_name or "B")
            s.add(outcome_a)
            s.add(outcome_b)
            s.flush()
            
            # Link targets
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_a.id, target_id=target_a_id))
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_b.id, target_id=target_b_id))
        
        s.commit()
        # Clear homepage cache to show new trial
        _get_cached_homepage_data.cache_clear()
        
        # Always redirect to trial detail page - wizard will handle sharing step internally
        return RedirectResponse(f"/trials/{trial_id}", status_code=302)
        
    except Exception as e:
        s.rollback()
        raise HTTPException(400, f"Failed to create trial: {str(e)}")

@app.get("/trials/{trial_id}/share", response_class=HTMLResponse)
def group_task_sharing_page(trial_id: int, request: Request, s: Session = Depends(db)):
    """Sharing encouragement page for group task creators"""
    # Check if user is logged in, redirect to login if not
    user = get_current_user(s, request.cookies.get("session"))
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Verify user is the creator of this trial
    if trial.created_by != user.id:
        raise HTTPException(403, "You can only access sharing page for trials you created")
    
    # Only show sharing page for group tasks
    if not trial.is_group_tasking:
        return RedirectResponse(f"/trials/{trial_id}", status_code=302)
    
    return templates.TemplateResponse("group_task_sharing.html", {
        "request": request,
        "user": user,
        "trial": trial
    })

@app.post("/admin/create_trial", response_model=None)
async def create_trial(request: Request,
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
                tasking_type: str = Form(default="individual"),
                max_participants: Optional[int] = Form(default=None),
                s: Session = Depends(db),
                user: User = Depends(require_user)):
    
    # Parse datetime
    result_time_utc = datetime.fromisoformat(result_time.replace('Z', '+00:00'))
    
    # Check if user has credits for individual tasking
    if tasking_type == "individual":
        if user.tasking_credits <= 0:
            raise HTTPException(400, "You don't have enough credits for individual taskings. Purchase credits or create a group tasking instead.")
        # Deduct credit for individual tasking
        user.tasking_credits -= 1
        s.add(user)
    
    # Determine group tasking settings
    is_group = (tasking_type == "group")
    
    # Create trial
    trial = Trial(
        title=title,
        domain=domain,
        event_spec_json=event_spec,
        result_time_utc=result_time_utc,
        status="draft",
        created_by=user.id,
        is_group_tasking=is_group,
        is_open_for_joining=is_group,  # Group taskings are open for joining
        max_participants=max_participants if is_group else None,
        participant_count=1 if is_group else 1  # Creator is first participant
    )
    s.add(trial)
    s.flush()
    
    # If it's a group tasking, add creator as first participant
    if is_group:
        creator_participant = TrialParticipant(
            trial_id=trial.id,
            user_id=user.id,
            is_creator=True,
            credited_creator=False  # Creator doesn't credit themselves
        )
        s.add(creator_participant)
    
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
        
        # Get ball descriptions from form data    
        form_data = await request.form()
        ball_descriptions = {}
        for i in range(1, lottery_balls + 1):
            desc_key = f"ball_desc_{i}"
            desc_value = form_data.get(desc_key, "").strip()
            if not desc_value:
                raise HTTPException(400, f"Ball {i} description is required for lottery trials")
            ball_descriptions[i] = desc_value
            
        # Get available targets for auto-assignment
        available_targets = s.exec(select(Target)).all()
        if len(available_targets) < lottery_balls:
            raise HTTPException(400, f"Need at least {lottery_balls} targets available, but only {len(available_targets)} found")
        
        # Create outcomes for each ball number with descriptions
        outcomes = []
        for i in range(lottery_balls):
            ball_number = i + 1
            ball_desc = ball_descriptions.get(ball_number, f"Ball {ball_number}")
            outcome = TrialOutcome(trial_id=trial.id, label=f"Ball {ball_number}: {ball_desc}")
            s.add(outcome)
            outcomes.append(outcome)
        s.flush()
        
        # Auto-assign targets to outcomes (randomly shuffle for variety)
        import random
        shuffled_targets = random.sample(available_targets, lottery_balls)
        for outcome, target in zip(outcomes, shuffled_targets):
            s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome.id, target_id=target.id))
    else:
        # Binary outcomes for other domains (stocks, etc.) - now auto-selected
        if not target_a_id or not target_b_id:
            if auto_select_targets:
                # This should already be handled above, but safety check
                backup_targets = auto_select_distinctive_targets(s, 2)
                if len(backup_targets) >= 2:
                    target_a_id = backup_targets[0].id if not target_a_id else target_a_id
                    target_b_id = backup_targets[1].id if not target_b_id else target_b_id
            
            if not target_a_id or not target_b_id:
                raise HTTPException(400, "Unable to auto-select targets. Please ensure targets are available in the database.")
            
        outcome_a = TrialOutcome(trial_id=trial.id, label="A")
        outcome_b = TrialOutcome(trial_id=trial.id, label="B")
        s.add(outcome_a)
        s.add(outcome_b)
        s.flush()
        
        # Link targets
        s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_a.id, target_id=target_a_id))
        s.add(TrialTarget(trial_id=trial.id, outcome_id=outcome_b.id, target_id=target_b_id))
    
    s.commit()
    # Clear homepage cache to show new trial
    _get_cached_homepage_data.cache_clear()
    return RedirectResponse(f"/trials/{trial.id}/edit", status_code=302)

@app.get("/trials/{trial_id}/edit", response_class=HTMLResponse)
def edit_trial_form(trial_id: int, request: Request, user: User = Depends(require_user), s: Session = Depends(db)):
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Allow trial creator or admin to edit draft trials
    # Admins can edit ANY trial status, regular users only draft
    if user.role != "admin":
        if trial.status != "draft":
            return templates.TemplateResponse("error.html", {
                "request": request,
                "user": user,
                "error_title": "Cannot Edit Task",
                "error_message": "Only draft tasks can be edited by regular users. This task is currently in '{}' status. Contact an admin for changes to active tasks.".format(trial.status.title()),
                "error_details": "Tasks can only be modified while they are in draft status. Once a task is started or becomes live, editing is disabled to maintain the integrity of the prediction process.",
                "trial_id": trial_id
            })
        # Non-admin users can only edit their own trials
        if trial.created_by != user.id:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "user": user,
                "error_title": "Cannot Edit Task",
                "error_message": "You can only edit tasks you created."
            })
    
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
async def update_trial(trial_id: int,
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
                auto_select_targets: bool = Form(default=True),
                s: Session = Depends(db),
                user: User = Depends(require_user)):
    
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Allow trial creator or admin to edit draft trials  
    # Admins can edit ANY trial status, regular users only draft
    if user.role != "admin":
        if trial.status != "draft":
            raise HTTPException(400, "Only draft trials can be edited by regular users. Contact an admin for changes to active tasks.")
        # Non-admin users can only edit their own trials
        if trial.created_by != user.id:
            raise HTTPException(403, "You can only edit tasks you created.")
    
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
    
    # Auto-select targets if enabled
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
        
        # Link targets (auto-selected or manual)
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
        
        # Get ball descriptions from form data    
        form_data = await request.form()
        ball_descriptions = {}
        for i in range(1, lottery_balls + 1):
            desc_key = f"ball_desc_{i}"
            desc_value = form_data.get(desc_key, "").strip()
            if not desc_value:
                raise HTTPException(400, f"Ball {i} description is required for lottery trials")
            ball_descriptions[i] = desc_value
            
        # Get available targets for auto-assignment
        available_targets = s.exec(select(Target)).all()
        if len(available_targets) < lottery_balls:
            raise HTTPException(400, f"Need at least {lottery_balls} targets available, but only {len(available_targets)} found")
        
        # Create outcomes for each ball number with descriptions
        outcomes = []
        for i in range(lottery_balls):
            ball_number = i + 1
            ball_desc = ball_descriptions.get(ball_number, f"Ball {ball_number}")
            outcome = TrialOutcome(trial_id=trial.id, label=f"Ball {ball_number}: {ball_desc}")
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
    # Clear homepage cache to show updated trial
    _get_cached_homepage_data.cache_clear()
    return RedirectResponse(f"/trials/{trial_id}", status_code=302)

@app.post("/trials/{trial_id}/start")
def start_trial(trial_id: int,
               s: Session = Depends(db),
               user: User = Depends(require_user)):
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Allow trial creator or admin to start the trial
    if user.role != "admin" and trial.created_by != user.id:
        raise HTTPException(403, "Only the trial creator or admin can start this trial")
    
    trial.status = "open"
    trial.live_start_utc = datetime.now(timezone.utc)
    trial.live_end_utc = datetime.now(timezone.utc) + timedelta(seconds=trial.draft_seconds or 300)
    s.commit()
    # Clear homepage cache to show status change
    _get_cached_homepage_data.cache_clear()
    
    return {"status": "started"}

# ==================== ADMIN TRIAL MANAGEMENT ROUTES ====================

@app.delete("/admin/trials/{trial_id}")
def admin_delete_trial(trial_id: int, s: Session = Depends(db), user: User = Depends(require_admin())):
    """Admin-only route to delete trials with all related data"""
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    try:
        # Delete all related data in proper order to avoid foreign key constraints
        
        # Delete predictions
        predictions = s.exec(select(Prediction).where(Prediction.trial_id == trial_id)).all()
        for prediction in predictions:
            s.delete(prediction)
        
        # Delete judgments
        judgments = s.exec(select(Judgment).where(Judgment.trial_id == trial_id)).all()
        for judgment in judgments:
            s.delete(judgment)
        
        # Delete consensus descriptors
        descriptors = s.exec(select(ConsensusDescriptor).where(ConsensusDescriptor.trial_id == trial_id)).all()
        for descriptor in descriptors:
            s.delete(descriptor)
        
        # Delete trial targets
        trial_targets = s.exec(select(TrialTarget).where(TrialTarget.trial_id == trial_id)).all()
        for tt in trial_targets:
            s.delete(tt)
        
        # Delete outcomes
        outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id)).all()
        for outcome in outcomes:
            s.delete(outcome)
        
        # Delete results
        result = s.exec(select(Result).where(Result.trial_id == trial_id)).first()
        if result:
            s.delete(result)
        
        # Delete aggregates
        aggregate = s.exec(select(Aggregate).where(Aggregate.trial_id == trial_id)).first()
        if aggregate:
            s.delete(aggregate)
        
        # Finally delete the trial itself
        s.delete(trial)
        s.commit()
        # Clear homepage cache to remove deleted trial
        _get_cached_homepage_data.cache_clear()
        
        return {"success": True, "message": f"Trial '{trial.title}' and all related data deleted successfully"}
        
    except Exception as e:
        s.rollback()
        return {"success": False, "message": f"Failed to delete trial: {str(e)}"}

@app.delete("/admin/trials/bulk")
def admin_bulk_delete_trials(s: Session = Depends(db), user: User = Depends(require_admin())):
    """Admin-only route to delete all trials and their related data"""
    
    try:
        # Get all trials
        all_trials = s.exec(select(Trial)).all()
        
        if not all_trials:
            return {"success": True, "message": "No trials found to delete"}
        
        deleted_count = 0
        deleted_titles = []
        
        for trial in all_trials:
            trial_id = trial.id
            trial_title = trial.title
            
            # Delete all related data in proper order to avoid foreign key constraints
            
            # Delete predictions
            predictions = s.exec(select(Prediction).where(Prediction.trial_id == trial_id)).all()
            for prediction in predictions:
                s.delete(prediction)
            
            # Delete judgments
            judgments = s.exec(select(Judgment).where(Judgment.trial_id == trial_id)).all()
            for judgment in judgments:
                s.delete(judgment)
            
            # Delete consensus descriptors
            descriptors = s.exec(select(ConsensusDescriptor).where(ConsensusDescriptor.trial_id == trial_id)).all()
            for descriptor in descriptors:
                s.delete(descriptor)
            
            # Delete trial targets
            trial_targets = s.exec(select(TrialTarget).where(TrialTarget.trial_id == trial_id)).all()
            for tt in trial_targets:
                s.delete(tt)
            
            # Delete outcomes
            outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id)).all()
            for outcome in outcomes:
                s.delete(outcome)
            
            # Delete results
            result = s.exec(select(Result).where(Result.trial_id == trial_id)).first()
            if result:
                s.delete(result)
            
            # Delete aggregates
            aggregate = s.exec(select(Aggregate).where(Aggregate.trial_id == trial_id)).first()
            if aggregate:
                s.delete(aggregate)
            
            # Finally delete the trial itself
            s.delete(trial)
            
            deleted_count += 1
            deleted_titles.append(trial_title)
        
        s.commit()
        
        return {
            "success": True, 
            "message": f"Successfully deleted {deleted_count} trials: {', '.join(deleted_titles)}"
        }
        
    except Exception as e:
        s.rollback()
        return {"success": False, "message": f"Failed to bulk delete trials: {str(e)}"}

@app.post("/admin/trials/{trial_id}/force-settle")
def admin_force_settle_trial(
    trial_id: int, 
    winning_outcome_id: int = Form(),
    s: Session = Depends(db), 
    user: User = Depends(require_admin())
):
    """Admin-only route to manually settle a trial"""
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Verify the winning outcome exists and belongs to this trial
    winning_outcome = s.exec(
        select(TrialOutcome)
        .where(TrialOutcome.id == winning_outcome_id)
        .where(TrialOutcome.trial_id == trial_id)
    ).first()
    
    if not winning_outcome:
        raise HTTPException(400, "Invalid winning outcome for this trial")
    
    try:
        # Update trial status
        trial.status = "settled"
        trial.result_time_utc = datetime.now(timezone.utc)
        
        # Create or update result
        existing_result = s.exec(select(Result).where(Result.trial_id == trial_id)).first()
        if existing_result:
            existing_result.outcome_id_won = winning_outcome_id
            existing_result.admin_settled = True
        else:
            result = Result(
                trial_id=trial_id,
                outcome_id_won=winning_outcome_id,
                admin_settled=True
            )
            s.add(result)
        
        # Update prediction accuracy for all participants
        predictions = s.exec(select(Prediction).where(Prediction.trial_id == trial_id)).all()
        for prediction in predictions:
            prediction.is_correct = (prediction.outcome_id == winning_outcome_id)
            
            # Update user statistics
            user_obj = s.get(User, prediction.user_id)
            if user_obj:
                # Reward credit for correct prediction
                if prediction.is_correct:
                    user_obj.reward_accuracy_credit(1)
                
                # Recalculate user's correct predictions count
                correct_count = s.exec(
                    select(func.count(Prediction.id))
                    .where(Prediction.user_id == user_obj.id)
                    .where(Prediction.is_correct == True)
                ).first() or 0
                
                user_obj.correct_predictions = correct_count
                user_obj.total_points = correct_count * 10  # Simple scoring system
        
        s.commit()
        
        # Send notifications to participants
        try:
            from notification_service import schedule_task_conclusion_notification
            schedule_task_conclusion_notification(s, trial_id)
        except ImportError:
            print("Notification service not available")
        except Exception as e:
            print(f"Failed to send notifications: {e}")
        
        # Clear homepage cache to show settled trial
        _get_cached_homepage_data.cache_clear()
        return {"success": True, "message": f"Trial manually settled with outcome: {winning_outcome.label}"}
        
    except Exception as e:
        s.rollback()
        raise HTTPException(500, f"Failed to settle trial: {str(e)}")


@app.post("/trials/{trial_id}/predict", response_model=None)
def submit_prediction(trial_id: int,
                     outcome_id: int = Form(),
                     s: Session = Depends(db),
                     user: User = Depends(require_user)):
    
    # Check if trial exists and is open
    trial = s.get(Trial, trial_id)
    if not trial or trial.status not in ['open', 'live']:
        raise HTTPException(400, "Trial is not accepting predictions")
    
    # Check if user already made a prediction
    existing = s.exec(
        select(Prediction)
        .where(Prediction.trial_id == trial_id)
        .where(Prediction.user_id == user.id)
    ).first()
    
    if existing:
        # Update existing prediction
        existing.outcome_id = outcome_id
    else:
        # Create new prediction
        prediction = Prediction(
            trial_id=trial_id,
            user_id=user.id,
            outcome_id=outcome_id
        )
        s.add(prediction)
        # Update user's total predictions
        user.total_predictions += 1
        s.add(user)
    
    s.commit()
    return RedirectResponse(f"/trials/{trial_id}", status_code=302)

@app.post("/trials/{trial_id}/ai_judge", response_model=None)
async def ai_judge_trial(trial_id: int,
                        s: Session = Depends(db),
                        user: User = Depends(require_admin())):
    """AI-powered automatic judging that analyzes descriptors against targets to determine the winner"""
    
    if not openai_client:
        raise HTTPException(400, "OpenAI API key required for AI judging")
    
    # Get trial and validate
    trial = s.get(Trial, trial_id)
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    if trial.status == "settled":
        raise HTTPException(400, "Trial already settled")
        
    if trial.status not in ['live']:
        raise HTTPException(400, "Trial must be live for AI judging")
    
    # Get trial outcomes and targets
    outcomes = s.exec(select(TrialOutcome).where(TrialOutcome.trial_id == trial_id)).all()
    if len(outcomes) < 2:
        raise HTTPException(400, "Trial needs at least 2 outcomes for judging")
    
    # Get all descriptors for this trial
    descriptors = s.exec(select(ConsensusDescriptor).where(ConsensusDescriptor.trial_id == trial_id)).all()
    if not descriptors:
        raise HTTPException(400, "No descriptors available for judging")
    
    # Get targets for each outcome
    outcome_scores = {}
    target_analyses = {}
    
    for outcome in outcomes:
        trial_targets = s.exec(
            select(TrialTarget, Target)
            .where(TrialTarget.trial_id == trial_id)
            .where(TrialTarget.outcome_id == outcome.id)
            .join(Target, TrialTarget.target_id == Target.id)
        ).all()
        
        if not trial_targets:
            outcome_scores[outcome.id] = 0.0
            continue
            
        target = trial_targets[0][1]  # Get the first target for this outcome
        
        # Analyze this target against all descriptors using AI
        accuracy_score = await analyze_target_accuracy(target, descriptors, openai_client)
        outcome_scores[outcome.id] = accuracy_score
        
        target_analyses[outcome.id] = {
            "target_description": target.description or "No description",
            "target_tags": target.tags or "No tags",
            "accuracy_percentage": accuracy_score
        }
    
    # Find the winning outcome (highest accuracy)
    winning_outcome_id = max(outcome_scores.keys(), key=lambda k: outcome_scores[k])
    winning_outcome = s.get(TrialOutcome, winning_outcome_id)
    
    # Create result record
    result = Result(
        trial_id=trial_id,
        outcome_label_won=winning_outcome.label,
        outcome_id_won=winning_outcome_id
    )
    s.add(result)
    
    # Update trial status
    trial.status = "settled"
    s.add(trial)
    
    # Calculate points for predictions
    predictions = s.exec(
        select(Prediction).where(Prediction.trial_id == trial_id)
    ).all()
    
    for prediction in predictions:
        # Check if prediction is correct
        is_correct = prediction.outcome_id == winning_outcome_id
        prediction.is_correct = is_correct
        
        # Award points (100 for correct, 0 for incorrect)
        if is_correct:
            prediction.points_awarded = 100
            # Update user statistics
            user_obj = s.get(User, prediction.user_id)
            if user_obj:
                # Reward credit for correct prediction
                user_obj.reward_accuracy_credit(1)
                user_obj.correct_predictions += 1
                user_obj.total_points += 100
                s.add(user_obj)
        else:
            prediction.points_awarded = 0
        
        s.add(prediction)
    
    s.commit()
    
    # Send notifications to participants
    try:
        from notification_service import notify_task_conclusion
        import asyncio
        asyncio.create_task(notify_task_conclusion(s, trial_id))
    except ImportError:
        print("Notification service not available")
    except Exception as e:
        print(f"Failed to send notifications: {e}")
    
    # Clear homepage cache to show settled trial
    _get_cached_homepage_data.cache_clear()
    return RedirectResponse(f"/trials/{trial_id}", status_code=302)

async def analyze_target_accuracy(target, descriptors, client):
    """Analyze how well the descriptors match a target using efficient pre-computed embeddings + AI analysis"""
    
    # First try fast CLIP-based similarity scoring if embeddings are available
    if hasattr(target, 'image_embed_json') and target.image_embed_json:
        try:
            clip_score = compute_clip_similarity(target, descriptors)
            if clip_score is not None:
                print(f"Using CLIP similarity score: {clip_score:.1f}%")
                return clip_score
        except Exception as e:
            print(f"CLIP analysis failed: {e}")
    
    # Fallback to AI visual analysis for more detailed scoring
    return await analyze_target_with_ai_vision(target, descriptors, client)

def compute_clip_similarity(target, descriptors):
    """Fast CLIP-based similarity scoring using pre-computed embeddings"""
    
    if not (_HAS_OPEN_CLIP or _HAS_CLIP):
        return None
        
    try:
        # Combine all descriptors into analysis text
        descriptor_texts = []
        for desc in descriptors:
            category_prefix = f"{desc.category}: " if desc.category else ""
            descriptor_texts.append(f"{category_prefix}{desc.text}")
        
        combined_descriptors = ". ".join(descriptor_texts)
        
        if _HAS_OPEN_CLIP:
            import open_clip
            import torch
            
            # Load CLIP model (you might want to cache this)
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # Get text embedding for descriptors
            text = tokenizer([combined_descriptors])
            
            with torch.no_grad():
                text_embedding = model.encode_text(text)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            
            # Load target's pre-computed visual embedding
            if target.image_embed_json:
                import json
                target_embedding = torch.tensor(json.loads(target.image_embed_json))
                target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = torch.cosine_similarity(text_embedding, target_embedding, dim=1)
                
                # Convert to percentage (0-100)
                percentage = float(similarity.item()) * 50 + 50  # Scale from [-1,1] to [0,100]
                return max(0, min(100, percentage))
        
        return None
        
    except Exception as e:
        print(f"CLIP similarity computation failed: {e}")
        return None

async def analyze_target_with_ai_vision(target, descriptors, client):
    """Detailed AI vision analysis when CLIP isn't available or for higher accuracy"""
    
    # Prepare descriptors text grouped by category
    descriptor_text = ""
    categories = {}
    
    for desc in descriptors:
        category = desc.category or "general"
        if category not in categories:
            categories[category] = []
        categories[category].append(desc.text)
    
    # Format descriptors by category for AI analysis
    for category, texts in categories.items():
        descriptor_text += f"\n{category.upper()}:\n"
        for text in texts:
            descriptor_text += f"- {text}\n"
    
    # Try to get actual image for visual analysis
    image_url = None
    if target.filename:
        # Construct proper image URL - check if it's a local file or external URL
        if target.filename.startswith('http'):
            image_url = target.filename  # External URL
        else:
            # Convert to base64 for local files
            try:
                import os
                import base64
                local_path = f"static/images/{target.filename}"
                if os.path.exists(local_path):
                    with open(local_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        # Determine MIME type
                        if target.filename.lower().endswith('.png'):
                            image_url = f"data:image/png;base64,{encoded_string}"
                        elif target.filename.lower().endswith('.jpg') or target.filename.lower().endswith('.jpeg'):
                            image_url = f"data:image/jpeg;base64,{encoded_string}"
                        else:
                            image_url = f"data:image/jpeg;base64,{encoded_string}"  # Default to JPEG
            except Exception as e:
                print(f"Failed to encode image {target.filename}: {e}")
                image_url = None
    
    # Prepare AI prompt for image analysis
    if image_url:
        # Visual analysis with actual image
        prompt = f"""You are an expert in Associative Remote Viewing (ARV) analysis. You will analyze an actual target image against remote viewing descriptors to determine accuracy.

REMOTE VIEWING DESCRIPTORS:
{descriptor_text}

Please look at the target image and analyze how accurately these descriptors match what you actually see. Consider:

1. VISUAL ACCURACY: Do the descriptors match the colors, shapes, patterns, lighting, and composition you see?
2. SENSORY CORRESPONDENCE: Do texture/tactile descriptors align with what the surfaces would feel like?  
3. ENERGY/ATMOSPHERE: Do energy descriptors match the mood/feeling the image conveys?
4. THEMATIC ALIGNMENT: Do general descriptors capture the essence of what's shown?

Rate accuracy as a percentage (0-100%) based on how well the descriptors correspond to what someone would actually perceive when remote viewing this target.

ARV Scoring Criteria:
- Direct visual hits (correct colors, shapes, objects): Highly weighted
- Sensory correspondences (textures, temperatures): Valuable
- Energy/mood matches: Important supporting evidence  
- Contradictory descriptors: Reduce the overall score

Respond with ONLY a number between 0 and 100 representing the percentage accuracy. No explanation needed."""

        try:
            # Use GPT-4o Vision for actual image analysis
            response = client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }],
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=10
            )
        except Exception as e:
            print(f"Vision API failed, falling back to metadata analysis: {e}")
            # Fallback to metadata-only analysis
            return await analyze_target_metadata_only(target, descriptors, client)
    else:
        # Fallback to metadata-only analysis
        return await analyze_target_metadata_only(target, descriptors, client)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=10
        )
        
        # Extract percentage from response
        result = response.choices[0].message.content.strip()
        try:
            percentage = float(result)
            return max(0, min(100, percentage))  # Ensure it's between 0-100
        except ValueError:
            # If AI doesn't return a pure number, try to extract it
            import re
            numbers = re.findall(r'\d+\.?\d*', result)
            if numbers:
                return max(0, min(100, float(numbers[0])))
            return 0.0
            
    except Exception as e:
        print(f"AI analysis error: {e}")
        return 0.0  # Default to 0 if AI analysis fails

async def analyze_target_metadata_only(target, descriptors, client):
    """Fallback function for metadata-only analysis when image analysis fails"""
    
    # Prepare descriptors text grouped by category
    descriptor_text = ""
    categories = {}
    
    for desc in descriptors:
        category = desc.category or "general"
        if category not in categories:
            categories[category] = []
        categories[category].append(desc.text)
    
    # Format descriptors by category for AI analysis
    for category, texts in categories.items():
        descriptor_text += f"\n{category.upper()}:\n"
        for text in texts:
            descriptor_text += f"- {text}\n"
    
    # Get target information from metadata
    target_info = f"Target Description: {target.description or 'No description'}\n"
    if target.tags:
        target_info += f"Target Tags: {target.tags}\n"
    
    # AI prompt for analyzing match accuracy using metadata only
    prompt = f"""You are an expert in Associative Remote Viewing (ARV) analysis. Your task is to determine how accurately a set of remote viewing descriptors match a specific target based on available metadata.

TARGET INFORMATION (metadata only):
{target_info}

REMOTE VIEWING DESCRIPTORS:
{descriptor_text}

Please analyze how well these descriptors match the target based on the available information. Consider:

1. Visual elements (colors, shapes, patterns, lighting)
2. Tactile sensations (textures, temperatures)  
3. Energy impressions (vibrational qualities, intensity)
4. Sensory details (sounds, smells)
5. Overall thematic alignment

Rate the accuracy as a percentage (0-100%) based on how well the descriptors correspond to what someone would actually perceive when viewing this target. Be realistic and objective.

Consider that in ARV:
- Direct hits on visual elements are highly significant
- Sensory correspondences (texture, energy, etc.) are valuable
- Thematic or conceptual matches are important but less weighted than sensory hits
- Contradictory descriptors reduce the overall score

Respond with ONLY a number between 0 and 100 representing the percentage accuracy. No explanation needed."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=10
        )
        
        # Extract percentage from response
        result = response.choices[0].message.content.strip()
        try:
            percentage = float(result)
            return max(0, min(100, percentage))  # Ensure it's between 0-100
        except ValueError:
            # If AI doesn't return a pure number, try to extract it
            import re
            numbers = re.findall(r'\d+\.?\d*', result)
            if numbers:
                return max(0, min(100, float(numbers[0])))
            return 0.0
            
    except Exception as e:
        print(f"AI analysis error: {e}")
        return 0.0  # Default to 0 if AI analysis fails

def precompute_target_embeddings(target):
    """Pre-compute visual embeddings for a target image to enable fast similarity matching"""
    
    if not (_HAS_OPEN_CLIP or _HAS_CLIP) or not target.uri:
        return None
        
    try:
        import os
        # Handle both local paths and URIs
        if target.uri.startswith('http'):
            # External URL - would need to download first
            print(f"External URL not supported for embedding: {target.uri}")
            return None
        else:
            # Local file path - resolve safely
            # Security: Prevent path traversal
            if '..' in target.uri or os.path.isabs(target.uri):
                print(f"Invalid target URI (security): {target.uri}")
                return None
                
            local_path = os.path.join(os.getcwd(), target.uri)
            base_dir = os.path.join(os.getcwd(), "static")
            
            # Ensure path is within static directory 
            if not os.path.commonpath([local_path, base_dir]) == base_dir:
                print(f"Target URI outside allowed directory: {target.uri}")
                return None
        
        if not os.path.exists(local_path):
            print(f"Image file not found: {local_path}")
            return None
            
        # Use the existing initialized CLIP model instead of loading a new one
        _init_clip()
        if _CLIP_MODEL is None or _CLIP_PREPROC is None:
            print("CLIP model not initialized")
            return None
            
        # Load and preprocess image
        image = Image.open(local_path).convert('RGB')
        
        if _HAS_OPEN_CLIP:
            image_tensor = _CLIP_PREPROC(image).unsqueeze(0).to(_CLIP_DEVICE)
            with torch.no_grad():
                image_embedding = _CLIP_MODEL.encode_image(image_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        else:
            image_tensor = _CLIP_PREPROC(image).unsqueeze(0).to(_CLIP_DEVICE)
            with torch.no_grad():
                image_embedding = _CLIP_MODEL.encode_image(image_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        # Convert to JSON-serializable format
        import json
        embedding_list = image_embedding.squeeze().cpu().tolist()
        return json.dumps(embedding_list)
            
    except Exception as e:
        print(f"Failed to precompute embedding for {target.filename}: {e}")
        return None
    
    return None

@app.post("/admin/preprocess_targets")
async def preprocess_target_embeddings(s: Session = Depends(db), user: User = Depends(require_admin())):
    """Admin tool to pre-compute visual embeddings for all targets"""
    
    if not (_HAS_OPEN_CLIP or _HAS_CLIP):
        return {"error": "CLIP libraries not available", "success": False}
    
    # Get all targets without embeddings
    targets = s.exec(
        select(Target).where(
            (Target.image_embed_json == None) | (Target.image_embed_json == "")
        )
    ).all()
    
    processed = 0
    failed = 0
    
    for target in targets:
        try:
            embedding = precompute_target_embeddings(target)
            if embedding:
                target.image_embed_json = embedding
                s.add(target)
                processed += 1
                print(f"Processed embedding for target {target.id}: {target.filename}")
            else:
                failed += 1
        except Exception as e:
            print(f"Failed to process target {target.id}: {e}")
            failed += 1
    
    s.commit()
    
    return {
        "success": True,
        "processed": processed,
        "failed": failed,
        "message": f"Pre-computed embeddings for {processed} targets ({failed} failed)"
    }

@app.post("/trials/{trial_id}/descriptors", response_model=None)
def add_descriptor(trial_id: int,
                  request: Request,
                  text: str = Form(),
                  category: str = Form(default="general"),
                  target_id: Optional[int] = Form(default=None),
                  outcome_id: Optional[int] = Form(default=None),
                  s: Session = Depends(db),
                  user: User = Depends(require_user)):
    
    descriptor = ConsensusDescriptor(
        trial_id=trial_id,
        text=text,
        category=category,
        target_id=target_id,
        outcome_id=outcome_id,
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
    
    # Check if this is an AJAX request (from the descriptor form)
    # AJAX requests typically include these headers
    if (request.headers.get("x-requested-with") == "XMLHttpRequest" or 
        request.headers.get("content-type", "").startswith("application/json") or
        "fetch" in request.headers.get("user-agent", "").lower()):
        return {"success": True, "message": "Descriptor added successfully"}
    
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

@app.get("/admin/homepage")
def admin_homepage(request: Request, 
                  s: Session = Depends(db), 
                  user: User = Depends(require_admin)):
    """Admin interface for editing homepage content"""
    content = {}
    homepage_content = s.exec(select(HomepageContent)).all()
    
    # Group content by section for better organization
    for item in homepage_content:
        if item.section not in content:
            content[item.section] = {}
        content[item.section][item.key] = {
            'value': item.content,
            'id': item.id
        }
    
    return templates.TemplateResponse("admin/homepage_editor.html", {
        "request": request,
        "user": user,
        "content": content
    })

@app.post("/admin/homepage/update")
def update_homepage_content(request: Request,
                           section: str = Form(),
                           key: str = Form(),
                           content_value: str = Form(),
                           s: Session = Depends(db),
                           user: User = Depends(require_admin)):
    """Update homepage content"""
    try:
        # Try to update existing content
        existing = s.exec(
            select(HomepageContent)
            .where(HomepageContent.section == section)
            .where(HomepageContent.key == key)
        ).first()
        
        if existing:
            existing.content = content_value
            existing.updated_at = datetime.now()
        else:
            # Create new content
            new_content = HomepageContent(
                section=section,
                key=key,
                content=content_value
            )
            s.add(new_content)
        
        s.commit()
        return RedirectResponse("/admin/homepage?updated=success", status_code=302)
        
    except Exception as e:
        s.rollback()
        return RedirectResponse(f"/admin/homepage?error={str(e)}", status_code=302)

def get_homepage_content(s: Session, section: str, key: str, default: str = "") -> str:
    """Helper function to get homepage content from database"""
    try:
        content = s.exec(
            select(HomepageContent)
            .where(HomepageContent.section == section)
            .where(HomepageContent.key == key)
        ).first()
        return content.content if content else default
    except:
        return default

# ---------------------------- Credit Purchase System ----------------------------

# Initialize Stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

CREDIT_PACKAGES = {
    "starter_10": {"credits": 10, "price": 9.00, "name": "Starter Pack"},
    "value_25": {"credits": 25, "price": 15.00, "name": "Value Pack"},
    "bulk_50": {"credits": 50, "price": 20.00, "name": "Bulk Pack"},
    "test": {"credits": 1, "price": 1.00, "name": "Test", "product_id": "prod_T4fnR0XngAsmGV"}
}

@app.get("/purchase-credits", response_class=HTMLResponse)
def purchase_credits_page(request: Request, 
                         reason: Optional[str] = None,
                         trial_title: Optional[str] = None,
                         s: Session = Depends(db),
                         user: User = Depends(require_user)):
    return templates.TemplateResponse("purchase_credits.html", {
        "request": request,
        "user": user,
        "packages": CREDIT_PACKAGES,
        "reason": reason,
        "trial_title": trial_title
    })

@app.post("/create-checkout-session/{package_type}")
def create_checkout_session(package_type: str,
                           request: Request,
                           s: Session = Depends(db),
                           user: User = Depends(require_user)):
    """Create Stripe checkout session for credit purchase"""
    
    if package_type not in CREDIT_PACKAGES:
        raise HTTPException(400, "Invalid package type")
    
    package = CREDIT_PACKAGES[package_type]
    
    try:
        # Get the domain for success/cancel URLs - prioritize custom domain
        custom_domain = "arvlab.xyz"  # Your stable custom domain
        host_header = request.headers.get('host')
        
        # Use custom domain if available, otherwise fall back to current domain
        if custom_domain:
            domain = custom_domain
        elif host_header and 'replit.dev' in host_header:
            domain = host_header
        else:
            # Fallback to environment variables
            domain = os.environ.get('REPLIT_DEV_DOMAIN') or os.environ.get('REPLIT_DOMAINS', '').split(',')[0]
            if not domain:
                domain = "localhost:5000"  # Fallback for local development
        
        print(f"DEBUG: Using domain for success/cancel URLs: {domain} (custom domain prioritized, from host: {host_header})")
        
        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[
                {
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f'ARVLab {package["name"]}',
                            'description': f'{package["credits"]} Tasking Credits for ARVLab',
                        },
                        'unit_amount': int(package["price"] * 100),  # Convert to cents
                    },
                    'quantity': 1,
                },
            ],
            mode='payment',
            success_url=f'https://{domain}/static/payment-success.html?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'https://{domain}/purchase-credits?reason=cancelled',
            metadata={
                'user_id': str(user.id),
                'package_type': package_type,
                'credits': str(package["credits"]),
            }
        )
        
        # Create pending purchase record
        purchase = CreditPurchase(
            user_id=user.id,
            package_type=package_type,
            credits_purchased=package["credits"],
            cost=package["price"],
            stripe_session_id=checkout_session.id,
            payment_status="pending"
        )
        s.add(purchase)
        s.commit()
        
        # Log the checkout session URL for debugging
        print(f"DEBUG: Checkout session URL: {checkout_session.url}")
        
        # For external redirects to Stripe, we need to return an HTML page that redirects
        # This ensures the browser properly follows the redirect to the external domain
        redirect_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Redirecting to Stripe...</title>
            <meta http-equiv="refresh" content="0; url={checkout_session.url}">
        </head>
        <body>
            <p>Redirecting to secure checkout...</p>
            <script>
                window.location.href = "{checkout_session.url}";
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=redirect_html)
        
    except stripe.error.StripeError as e:
        # Handle Stripe-specific errors
        raise HTTPException(400, f"Payment processing error: {str(e)}")
    except Exception as e:
        # Handle other errors
        raise HTTPException(500, f"Server error: {str(e)}")

@app.get("/payment-success", response_class=HTMLResponse)
def payment_success(request: Request, session_id: Optional[str] = None, 
                   s: Session = Depends(db)):
    """Handle successful payment redirect from Stripe"""
    
    # Handle case where user might not be logged in due to domain switching
    user = None
    try:
        user = require_user(request.cookies.get("session"), s)
    except:
        # If authentication fails, show a success page with login prompt
        return templates.TemplateResponse("payment_success.html", {
            "request": request,
            "session_id": session_id,
            "need_login": True
        })
    
    # If user is authenticated, process the success
    if session_id:
        # Verify and process the payment
        try:
            stripe_session = stripe.checkout.Session.retrieve(session_id)
            if stripe_session.payment_status == 'paid':
                # Find the purchase record and ensure credits are added
                purchase = s.exec(
                    select(CreditPurchase).where(
                        CreditPurchase.stripe_session_id == session_id,
                        CreditPurchase.user_id == user.id
                    )
                ).first()
                
                if purchase and purchase.payment_status == "pending":
                    # Update purchase status and add credits
                    purchase.payment_status = "completed"
                    purchase.completed_at = datetime.now(timezone.utc)
                    user.tasking_credits += purchase.credits_purchased
                    s.add(purchase)
                    s.add(user)
                    s.commit()
                    
                    return templates.TemplateResponse("payment_success.html", {
                        "request": request,
                        "user": user,
                        "session_id": session_id,
                        "credits_added": purchase.credits_purchased,
                        "success": True
                    })
        except Exception as e:
            print(f"Error processing payment success: {e}")
    
    # Default success page
    return templates.TemplateResponse("payment_success.html", {
        "request": request,
        "user": user,
        "session_id": session_id,
        "success": True
    })

@app.get("/payment-success-static", response_class=HTMLResponse)
def payment_success_static(request: Request, session_id: Optional[str] = None):
    """Static payment success page that works regardless of domain issues"""
    return templates.TemplateResponse("payment_success_static.html", {
        "request": request,
        "session_id": session_id
    })

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, s: Session = Depends(db)):
    """Handle Stripe webhook events"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    print(f"DEBUG: Received webhook with signature: {sig_header}")
    
    try:
        # Verify webhook signature
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.environ.get('STRIPE_WEBHOOK_SECRET')
        )
        print(f"DEBUG: Webhook verified successfully, event type: {event['type']}")
    except ValueError as e:
        print(f"DEBUG: Invalid payload: {e}")
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        print(f"DEBUG: Invalid signature: {e}")
        raise HTTPException(400, "Invalid signature")
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        print(f"DEBUG: Processing checkout session: {session['id']}")
        
        # Find the purchase record
        purchase = s.exec(
            select(CreditPurchase).where(
                CreditPurchase.stripe_session_id == session['id']
            )
        ).first()
        
        if purchase and purchase.payment_status == "pending":
            print(f"DEBUG: Found pending purchase, adding {purchase.credits_purchased} credits to user {purchase.user_id}")
            
            # Update purchase status
            purchase.payment_status = "completed"
            purchase.completed_at = datetime.now(timezone.utc)
            
            # Add credits to user account
            user = s.get(User, purchase.user_id)
            if user:
                user.tasking_credits += purchase.credits_purchased
                s.add(user)
                print(f"DEBUG: Credits added successfully. User now has {user.tasking_credits} credits")
            else:
                print(f"DEBUG: ERROR - User {purchase.user_id} not found")
            
            s.add(purchase)
            s.commit()
            print(f"DEBUG: Purchase completed successfully")
        else:
            print(f"DEBUG: Purchase not found or already processed")
    else:
        print(f"DEBUG: Ignoring webhook event type: {event['type']}")
    
    return {"status": "success"}

@app.post("/admin/process-pending-payments")
def process_pending_payments(s: Session = Depends(db)):
    """Manual endpoint to process pending payments"""
    
    # Get all pending purchases
    pending_purchases = s.exec(
        select(CreditPurchase).where(CreditPurchase.payment_status == "pending")
    ).all()
    
    processed_count = 0
    
    for purchase in pending_purchases:
        try:
            # Verify payment status with Stripe
            stripe_session = stripe.checkout.Session.retrieve(purchase.stripe_session_id)
            
            if stripe_session.payment_status == 'paid':
                print(f"Processing pending payment: {purchase.stripe_session_id}")
                
                # Update purchase status
                purchase.payment_status = "completed"
                purchase.completed_at = datetime.now(timezone.utc)
                
                # Add credits to user
                user = s.get(User, purchase.user_id)
                if user:
                    user.tasking_credits += purchase.credits_purchased
                    s.add(user)
                    print(f"Added {purchase.credits_purchased} credits to user {user.id}")
                
                s.add(purchase)
                processed_count += 1
            else:
                print(f"Payment not yet completed for session: {purchase.stripe_session_id}")
                
        except Exception as e:
            print(f"Error processing purchase {purchase.id}: {e}")
    
    s.commit()
    
    return {
        "message": f"Processed {processed_count} pending payments",
        "processed_count": processed_count
    }

@app.get("/account/credits", response_class=HTMLResponse)
def account_credits(request: Request, 
                   s: Session = Depends(db),
                   user: User = Depends(require_user)):
    # Ensure user has referral code
    if not user.referral_code:
        user.generate_referral_code()
        s.add(user)
        s.commit()
    
    # Get purchase history
    purchases = s.exec(
        select(CreditPurchase)
        .where(CreditPurchase.user_id == user.id)
        .order_by(CreditPurchase.created_at.desc())
    ).all()
    
    # Get recent referrals
    referrals = s.exec(
        select(User)
        .where(User.referred_by_user_id == user.id)
        .order_by(User.created_at.desc())
        .limit(5)
    ).all()
    
    return templates.TemplateResponse("account_credits.html", {
        "request": request,
        "user": user,
        "purchases": purchases,
        "packages": CREDIT_PACKAGES,
        "referrals": referrals,
        "referral_url": f"https://arvlab.xyz/register?ref={user.referral_code}"
    })

# ---------------------------- Referral System ----------------------------

@app.get("/referrals", response_class=HTMLResponse)
def referrals_page(request: Request, 
                  s: Session = Depends(db),
                  user: User = Depends(require_user)):
    
    # Ensure user has referral code
    if not user.referral_code:
        user.generate_referral_code()
        s.add(user)
        s.commit()
    
    # Get referral stats
    referrals = s.exec(
        select(User)
        .where(User.referred_by_user_id == user.id)
        .order_by(User.created_at.desc())
    ).all()
    
    return templates.TemplateResponse("referrals.html", {
        "request": request,
        "user": user,
        "referrals": referrals,
        "referral_url": f"https://arvlab.xyz/register?ref={user.referral_code}"
    })


# ---------------------------- Group Tasking System ----------------------------

@app.get("/join-group", response_class=HTMLResponse)
def join_group_page(request: Request, 
                   s: Session = Depends(db),
                   user: User = Depends(require_user)):
    """Display available group taskings to join"""
    # Get available group taskings that user hasn't joined yet
    user_joined_trials = s.exec(
        select(TrialParticipant.trial_id)
        .where(TrialParticipant.user_id == user.id)
    ).all()
    
    available_groups = s.exec(
        select(Trial)
        .where(Trial.is_group_tasking == True)
        .where(Trial.is_open_for_joining == True)
        .where(Trial.status.in_(["draft", "open"]))
        .where(Trial.id.not_in(user_joined_trials) if user_joined_trials else True)
        .order_by(Trial.created_at.desc())
    ).all()
    
    return templates.TemplateResponse("join_group.html", {
        "request": request,
        "user": user,
        "available_groups": available_groups
    })

@app.post("/join-trial/{trial_id}")
def join_trial(trial_id: int,
              s: Session = Depends(db),
              user: User = Depends(require_user)):
    """Join a group tasking"""
    
    # Get the trial
    trial = s.exec(select(Trial).where(Trial.id == trial_id)).first()
    if not trial:
        raise HTTPException(404, "Trial not found")
    
    # Check if it's a group tasking
    if not trial.is_group_tasking or not trial.is_open_for_joining:
        raise HTTPException(400, "Trial is not open for joining")
    
    # Check if user already joined
    existing_participant = s.exec(
        select(TrialParticipant)
        .where(TrialParticipant.trial_id == trial_id)
        .where(TrialParticipant.user_id == user.id)
    ).first()
    
    if existing_participant:
        raise HTTPException(400, "You are already part of this group")
    
    # Check max participants limit
    if trial.max_participants and trial.participant_count >= trial.max_participants:
        raise HTTPException(400, "This group is full")
    
    # Add user as participant
    participant = TrialParticipant(
        trial_id=trial_id,
        user_id=user.id,
        is_creator=False,
        credited_creator=True  # Will credit the creator
    )
    s.add(participant)
    
    # Update participant count
    trial.participant_count += 1
    s.add(trial)
    
    # Credit the creator
    creator = s.exec(select(User).where(User.id == trial.created_by)).first()
    if creator:
        creator.reward_group_join_credit()
        s.add(creator)
    
    s.commit()
    
    return RedirectResponse(url=f"/trials/{trial_id}?joined=success", status_code=303)

@app.post("/create-group-tasking")
def create_group_tasking(request: Request,
                        title: str = Form(),
                        domain: str = Form(),
                        event_spec: str = Form(),
                        result_time: str = Form(),
                        max_participants: Optional[int] = Form(default=None),
                        s: Session = Depends(db),
                        user: User = Depends(require_user)):
    """Admin-only endpoint to create group taskings"""
    
    if not user.is_admin:
        raise HTTPException(403, "Only administrators can create group taskings")
    
    try:
        result_time_utc = datetime.fromisoformat(result_time).replace(tzinfo=timezone.utc)
        
        # Create group trial
        trial = Trial(
            title=title,
            domain=domain,
            event_spec_json=json.dumps({"description": event_spec}),
            result_time_utc=result_time_utc,
            status="draft",
            created_by=user.id,
            is_group_tasking=True,
            is_open_for_joining=True,
            max_participants=max_participants,
            participant_count=1
        )
        s.add(trial)
        s.flush()
        
        # Add creator as first participant
        creator_participant = TrialParticipant(
            trial_id=trial.id,
            user_id=user.id,
            is_creator=True,
            credited_creator=False
        )
        s.add(creator_participant)
        s.commit()
        
        return RedirectResponse(url=f"/trials/{trial.id}?created=success", status_code=303)
        
    except Exception as e:
        s.rollback()
        raise HTTPException(400, f"Failed to create group tasking: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        # Test database connection and initialize default data
        with Session(engine) as session:
            ensure_sample_targets(session)
            ensure_admin_user(session)
            ensure_homepage_content(session)
        print("✓ Database connection established and initialized")
    except Exception as e:
        print(f"⚠ Database initialization warning: {e}")
        # Don't fail startup for non-critical database issues
    
    if AI_ENABLED:
        print("✓ AI suggestions enabled")
    else:
        print("ℹ AI suggestions disabled (no OpenAI API key)")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
