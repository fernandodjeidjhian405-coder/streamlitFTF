import calendar
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "mood_tracker.db")


def init_db():
    """Create data directory and database tables if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if cur.fetchone():
        cur.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cur.fetchall()]
        if "password_hash" in columns:
            cur.execute("DROP TABLE IF EXISTS mood_logs")
            cur.execute("DROP TABLE users")
            conn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nickname TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mood_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mood TEXT NOT NULL,
            intensity INTEGER DEFAULT 3 CHECK (intensity >= 1 AND intensity <= 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # Migration: add intensity column if missing (older DBs)
    cur.execute("PRAGMA table_info(mood_logs)")
    cols = [row[1] for row in cur.fetchall()]
    if "intensity" not in cols:
        cur.execute("ALTER TABLE mood_logs ADD COLUMN intensity INTEGER DEFAULT 3")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mood_logs_user ON mood_logs(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mood_logs_created ON mood_logs(created_at)")
    conn.commit()
    conn.close()


def get_conn():
    return sqlite3.connect(DB_PATH)


def get_or_create_user_by_nickname(nickname: str) -> Optional[int]:
    """Find existing user by nickname or create new one. Returns user_id."""
    nick = nickname.strip()
    if not nick:
        return None
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE nickname = ?", (nick,))
    row = cur.fetchone()
    if row:
        user_id = row[0]
        conn.close()
        return user_id
    cur.execute("INSERT INTO users (nickname) VALUES (?)", (nick,))
    user_id = cur.lastrowid
    conn.commit()
    conn.close()
    return user_id


def log_mood(user_id: int, mood: str, intensity: int = 3) -> bool:
    """Log a mood entry for the user with intensity 1-5."""
    intensity = max(1, min(5, intensity))
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mood_logs (user_id, mood, intensity) VALUES (?, ?, ?)",
        (user_id, mood, intensity),
    )
    conn.commit()
    conn.close()
    return True


def get_daily_moods(user_id: int) -> List[tuple]:
    """Get mood counts by hour for today."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT mood, strftime('%H', created_at) AS hour, COUNT(*) as cnt
        FROM mood_logs
        WHERE user_id = ? AND date(created_at) = date('now')
        GROUP BY hour, mood
        ORDER BY hour
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_total_logs(user_id: int) -> int:
    """Get total mood log count for user (all time)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM mood_logs WHERE user_id = ?", (user_id,))
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_logs_today(user_id: int) -> int:
    """Get mood log count for today."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM mood_logs WHERE user_id = ? AND date(created_at) = date('now')",
        (user_id,),
    )
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_dominant_mood(user_id: int, days: int = 7) -> Optional[str]:
    """Get most frequent mood in last N days. Returns None if no data."""
    dist = get_mood_distribution(user_id, days)
    if not dist:
        return None
    return max(dist, key=lambda x: x[1])[0]


def get_weekly_moods(user_id: int, days: int = 7) -> List[tuple]:
    """Get mood counts by day for last N days."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT mood, date(created_at) AS day, COUNT(*) as cnt
        FROM mood_logs
        WHERE user_id = ? AND created_at >= date('now', ? || ' days')
        GROUP BY day, mood
        ORDER BY day
    """, (user_id, f"-{days}"))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_mood_distribution(user_id: int, days: int = 7) -> List[tuple]:
    """Get total mood counts over last N days for pie chart."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT mood, COUNT(*) as cnt
        FROM mood_logs
        WHERE user_id = ? AND created_at >= date('now', ? || ' days')
        GROUP BY mood
    """, (user_id, f"-{days}"))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_moods_by_date(user_id: int, date_str: str) -> List[tuple]:
    """Get all mood entries for a specific date."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT mood, intensity, strftime('%H:%M', created_at) as time
        FROM mood_logs
        WHERE user_id = ? AND date(created_at) = ?
        ORDER BY created_at DESC
    """, (user_id, date_str))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_dates_with_moods(user_id: int, year: int, month: int) -> List[str]:
    """Get list of dates that have mood entries for a specific month."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT date(created_at) as date
        FROM mood_logs
        WHERE user_id = ? 
          AND strftime('%Y', created_at) = ?
          AND strftime('%m', created_at) = ?
    """, (user_id, str(year), f"{month:02d}"))
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]


st.set_page_config(
    page_title="FACES TO FEELINGS", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

init_db()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "nickname" not in st.session_state:
    st.session_state.nickname = None
if "selected_calendar_date" not in st.session_state:
    st.session_state.selected_calendar_date = None
if "calendar_year" not in st.session_state:
    st.session_state.calendar_year = datetime.now().year
if "calendar_month" not in st.session_state:
    st.session_state.calendar_month = datetime.now().month


if st.session_state.user_id is None:
    # Floating motivational quotes with colorful text on white background
    quotes_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: transparent;
                overflow: hidden;
                pointer-events: none;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .quote {
                position: absolute;
                bottom: -50px;
                font-size: 1.1rem;
                font-style: italic;
                font-weight: 500;
                white-space: nowrap;
                animation: riseUp 14s linear forwards;
                text-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }
            
            @keyframes riseUp {
                0% {
                    bottom: -50px;
                    opacity: 0;
                }
                10% {
                    opacity: 0.7;
                }
                60% {
                    opacity: 0.7;
                }
                100% {
                    bottom: 65%;
                    opacity: 0;
                }
            }
        </style>
    </head>
    <body>
        <div id="quotes-container"></div>
        
        <script>
            const quotes = [
                "Believe in yourself",
                "Every day is a fresh start",
                "Your feelings are valid",
                "Progress, not perfection",
                "You are stronger than you think",
                "One step at a time",
                "Be kind to yourself",
                "Today is full of possibilities",
                "You matter",
                "Breathe and let go",
                "Small steps lead to big changes",
                "You are enough",
                "This too shall pass",
                "Choose joy today",
                "Your story isn't over yet",
                "Embrace the journey",
                "You deserve happiness",
                "Keep going, you're doing great"
            ];
            
            // Colorful palette
            const colors = [
                '#7c3aed', // purple
                '#3b82f6', // blue
                '#10b981', // emerald
                '#f59e0b', // amber
                '#ec4899', // pink
                '#6366f1', // indigo
                '#14b8a6', // teal
                '#f97316', // orange
                '#8b5cf6', // violet
                '#06b6d4'  // cyan
            ];
            
            const container = document.getElementById('quotes-container');
            let quoteIndex = 0;
            let colorIndex = 0;
            
            function createQuote() {
                const quote = document.createElement('div');
                quote.className = 'quote';
                quote.textContent = '"' + quotes[quoteIndex] + '"';
                quote.style.left = (Math.random() * 65 + 5) + '%';
                quote.style.fontSize = (1.0 + Math.random() * 0.3) + 'rem';
                quote.style.color = colors[colorIndex];
                
                container.appendChild(quote);
                
                // Remove quote after animation
                setTimeout(() => {
                    quote.remove();
                }, 14000);
                
                quoteIndex = (quoteIndex + 1) % quotes.length;
                colorIndex = (colorIndex + 1) % colors.length;
            }
            
            // Create initial quotes with stagger
            for (let i = 0; i < 6; i++) {
                setTimeout(() => createQuote(), i * 1500);
            }
            
            // Continue creating quotes at slow interval
            setInterval(createQuote, 3000);
        </script>
    </body>
    </html>
    """
    
    # Render the floating quotes as a fixed iframe
    components.html(quotes_html, height=0, scrolling=False)
    
    # White background styling for nickname page
    st.markdown(
        """
        <style>
        /* White background */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff;
            overflow: hidden;
        }
        .stApp {
            background-color: #ffffff;
        }
        
        /* Position iframe as background */
        iframe {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            z-index: 0 !important;
            pointer-events: none !important;
        }
        
        /* Keep content above quotes */
        .stApp > header, .main, .block-container {
            position: relative;
            z-index: 1;
        }
        
        /* Fade-in animation */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .main .block-container { animation: fadeInUp 0.6s ease-out; }
        
        /* Input styling for white background */
        .stTextInput input {
            background-color: #f8fafc;
            border: 2px solid #e2e8f0;
            color: #1e293b;
        }
        .stTextInput input:focus {
            box-shadow: 0 0 15px rgba(124, 58, 237, 0.3);
            border-color: #7c3aed;
        }
        .stTextInput label {
            color: #475569 !important;
        }
        
        /* Button styling - purple gradient */
        .stButton > button {
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%);
            color: #ffffff;
            border: none;
        }
        .stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
        }
        
        /* Form styling */
        [data-testid="stForm"] {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1rem; position: relative; z-index: 1;">
            <h1 style="margin-bottom: 0.5rem; color: #1e293b; font-weight: 700;">FACES TO FEELINGS</h1>
            <p style="color: #64748b; font-size: 1.1rem;">Let your emotions fill the pages today, and discover what's waiting for you..</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("nickname_form"):
            nickname = st.text_input("Hi there! Ready to check in?", placeholder="What should we call you?", key="nickname_input")
            submitted = st.form_submit_button("Continue")
            if submitted and nickname.strip():
                uid = get_or_create_user_by_nickname(nickname)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.nickname = nickname.strip()
                    st.toast(f"Hello, {nickname.strip()}!", icon="👋")
                    st.rerun()
                else:
                    st.error("Please enter a nickname.")
            elif submitted and not nickname.strip():
                st.error("Please enter a nickname.")

    st.stop()

# -------------------------------------------------------------------
# Main app (authenticated)
# -------------------------------------------------------------------

# Global interactive CSS for main app with light gradient background
st.markdown(
    """
    <style>
    /* Light gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        background-attachment: fixed;
    }
    
    /* Dark text colors for light background */
    .stApp h1, .stApp h2, .stApp h3 {
        color: #1e293b;
    }
    .stApp p, .stApp span, .stApp label {
        color: #475569;
    }
    
    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Apply fade-in to main content */
    .main .block-container { animation: fadeIn 0.5s ease-out; }
    
    /* Button styling for light theme */
    .stButton > button {
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%);
        color: white;
        border: none;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.5);
    }
    .stButton > button:active {
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.7);
    }
    
    /* Card hover effect */
    .activity-card {
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(124, 58, 237, 0.2);
    }
    .activity-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.2);
        border-color: rgba(124, 58, 237, 0.5);
    }
    
    /* Input styling for light theme */
    .stTextInput input {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        color: #1e293b;
    }
    .stTextInput input:focus {
        box-shadow: 0 0 10px rgba(124, 58, 237, 0.3);
        border-color: #7C3AED;
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] div {
        transition: all 0.2s ease;
    }
    
    /* Tab styling for light theme */
    .stTabs [data-baseweb="tab"] {
        color: #475569;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #7C3AED;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #7C3AED;
    }
    
    /* Sidebar styling - keep dark for contrast */
    .stSidebar {
        background-color: rgba(15, 23, 42, 0.95);
    }
    .stSidebar .stButton > button:hover {
        background: linear-gradient(135deg, #667eea 0%, #7C3AED 100%);
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar span, .stSidebar label {
        color: #ffffff !important;
    }
    
    /* Metric cards for light theme */
    [data-testid="stMetric"] {
        transition: all 0.3s ease;
        border-radius: 12px;
        padding: 12px;
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(124, 58, 237, 0.15);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.15);
    }
    [data-testid="stMetric"] label {
        color: #64748b !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1e293b !important;
    }
    
    /* Form and container styling */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(124, 58, 237, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

hour = datetime.now().hour
if 5 <= hour < 12:
    time_greeting = "Good morning"
elif 12 <= hour < 17:
    time_greeting = "Good afternoon"
else:
    time_greeting = "Good evening"
nickname = st.session_state.get("nickname", "there")
st.title(f"Hello, {time_greeting}, {nickname}!")

# Sidebar
with st.sidebar:
    st.markdown("## FacesToFeelings")
    with st.expander("About"):
        st.caption(
            "Mood Tracker uses facial recognition to detect Happy, Sad, Angry, or Neutral. "
            "On Streamlit Cloud, data may not persist across app restarts."
        )
    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.header("Settings")
    refine = st.toggle("Refine landmarks (eyes/irises)", value=True)
    det_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
    track_conf = st.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.05)
    draw_mesh = st.toggle("Draw face mesh overlay", value=False)


# FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LM = {
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_inner": 13,
    "lower_inner": 14,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "nose_tip": 1,
    "left_brow_inner": 276,
    "left_brow_outer": 282,
    "right_brow_inner": 336,
    "right_brow_outer": 334,
    "left_eye_upper": 159,
    "right_eye_upper": 386,
}


def _dist_norm(a, b) -> float:
    dx, dy = (a.x - b.x), (a.y - b.y)
    return math.hypot(dx, dy)


def classify_expression(landmarks) -> str:
    """
    Returns: Happy, Sad, Angry, or Neutral
    """
    L = landmarks[LM["mouth_left"]]
    R = landmarks[LM["mouth_right"]]
    U = landmarks[LM["upper_inner"]]
    D = landmarks[LM["lower_inner"]]
    LE_in = landmarks[LM["left_eye_inner"]]
    LE_out = landmarks[LM["left_eye_outer"]]
    RE_in = landmarks[LM["right_eye_inner"]]
    RE_out = landmarks[LM["right_eye_outer"]]

    mouth_width = _dist_norm(L, R) + 1e-6
    mouth_height = _dist_norm(U, D)
    mar = mouth_height / mouth_width

    mouth_center_y = (U.y + D.y) / 2.0
    corner_height = ((mouth_center_y - L.y) + (mouth_center_y - R.y)) / 2.0
    corner_norm = corner_height / mouth_width

    left_eye = _dist_norm(LE_in, LE_out)
    right_eye = _dist_norm(RE_in, RE_out)
    eye_avg = (left_eye + right_eye) / 2.0
    eye_squint = max(0.0, 0.14 - eye_avg)

    smile_score = (0.65 * corner_norm) + (0.25 * eye_squint) + (0.10 * mar)
    sad_score = (-0.70 * corner_norm) + (0.10 * (0.22 - mar))

    # Angry: furrowed brows (lowered eyebrows relative to eyes)
    lb_in = landmarks[LM["left_brow_inner"]]
    lb_out = landmarks[LM["left_brow_outer"]]
    rb_in = landmarks[LM["right_brow_inner"]]
    rb_out = landmarks[LM["right_brow_outer"]]
    le_up = landmarks[LM["left_eye_upper"]]
    re_up = landmarks[LM["right_eye_upper"]]

    left_brow_y = (lb_in.y + lb_out.y) / 2
    right_brow_y = (rb_in.y + rb_out.y) / 2
    left_eye_y = le_up.y
    right_eye_y = re_up.y

    brow_lowered_left = left_brow_y - left_eye_y
    brow_lowered_right = right_brow_y - right_eye_y
    angry_score = (brow_lowered_left + brow_lowered_right) / 2

    if smile_score > 0.045:
        return "Happy"
    elif sad_score > 0.12:
        return "Sad"
    elif angry_score > 0.015:
        return "Angry"
    else:
        return "Neutral"


@st.cache_resource(show_spinner=False)
def get_mesh(refine: bool, det_conf: float, track_conf: float):
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=refine,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )


mesh = get_mesh(refine, det_conf, track_conf)

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_mood, tab_profile, tab_calendar = st.tabs(["Mood Tracker", "Profile", "Mood Calendar"])

MOOD_ACTIVITIES = {
    "Happy": [
        ("Share it", "Tell someone about your good mood"),
        ("Journal", "Write down what made you happy"),
        ("Take a photo", "Capture the moment"),
        ("Go for a walk", "Enjoy the outdoors"),
    ],
    "Sad": [
        ("Listen to uplifting music", "Let music lift your spirits"),
        ("Talk to a friend", "Reach out for support"),
        ("Light exercise", "A short walk or stretch"),
        ("Self-care", "Take a relaxing bath or rest"),
    ],
    "Angry": [
        ("Deep breathing", "4-7-8 breath: inhale 4, hold 7, exhale 8"),
        ("Short walk", "Step away and cool down"),
        ("Journal", "Write down what you feel"),
        ("Cool-down exercise", "Stretching or gentle movement"),
    ],
    "Neutral": [
        ("Try something new", "Learn a new skill or hobby"),
        ("Short puzzle", "Crossword, Sudoku, or a game"),
        ("Listen to a podcast", "Explore an interesting topic"),
        ("Stretch", "A few minutes of gentle stretching"),
    ],
}
INTENSITY_LABELS = {1: "1 – Very slight", 2: "2 – Slight", 3: "3 – Moderate", 4: "4 – Strong", 5: "5 – Very strong"}

ACTIVITY_ICONS = {
    "Share it": "💬", "Journal": "📝", "Take a photo": "📷", "Go for a walk": "🚶",
    "Listen to uplifting music": "🎵", "Talk to a friend": "👋", "Light exercise": "🏃", "Self-care": "🧖",
    "Deep breathing": "🌬️", "Short walk": "🚶", "Cool-down exercise": "🧘",
    "Try something new": "✨", "Short puzzle": "🧩", "Listen to a podcast": "🎧", "Stretch": "🙆",
}

# -------------------------------------------------------------------
# Mood Tracker tab
# -------------------------------------------------------------------
with tab_mood:
    st.subheader("Let’s see what’s written in your face.")

    with st.expander("How it works"):
        st.markdown(
            """
            - **Live mode (WebRTC):** Start the camera to see real-time mood detection from your facial expressions.
            - **Snapshot mode:** Take a photo with your camera for one-time mood detection.
            - The app analyzes your face (smile, mouth shape, eyebrows) to detect Happy, Sad, Angry, or Neutral.
            - Rate your mood intensity (1–5), then click **Log mood** to save to your profile.
            """
        )

    # -------------------------------------------------------------------
    # Face scanning - Take a Photo
    # -------------------------------------------------------------------
    st.subheader("Capture this moment")
    st.caption("Position yourself in the frame and take your photo when you're ready. Let’s see what your expression has to tell us about you today")
    
    img_file = st.camera_input("Capture your face", label_visibility="collapsed")
    
    if img_file is not None:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode image from camera.")
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0].landmark
                label = classify_expression(face)
                st.session_state.last_detected_mood = label

                if draw_mesh:
                    mp_drawing.draw_landmarks(
                        image=bgr,
                        landmark_list=res.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=bgr,
                        landmark_list=res.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                    )
                st.success(f"Detected mood: {label}")
                st.caption("Scroll down to share how deeply you’re feeling today. Whenever you're ready, let’s log your mood together.")
            else:
                st.warning("Oops! We missed you that time. Let’s try again—make sure you’re front and center!")

    # -------------------------------------------------------------------
    # Mood result, intensity, and activities (below face scanning)
    # -------------------------------------------------------------------
    st.markdown("---")
    current_mood = st.session_state.get("last_detected_mood", None)

    if current_mood:
        mood_emoji = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Neutral": "😐"}
        em = mood_emoji.get(current_mood, "😐")
        st.markdown(f"## {em} {current_mood}")
        st.caption("How much space is this mood taking up today? Choose from 1–5")
        intensity = st.select_slider(
            "Mood intensity",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: INTENSITY_LABELS[x],
            key="mood_intensity",
            label_visibility="collapsed",
        )
        if st.button("Save mood"):
            log_mood(st.session_state.user_id, current_mood, intensity)
            st.toast("Received with care", icon="✅")
            st.rerun()

        st.markdown("---")
        st.markdown("#### Suggested activities for your mood")
        activities = MOOD_ACTIVITIES.get(current_mood, MOOD_ACTIVITIES["Neutral"])
        st.markdown(
            """
            <style>
            .activity-card { padding: 1rem; border-radius: 0.5rem; border: 1px solid rgba(124, 58, 237, 0.3);
                background: rgba(30, 30, 46, 0.6); margin-bottom: 0.75rem; }
            .activity-card h4 { margin: 0 0 0.25rem 0; }
            .activity-card p { margin: 0; color: #94a3b8; font-size: 0.9rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        for i in range(0, len(activities), 2):
            row = activities[i : i + 2]
            cols = st.columns(2)
            for j, (title, desc) in enumerate(row):
                icon = ACTIVITY_ICONS.get(title, "•")
                with cols[j]:
                    st.markdown(
                        f'<div class="activity-card"><h4>{icon} {title}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True,
                    )

# -------------------------------------------------------------------
# Profile tab
# -------------------------------------------------------------------
PLOTLY_DARK_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "responsive": True,
}

def _apply_plotly_dark(fig):
    """Apply dark template and layout for consistent styling."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

with tab_profile:
    st.subheader("Your mood statistics")
    user_id = st.session_state.user_id

    if st.button("Refresh charts", key="refresh_profile"):
        st.toast("Charts refreshed!", icon="🔄")
        st.rerun()

    days_range = st.radio(
        "Date range",
        [7, 14, 30],
        format_func=lambda x: f"Last {x} days",
        horizontal=True,
        key="profile_days",
    )

    total_logs = get_total_logs(user_id)
    logs_today = get_logs_today(user_id)
    dominant = get_dominant_mood(user_id, days_range) or "—"

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total mood logs", total_logs)
    with m2:
        st.metric("Logs today", logs_today)
    with m3:
        st.metric("Dominant mood", dominant)

    st.divider()

    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly for charts: `pip install plotly`")
    else:
        daily = get_daily_moods(user_id)
        weekly = get_weekly_moods(user_id, days_range)
        dist = get_mood_distribution(user_id, days_range)

        if not daily and not weekly and not dist:
            st.info("No mood data yet. Start tracking your mood in the Mood Tracker tab!")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Daily trends (today)")
                if daily:
                    df_daily = [{"Hour": int(h), "Mood": m, "Count": c} for m, h, c in daily]
                    fig_daily = px.bar(
                        df_daily,
                        x="Hour",
                        y="Count",
                        color="Mood",
                        title="Mood by hour today",
                        barmode="stack",
                    )
                    _apply_plotly_dark(fig_daily)
                    st.plotly_chart(fig_daily, use_container_width=True, config=PLOTLY_DARK_CONFIG)
                else:
                    st.caption("No data for today yet.")

            with col2:
                st.markdown(f"#### Weekly trends (last {days_range} days)")
                if weekly:
                    df_weekly = [{"Day": d, "Mood": m, "Count": c} for m, d, c in weekly]
                    fig_weekly = px.bar(
                        df_weekly,
                        x="Day",
                        y="Count",
                        color="Mood",
                        title=f"Mood by day (last {days_range} days)",
                        barmode="stack",
                    )
                    _apply_plotly_dark(fig_weekly)
                    st.plotly_chart(fig_weekly, use_container_width=True, config=PLOTLY_DARK_CONFIG)
                else:
                    st.caption(f"No data for the last {days_range} days yet.")

            st.markdown(f"#### Mood distribution (last {days_range} days)")
            if dist:
                df_dist = [{"Mood": m, "Count": c} for m, c in dist]
                fig_pie = px.pie(
                    df_dist,
                    values="Count",
                    names="Mood",
                    title="Overall mood distribution",
                )
                _apply_plotly_dark(fig_pie)
                st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_DARK_CONFIG)
            else:
                st.caption(f"No mood distribution data yet.")

# -------------------------------------------------------------------
# Mood Calendar tab (view mood entries by date)
# -------------------------------------------------------------------
with tab_calendar:
    st.subheader("Mood Calendar")
    st.write("Click on a date to view your mood entries for that day.")
    
    # Calendar styling
    st.markdown(
        """
        <style>
        .calendar-header {
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        .calendar-day-header {
            text-align: center;
            font-weight: 600;
            color: #94a3b8;
            padding: 0.5rem 0;
        }
        .mood-entry-card {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(124, 58, 237, 0.3);
            background: rgba(30, 30, 46, 0.6);
            margin-bottom: 0.75rem;
        }
        .mood-entry-card .time {
            color: #94a3b8;
            font-size: 0.85rem;
        }
        .mood-entry-card .mood {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0.25rem 0;
        }
        .intensity-bar {
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #7c3aed, #a855f7);
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Navigation for month/year
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if st.button("◀ Previous", key="prev_month"):
            if st.session_state.calendar_month == 1:
                st.session_state.calendar_month = 12
                st.session_state.calendar_year -= 1
            else:
                st.session_state.calendar_month -= 1
            st.session_state.selected_calendar_date = None
            st.rerun()
    
    with nav_col2:
        month_name = calendar.month_name[st.session_state.calendar_month]
        st.markdown(
            f'<div class="calendar-header">{month_name} {st.session_state.calendar_year}</div>',
            unsafe_allow_html=True
        )
    
    with nav_col3:
        if st.button("Next ▶", key="next_month"):
            if st.session_state.calendar_month == 12:
                st.session_state.calendar_month = 1
                st.session_state.calendar_year += 1
            else:
                st.session_state.calendar_month += 1
            st.session_state.selected_calendar_date = None
            st.rerun()
    
    # Get dates with mood entries for this month
    dates_with_moods = get_dates_with_moods(
        user_id, 
        st.session_state.calendar_year, 
        st.session_state.calendar_month
    )
    today = datetime.now().date()
    
    # Calendar grid
    cal = calendar.Calendar(firstweekday=6)  # Start with Sunday
    month_days = cal.monthdayscalendar(
        st.session_state.calendar_year, 
        st.session_state.calendar_month
    )
    
    # Day headers
    day_headers = st.columns(7)
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    for i, day_name in enumerate(day_names):
        with day_headers[i]:
            st.markdown(f'<div class="calendar-day-header">{day_name}</div>', unsafe_allow_html=True)
    
    # Calendar days
    for week in month_days:
        week_cols = st.columns(7)
        for i, day in enumerate(week):
            with week_cols[i]:
                if day == 0:
                    st.write("")  # Empty cell
                else:
                    date_str = f"{st.session_state.calendar_year}-{st.session_state.calendar_month:02d}-{day:02d}"
                    current_date = datetime(
                        st.session_state.calendar_year, 
                        st.session_state.calendar_month, 
                        day
                    ).date()
                    
                    # Determine button style
                    has_entries = date_str in dates_with_moods
                    is_today = current_date == today
                    is_selected = st.session_state.selected_calendar_date == date_str
                    
                    # Button label with indicators
                    label = str(day)
                    if has_entries:
                        label = f"🔵 {day}"
                    if is_today:
                        label = f"📍 {day}" if not has_entries else f"📍🔵 {day}"
                    
                    button_type = "primary" if is_selected else "secondary"
                    
                    if st.button(label, key=f"cal_{date_str}", type=button_type, use_container_width=True):
                        st.session_state.selected_calendar_date = date_str
                        st.rerun()
    
    st.divider()
    
    # Display mood entries for selected date
    if st.session_state.selected_calendar_date:
        selected_date = datetime.strptime(st.session_state.selected_calendar_date, "%Y-%m-%d")
        formatted_date = selected_date.strftime("%B %d, %Y")
        st.markdown(f"### Mood entries for {formatted_date}")
        
        entries = get_moods_by_date(user_id, st.session_state.selected_calendar_date)
        
        if entries:
            mood_emoji = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Neutral": "😐"}
            intensity_labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
            
            for mood, intensity, time in entries:
                emoji = mood_emoji.get(mood, "😐")
                intensity_pct = (intensity / 5) * 100
                intensity_label = intensity_labels.get(intensity, "Moderate")
                
                st.markdown(
                    f"""
                    <div class="mood-entry-card">
                        <div class="time">🕐 {time}</div>
                        <div class="mood">{emoji} {mood}</div>
                        <div>Intensity: {intensity_label} ({intensity}/5)</div>
                        <div class="intensity-bar" style="width: {intensity_pct}%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No mood entries for this date.")
    else:
        st.info("📅 Select a date above to view your mood entries. Days with 🔵 have recorded entries.")
