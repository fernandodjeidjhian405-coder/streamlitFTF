import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

WEBRTC_AVAILABLE = False
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

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


st.set_page_config(page_title="Mood Tracker", page_icon="😊", layout="wide")

init_db()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "nickname" not in st.session_state:
    st.session_state.nickname = None


if st.session_state.user_id is None:
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1rem;">
            <h1 style="margin-bottom: 0.5rem;">😊 Mood Tracker</h1>
            <p style="color: #94a3b8; font-size: 1.1rem;">Enter your nickname to track your mood with facial recognition.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("nickname_form"):
            nickname = st.text_input("Your nickname", placeholder="e.g. Alex, Sam...", key="nickname_input")
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

    if WEBRTC_AVAILABLE:
        st.subheader("Camera Constraints (WebRTC)")
        facing = st.selectbox("Facing mode", ["user (front)", "environment (back)"], index=0)
        width = st.selectbox("Width", [640, 1280], index=0)
        height = st.selectbox("Height", [480, 720], index=0)
        fps = st.selectbox("FPS", [15, 30], index=0)

# -------------------------------------------------------------------
# MediaPipe FaceMesh setup
# -------------------------------------------------------------------
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
tab_mood, tab_profile, tab_activities = st.tabs(["Mood Tracker", "Profile", "Mood Activities"])

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
    st.subheader("Track your mood with facial recognition")
    st.write("Allow camera access or take a snapshot. Your detected mood will be shown.")

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
    # Face scanning at the top
    # -------------------------------------------------------------------
    @dataclass
    class _StreamState:
        label: str = "No face"

    if WEBRTC_AVAILABLE:
        _facing = "user" if facing.startswith("user") else "environment"

        class VideoProcessor:
            def __init__(self):
                self.state = _StreamState()

            def recv(self, frame):
                img_bgr = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                result = mesh.process(img_rgb)

                label = "No face"
                if result.multi_face_landmarks:
                    face = result.multi_face_landmarks[0].landmark
                    label = classify_expression(face)

                    if draw_mesh:
                        mp_drawing.draw_landmarks(
                            image=img_bgr,
                            landmark_list=result.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=img_bgr,
                            landmark_list=result.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                        )

                text = label
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(img_bgr, (10, 10), (10 + tw + 16, 10 + th + 16), (0, 0, 0), -1)
                cv2.putText(img_bgr, text, (18, 10 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                self.state.label = label
                return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        ice_servers: List[Dict[str, Any]] = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478?transport=udp"]},
        ]
        try:
            turn = st.secrets.get("turn", {})
            turn_urls = turn.get("urls")
            turn_username = turn.get("username")
            turn_credential = turn.get("credential")
            if turn_urls:
                turn_entry = {"urls": list(turn_urls) if isinstance(turn_urls, (list, tuple)) else [str(turn_urls)]}
                if turn_username and turn_credential:
                    turn_entry["username"] = str(turn_username)
                    turn_entry["credential"] = str(turn_credential)
                ice_servers.append(turn_entry)
        except Exception:
            pass

        rtc_config = RTCConfiguration({"iceServers": ice_servers})
        media_stream_constraints = {
            "video": {
                "facingMode": _facing,
                "width": {"ideal": int(width)},
                "height": {"ideal": int(height)},
                "frameRate": {"ideal": int(fps)},
            },
            "audio": False,
        }

        ctx = webrtc_streamer(
            key="expr-detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints=media_stream_constraints,
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

        if ctx and ctx.video_processor and ctx.video_processor.state.label not in ("No face", ""):
            st.session_state.last_detected_mood = ctx.video_processor.state.label

        with st.expander("Connection / Permission Status", expanded=False):
            if ctx is None:
                st.write("WebRTC context is not available.")
            else:
                st.write(f"WebRTC state: **{getattr(ctx, 'state', None)}**")
                st.write(f"Playing: **{getattr(ctx, 'playing', None)}**")
                if ctx.video_processor:
                    st.write(f"Last label: **{ctx.video_processor.state.label}**")
                st.info(
                    "If you don't get a camera prompt: "
                    "1) Click the lock icon (🔒) → Allow Camera, then reload. "
                    "2) Close other apps using the camera (Zoom/Teams/Meet). "
                    "3) Try a different network or mobile hotspot (WebRTC may be blocked)."
                )

    st.markdown("---")
    st.subheader("Snapshot mode (no WebRTC)")
    img_file = st.camera_input("Take a photo")
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
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption=f"Detected: {label}")
                st.caption("Scroll down to rate intensity and log your mood.")
            else:
                st.warning("No face detected in the snapshot.")

    if not WEBRTC_AVAILABLE:
        st.warning("Install streamlit-webrtc and av for live video. Snapshot mode is available.")

    # -------------------------------------------------------------------
    # Mood result, intensity, and activities (below face scanning)
    # -------------------------------------------------------------------
    st.markdown("---")
    current_mood = st.session_state.get("last_detected_mood", None)

    if current_mood:
        mood_emoji = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Neutral": "😐"}
        em = mood_emoji.get(current_mood, "😐")
        st.markdown(f"## {em} {current_mood}")
        st.caption("How strongly do you feel this mood? Rate 1–5, then log.")
        intensity = st.select_slider(
            "Mood intensity",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: INTENSITY_LABELS[x],
            key="mood_intensity",
            label_visibility="collapsed",
        )
        if st.button("Log mood"):
            log_mood(st.session_state.user_id, current_mood, intensity)
            st.toast("Mood logged!", icon="✅")
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
# Mood Activities tab (browse all moods)
# -------------------------------------------------------------------
with tab_activities:
    st.subheader("Mood-based activities")
    st.write("Browse activities by mood, or use the Mood Tracker to see suggestions based on your face detection.")

    selected_mood = st.selectbox(
        "How are you feeling?",
        ["Happy", "Sad", "Angry", "Neutral"],
        key="activity_mood",
    )

    activities = MOOD_ACTIVITIES.get(selected_mood, MOOD_ACTIVITIES["Neutral"])
    mood_emoji = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Neutral": "😐"}
    em = mood_emoji.get(selected_mood, "😐")

    st.markdown(f"### {em} Suggestions for when you feel **{selected_mood}**")

    st.markdown(
        """
        <style>
        .activity-card {
            padding: 1.25rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(124, 58, 237, 0.3);
            background: rgba(30, 30, 46, 0.6);
            margin-bottom: 1rem;
        }
        .activity-card h4 { margin: 0 0 0.5rem 0; }
        .activity-card p { margin: 0; color: #94a3b8; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for i in range(0, len(activities), 2):
        row_activities = activities[i : i + 2]
        cols = st.columns(2)
        for j, (title, desc) in enumerate(row_activities):
            icon = ACTIVITY_ICONS.get(title, "•")
            with cols[j]:
                with st.container():
                    st.markdown(
                        f'<div class="activity-card"><h4>{icon} {title}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True,
                    )
