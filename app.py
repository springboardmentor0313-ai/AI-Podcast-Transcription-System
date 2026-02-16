import sys
import os
import json
import tempfile
import base64
import streamlit as st

# --------------------------------------------------
# Ensure project root is in Python path
# --------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------
# Import project modules
# --------------------------------------------------
from stt.whisper_stt import transcribe
from topic_segmentation.segmenter import segment_topics

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Podcast Navigation System",
    layout="wide"
)

st.title("🎧 Podcast Navigation System")
st.markdown("Navigate podcasts using topic-based audio navigation.")

# --------------------------------------------------
# UI & Styling Assets (UNCHANGED)
# --------------------------------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

bg_img_path = os.path.join(os.path.dirname(__file__), "assets/waveform_bg.png")

try:
    bin_str = get_base64_of_bin_file(bg_img_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
            linear-gradient(rgba(10,10,10,0.92), rgba(10,10,10,0.95)),
            url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            font-family: 'Inter', sans-serif;
        }}
        mark {{
            background-color: #facc15;
            color: black;
            padding: 2px 4px;
            border-radius: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except:
    pass

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "seek_time" not in st.session_state:
    st.session_state.seek_time = 0

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def highlight(text, keyword):
    if not keyword:
        return text
    return text.replace(
        keyword,
        f"<mark>{keyword}</mark>"
    )

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload podcast audio",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file:

    # Detect new audio
    if st.session_state.last_uploaded_name != uploaded_file.name:
        if os.path.exists("data/transcript/transcript.json"):
            os.remove("data/transcript/transcript.json")
        if os.path.exists("data/segments/segments.json"):
            os.remove("data/segments/segments.json")

        st.session_state.seek_time = 0
        st.session_state.selected_topic = None
        st.session_state.processed = False
        st.session_state.last_uploaded_name = uploaded_file.name

    # Save audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Heavy processing (once)
    if not st.session_state.processed:
        with st.spinner("Processing podcast audio..."):
            transcribe(audio_path)
            segment_topics()

        st.session_state.processed = True
        st.success("Podcast processed successfully ✅")

    # Load outputs
    with open("data/segments/segments.json") as f:
        segments = json.load(f)

    with open("data/transcript/transcript.json") as f:
        transcript_json = json.load(f)

    # Audio player
    st.audio(audio_path, start_time=int(st.session_state.seek_time))

    # Layout
    col1, col2 = st.columns([1.3, 2])

    # ---------------- LEFT: Chapters ----------------
    with col1:
        st.subheader("🧭 Chapters")

        search_query = st.text_input(
            "🔍 Search keyword",
            placeholder="e.g. Malta, learning, podcast"
        )

        filtered_segments = segments
        if search_query:
            q = search_query.lower()
            filtered_segments = [
                seg for seg in segments
                if q in seg["title"].lower()
                or q in seg.get("summary", "").lower()
                or q in seg["text"].lower()
            ]

        if search_query and not filtered_segments:
            st.warning("No matching segments found.")

        for i, seg in enumerate(filtered_segments, start=1):
            title_html = highlight(seg["title"], search_query)

            st.markdown(
                f"""
                <div style="
                    padding:16px;
                    margin-bottom:10px;
                    border-radius:12px;
                    background:rgba(255,255,255,0.04);
                    border:1px solid rgba(255,255,255,0.08);
                ">
                    <div style="font-size:15px;color:#f1f5f9;">
                        {title_html}
                    </div>
                    <div style="font-size:13px;color:#94a3b8;">
                        ⏱ {format_time(seg['start'])} – {format_time(seg['end'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(f"▶ Play", key=f"play_{i}"):
                st.session_state.seek_time = int(seg["start"])
                st.session_state.selected_topic = seg
                st.rerun()

    # ---------------- RIGHT: Topic Details ----------------
    with col2:
        if st.session_state.selected_topic:
            seg = st.session_state.selected_topic

            st.subheader("📌 Topic Details")
            st.markdown(f"**Title:** {seg['title']}")
            st.markdown(
                f"**Time:** {format_time(seg['start'])} – {format_time(seg['end'])}"
            )

            # Summary (highlighted)
            st.markdown("### ✨ Summary")
            summary_html = highlight(seg.get("summary", ""), search_query)

            st.markdown(
                f"""
                <div style="
                    padding:14px;
                    border-radius:10px;
                    background:rgba(59,130,246,0.12);
                    border:1px solid rgba(59,130,246,0.35);
                    color:#e5e7eb;
                    line-height:1.6;
                    margin-bottom:18px;
                ">
                    {summary_html}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Transcript (highlighted)
            st.markdown("### 📄 Transcript")
            st.text_area(
                "",
                highlight(seg["text"], search_query),
                height=320
            )
        else:
            # Display the high-impact title
            display_title = "English Learning For Curious Minds: The Evolution"

            st.markdown(
                f"""
                <div style="
                    text-align: center;
                    padding: 40px;
                    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
                    border-radius: 16px;
                    border: 1px solid rgba(255,255,255,0.1);
                    margin-top: 20px;
                ">
                    <h1 style="
                        font-size: 2.5rem;
                        font-weight: 700;
                        color: #f8fafc;
                        background: linear-gradient(90deg, #60a5fa, #c084fc);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        margin-bottom: 16px;
                    ">
                        {display_title}
                    </h1>
                    <p style="
                        font-size: 1.1rem;
                        color: #94a3b8;
                        line-height: 1.6;
                    ">
                        Select a topic from the sidebar to explore the transcript segments.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Downloads
    st.divider()

    st.download_button(
        "⬇️ Download Full Transcript (JSON)",
        json.dumps(transcript_json, indent=4),
        "transcript.json",
        "application/json"
    )

    st.download_button(
        "⬇️ Download Topic Segments (JSON)",
        json.dumps(segments, indent=4),
        "segments.json",
        "application/json"
    )

    # Cleanup
    try:
        os.remove(audio_path)
    except:
        pass

else:
    st.info("Upload a podcast audio file to begin.")
