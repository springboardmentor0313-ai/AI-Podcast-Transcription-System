import streamlit as st
import tempfile
import os
import re
from faster_whisper import WhisperModel
import google.generativeai as genai

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Podcast Navigator", layout="wide", page_icon="🎙️")

# ================== MIDNIGHT BLACK THEME (HIGH VISIBILITY) ==================
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    /* Sidebar Visibility */
    [data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #262626; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }

    /* File Uploader visibility */
    .stFileUploader section div { color: #FFFFFF !important; }
    [data-testid="stFileUploadDropzone"] div div span { color: #FFFFFF !important; }
    [data-testid="stWidgetLabel"] p { color: #FFFFFF !important; font-size: 1.1rem !important; }
    
    /* Segment Cards */
    .segment-card {
        padding: 20px; border-radius: 12px; background-color: #111111;
        border: 1px solid #262626; border-left: 5px solid #ff4b4b;
        margin-bottom: 15px; transition: all 0.2s ease;
    }
    .segment-card:hover { border-color: #ff4b4b; transform: translateX(5px); }
    
    /* Keyword Tags */
    .keyword-tag {
        display: inline-block; background: #1a1a1a; border-radius: 6px;
        padding: 2px 10px; margin: 3px; font-size: 0.75rem;
        color: #ff4b4b; border: 1px solid #333333;
    }

    /* Primary Buttons (Analyze/Jump) */
    .stButton>button {
        width: 100%; border-radius: 8px; background-color: #ff4b4b;
        color: white; border: none; font-weight: bold; height: 3em;
    }
    .stButton>button:hover { background-color: #e63939; color: white; }

    /* Input Fields & Text Area */
    .stTextInput label p, .stTextArea label p {
        color: #FFFFFF !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #111111 !important; 
        color: #FFFFFF !important; 
        border: 1px solid #444444 !important;
    }

    /* General Markdown / Text visibility */
    .stMarkdown p, .stHeader h3 { color: #FFFFFF !important; }
    mark { background-color: #ff4b4b; color: white; padding: 0 4px; border-radius: 3px; }

    /* Custom Style for Toggle/Download Buttons */
    .stDownloadButton>button, div[data-testid="stVerticalBlock"] > div:nth-child(2) .stButton>button {
        background-color: #222222 !important;
        color: #ff4b4b !important;
        border: 1px solid #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
def convert_to_seconds(time_str):
    """Converts MM:SS or HH:MM:SS to total seconds for st.audio."""
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except: return 0
    return 0

def highlight(text, keyword):
    if not keyword: return text
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group()}</mark>", text)

def parse_llm_output(text):
    segments = []
    blocks = re.split(r"(?=Segment \d+)", text.strip())
    for block in blocks:
        if not block.strip(): continue
        try:
            segments.append({
                "title": re.search(r"Title:\s*(.*)", block).group(1),
                "start_time": re.search(r"Start Time:\s*(.*)", block).group(1),
                "end_time": re.search(r"End Time:\s*(.*)", block).group(1),
                "keywords": [k.strip() for k in re.search(r"Keywords:\s*(.*)", block).group(1).split(",")],
                "summary": re.search(r"Summary:\s*(.*)", block, re.S).group(1).strip()
            })
        except: continue
    return segments

def toggle_transcript():
    """Callback to flip visibility state."""
    st.session_state.show_transcript = not st.session_state.show_transcript

# ================== SESSION STATE ==================
if "segments" not in st.session_state: st.session_state.segments = None
if "transcript" not in st.session_state: st.session_state.transcript = ""
if "current_time" not in st.session_state: st.session_state.current_time = 0
if "show_transcript" not in st.session_state: st.session_state.show_transcript = False

# ================== SIDEBAR ==================
API_KEY = os.getenv("GEMINI_API_KEY")
with st.sidebar:
    st.title("🎧 Podcast Studio")
    uploaded_audio = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    process_btn = st.button("🚀 Analyze Podcast")
    if not API_KEY: 
        st.error("🔑 Set GEMINI_API_KEY in Env Variables.")

# ================== PROCESSING ==================
if process_btn and uploaded_audio:
    with st.spinner("⏳ Stage 1: Transcribing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_audio.read())
            audio_path = tmp.name
        
        whisper = WhisperModel("base", device="cpu", compute_type="int8")
        fw_segments, _ = whisper.transcribe(audio_path)
        
        full_text, raw_data = "", []
        for s in fw_segments:
            full_text += s.text + " "
            raw_data.append({"start": s.start, "end": s.end, "text": s.text})
        st.session_state.transcript = full_text

        chunks, current = [], raw_data[0]
        for seg in raw_data[1:]:
            if seg["end"] - current["start"] <= 120:
                current["text"] += " " + seg["text"]
                current["end"] = seg["end"]
            else:
                chunks.append(current); current = seg
        chunks.append(current)

    with st.spinner("🤖 Stage 2: AI Analysis..."):
        genai.configure(api_key=API_KEY)
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            target_model = next((m for m in available_models if "flash" in m), "models/gemini-pro")
            model = genai.GenerativeModel(target_model)
        except:
            model = genai.GenerativeModel("gemini-pro")

        payload = ""
        for i, c in enumerate(chunks, 1):
            payload += f"\nSegment {i}\nStart: {int(c['start'])}s\nEnd: {int(c['end'])}s\nText: {c['text']}\n"

        prompt = f"""Act as a podcast editor. Format exactly:
        Segment X
        Title: [Title]
        Start Time: [MM:SS]
        End Time: [MM:SS]
        Keywords: [key1, key2]
        Summary: [Summary]

        Data:
        {payload}"""
        
        try:
            response = model.generate_content(prompt)
            st.session_state.segments = parse_llm_output(response.text)
            st.success("✅ Success!")
        except Exception as e:
            st.error(f"AI Error: {e}")

# ================== UI DISPLAY ==================
if st.session_state.segments:
    st.audio(uploaded_audio, start_time=st.session_state.current_time)
    
    l, r = st.columns([1.1, 0.9], gap="large")
    with l:
        q = st.text_input("🔍 Search Segments")
        for i, seg in enumerate(st.session_state.segments):
            if q and q.lower() not in (seg["title"]+seg["summary"]).lower(): continue
            kw_tags = "".join([f"<span class='keyword-tag'>#{k}</span>" for k in seg["keywords"]])
            st.markdown(f'''
                <div class="segment-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#ff4b4b; font-size:1.2rem;">{highlight(seg["title"],q)}</b>
                        <span style="color:#ffffff; background:#333; padding:2px 8px; border-radius:4px;">{seg["start_time"]}</span>
                    </div>
                    <p style="margin-top:10px;">{highlight(seg["summary"],q)}</p>
                    {kw_tags}
                </div>
            ''', unsafe_allow_html=True)
            if st.button(f"🎯 Jump to {seg['start_time']}", key=f"j_{i}"):
                st.session_state.current_time = convert_to_seconds(seg["start_time"])
                st.rerun()

    with r:
        st.subheader("📜 Transcript Tools")
        
        # Toggle visibility button
        toggle_label = "📕 Hide Transcript" if st.session_state.show_transcript else "📖 Show Full Transcript"
        st.button(toggle_label, on_click=toggle_transcript)
        
        if st.session_state.show_transcript:
            st.download_button(
                label="📥 Download .txt",
                data=st.session_state.transcript,
                file_name="podcast_transcript.txt",
                mime="text/plain"
            )
            st.text_area("Full Transcribed Text", st.session_state.transcript, height=500)