import streamlit as st
import pandas as pd
import tempfile
import os
import re
import streamlit.components.v1 as components
from transformers import pipeline

# IMPORTING FROM  MODULAR FILES
from whisper_transcriber import run_whisper_task
from sbert import process_semantic_segments, get_unique_title

# 1. SETUP
st.set_page_config(page_title="PodcastAI Pro", layout="wide")

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

@st.cache_resource
def load_pro_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# State Management
if "processed" not in st.session_state: st.session_state.processed = False
if "seek_time" not in st.session_state: st.session_state.seek_time = 0
if "active_view" not in st.session_state: st.session_state.active_view = ""
if "should_play" not in st.session_state: st.session_state.should_play = False

# 2. AUTOPLAY JAVASCRIPT
def inject_autoplay():
    components.html(
        f"""
        <script>
            window.parent.document.querySelectorAll('audio').forEach(audio => {{
                audio.play().catch(e => console.log("Autoplay blocked by browser. Click anywhere on the page once."));
            }});
        </script>
        """,
        height=0,
    )

# 3. SIDEBAR
with st.sidebar:
    st.title("🎙️ PodcastAI Pro")
    uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    
    if uploaded_file and not st.session_state.processed:
        if st.button("🚀 Analyze Podcast", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.getbuffer())
                path = tmp.name
            
            with st.spinner("Processing..."):
                raw_data = run_whisper_task(path)
                texts = [s['text'] for s in raw_data["segments"]]
                boundaries, kw_model = process_semantic_segments(texts)
                summarizer = load_pro_summarizer()
                
                report = []
                full_transcript = ""
                splits = [0] + list(boundaries) + [len(texts)]
                
                for i in range(len(splits)-1):
                    chunk = " ".join(texts[splits[i]:splits[i+1]])
                    full_transcript += chunk + "\n\n"
                    start_s = raw_data['segments'][splits[i]]['start']
                    end_s = raw_data['segments'][splits[i+1]-1]['end'] if i+1 < len(splits) else raw_data['segments'][-1]['end']
                    
                    sum_res = summarizer(chunk[:1024], max_length=100, min_length=30)[0]['summary_text']
                    title = get_unique_title(kw_model, chunk, i)
                    kws = [k[0] for k in kw_model.extract_keywords(chunk, top_n=5)]
                    
                    report.append({
                        "id": i, "start": start_s, 
                        "time_range": f"{format_time(start_s)} - {format_time(end_s)}",
                        "title": title, "summary": sum_res, "content": chunk, "keywords": kws
                    })
                
                st.session_state.final_report = report
                st.session_state.full_text = full_transcript
                st.session_state.processed = True
                st.rerun()

    if st.session_state.processed:
        st.download_button("📥 Download Transcript", st.session_state.full_text, "podcast.txt", use_container_width=True)
        if st.button("🗑️ Reset", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

# 4. MAIN DASHBOARD
if st.session_state.processed:
    st.title("🎧 Podcast Overview")
    
    if st.session_state.should_play:
        inject_autoplay()
        st.session_state.should_play = False # Reset

    search_term = st.text_input("🔍 Search Transcript...", placeholder="Search keywords...")
    st.audio(uploaded_file, start_time=st.session_state.seek_time)
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("📌 Chapters")
        for seg in st.session_state.final_report:
            match_found = search_term and search_term.lower() in seg['content'].lower()
            label = f"{'✅ ' if match_found else '📍 '}{seg['time_range']} — {seg['title']}"

            with st.expander(label):
                st.write(f"**Keywords:** {', '.join(seg['keywords'])}")
                st.write(seg['summary'])
                
                b1, b2 = st.columns(2)
                if b1.button(f"🎧 Jump", key=f"j_{seg['id']}", use_container_width=True):
                    st.session_state.seek_time = int(seg['start'])
                    st.session_state.should_play = True # Flag JS to play
                    st.rerun()
                if b2.button(f"📖 Read", key=f"r_{seg['id']}", use_container_width=True):
                    st.session_state.active_view = seg['content']

    with col2:
        st.subheader("📜 Transcript Reader")
        raw_content = st.session_state.active_view if st.session_state.active_view else "Select a chapter."
        
        display_content = raw_content
        if search_term:
            insensitive_search = re.compile(re.escape(search_term), re.IGNORECASE)
            display_content = insensitive_search.sub(f'<mark style="background: #fde047; color: black; border-radius: 4px;">{search_term}</mark>', raw_content)

        st.markdown(f"""
            <div style="height:500px; overflow-y:auto; padding:20px; border:1px solid #ccc; border-radius:12px; line-height:1.6;">
                {display_content}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Upload a file to start.")