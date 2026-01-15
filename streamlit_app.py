import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import easyocr
import re
import docx
import requests
from PIL import Image, ImageOps, ImageFilter
from pdf2image import convert_from_bytes
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sage_model import MultiRelationalGNN

# Force consistent language detection
DetectorFactory.seed = 0

# --- UI CONFIG ---
st.set_page_config(page_title="Malay News AI Verification", layout="wide")

# Initialize session state
if 'raw_content' not in st.session_state:
    st.session_state['raw_content'] = ""
if 'translated_content' not in st.session_state:
    st.session_state['translated_content'] = ""

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    reader = easyocr.Reader(['ms', 'en'])
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df = pd.read_csv('final_merged_fake_news_data.csv')
    le = LabelEncoder()
    unique_sources = df['site_url'].unique().tolist()
    le.fit(unique_sources)
    
    model = MultiRelationalGNN(128, 2, 384, len(le.classes_))
    checkpoint = torch.load('best_gnn_malay_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return reader, st_model, model, le, unique_sources

reader, st_model, gnn_model, le_source, known_sources = load_resources()

# --- PREPROCESSING FUNCTIONS ---
@st.cache_data
def get_malay_stopwords():
    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ms/master/stopwords-ms.txt"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200: return set(r.text.splitlines())
    except: pass
    return {'dan', 'yang', 'untuk', 'di', 'ke', 'dari'}

def clean_malay_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    iso_stop = get_malay_stopwords()
    return " ".join([w for w in text.split() if w not in iso_stop and len(w) > 1])

# --- APP LAYOUT ---
st.title("🛡️ Malay News Graph-AI Verifier")

# --- NEW: SOURCE METADATA AT THE TOP ---
# This replaces the sidebar version
with st.container(border=True):
    st.subheader("1. Source Metadata")
    user_source = st.text_input(
        "Enter News Source URL or Name", 
        placeholder="e.g. hmetro.com.my",
        help="Type the source name. If it's not in our database, it will be treated as 'unknown'."
    )
    
    source_to_use = "unknown"
    if user_source:
        clean_s = user_source.strip().lower()
        if clean_s in known_sources:
            source_to_use = clean_s
            st.caption(f"✅ Recognized source: **{source_to_use}**")
        else:
            st.caption("⚠️ Source not found in database. Using fallback mapping.")
    else:
        st.caption("ℹ️ No source provided. Defaulting to 'unknown'.")

st.write("---") # Visual separator

# --- EXTRACTION & ANALYSIS COLUMNS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 2. Extraction")
    mode = st.selectbox("Input Method", ["Manual", "Camera", "Image", "PDF", "Word"])
    
    extracted = ""
    if mode == "Camera":
        cam = st.camera_input("Scan Article")
        if cam:
            extracted = " ".join([t[1] for t in reader.readtext(np.array(Image.open(cam)))])
    elif mode == "Image":
        up = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        if up: extracted = " ".join([t[1] for t in reader.readtext(np.array(Image.open(up)))])
    
    # ... (PDF/Word logic remains the same)

    if extracted: st.session_state['raw_content'] = extracted
    st.session_state['raw_content'] = st.text_area("Raw Text Output", value=st.session_state['raw_content'], height=150)

    if st.button("Translate to Malay"):
        if st.session_state['raw_content']:
            lang = detect(st.session_state['raw_content'])
            if lang != 'ms':
                st.session_state['translated_content'] = GoogleTranslator(source='auto', target='ms').translate(st.session_state['raw_content'])
            else:
                st.session_state['translated_content'] = st.session_state['raw_content']

with col2:
    st.subheader("🔍 3. Analysis")
    review_text = st.text_area("Review Malay Translation", value=st.session_state['translated_content'], height=150)
    st.session_state['translated_content'] = review_text

    btn_row = st.columns(2)
    if btn_row[0].button("Run Prediction", type="primary"):
        if review_text:
            cleaned = clean_malay_text(review_text)
            emb = torch.tensor(st_model.encode([cleaned]), dtype=torch.float)
            source_idx = le_source.transform([source_to_use])[0]
            
            # GNN Logic...
            # (Assuming the rest of your GNN data setup here)
            st.success("Analysis complete!")
            
    if btn_row[1].button("Reset App"):
        st.session_state['raw_content'] = ""
        st.session_state['translated_content'] = ""
        st.rerun()