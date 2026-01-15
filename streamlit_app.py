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

# Initialize session state for persistence
if 'raw_content' not in st.session_state:
    st.session_state['raw_content'] = ""
if 'translated_content' not in st.session_state:
    st.session_state['translated_content'] = ""

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    reader = easyocr.Reader(['ms', 'en'])
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Load data for source label encoding
    df = pd.read_csv('final_merged_fake_news_data.csv')
    le = LabelEncoder()
    unique_sources = df['site_url'].unique().tolist()
    le.fit(unique_sources)
    
    # Load GNN Model
    model = MultiRelationalGNN(128, 2, 384, len(le.classes_))
    checkpoint = torch.load('best_gnn_malay_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return reader, st_model, model, le, unique_sources

reader, st_model, gnn_model, le_source, known_sources = load_resources()

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_malay_stopwords():
    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ms/master/stopwords-ms.txt"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200: return set(r.text.splitlines())
    except: pass
    return {'dan', 'yang', 'untuk', 'di', 'ke', 'dari', 'itu', 'ini'}

def clean_malay_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'(\w+)-\1', r'\1', text) # Reduplication
    text = re.sub(r'(\w+)2', r'\1', text)   # bunga2 -> bunga
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    iso_stop = get_malay_stopwords()
    return " ".join([w for w in text.split() if w not in iso_stop and len(w) > 1])

# Suspicious words that don't change the model but alert the user
SUSPICIOUS_MALAY_WORDS = {
    'gempar', 'terbongkar', 'rahasia', 'rahsia', 'viralkan', 'sebarkan', 
    'tolong', 'pelik', 'ajaib', 'panas', 'palsu', 'tipu', 'fitnah',
    'kecoh', 'mustahil', 'untung', 'percuma', 'hadiah', 'mengejutkan'
}

def detect_suspicious_words(text):
    text_lower = text.lower()
    found = [word for word in SUSPICIOUS_MALAY_WORDS if re.search(rf'\b{word}\b', text_lower)]
    return found

# --- APP LAYOUT ---
st.title("🛡️ Malay News Graph-AI Verifier")

# Section 1: Source Input (Top)
with st.container(border=True):
    st.subheader("1. Source Metadata")
    user_source = st.text_input("Enter News Source URL or Name", placeholder="e.g. hmetro.com.my")
    source_to_use = "unknown"
    if user_source:
        clean_s = user_source.strip().lower()
        if clean_s in known_sources:
            source_to_use = clean_s
            st.caption(f"✅ Recognized: **{source_to_use}**")
        else:
            st.caption("⚠️ Source not in training data. Mapping to 'unknown'.")
    else:
        st.caption("ℹ️ No source provided. Defaulting to 'unknown'.")

st.write("---")

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
        if up:
            img = Image.open(up).convert('L')
            extracted = " ".join([t[1] for t in reader.readtext(np.array(img))])
    elif mode == "PDF":
        up = st.file_uploader("Upload PDF", type=['pdf'])
        if up:
            pages = convert_from_bytes(up.read())
            extracted = " ".join([" ".join([t[1] for t in reader.readtext(np.array(p))]) for p in pages])
    elif mode == "Word":
        up = st.file_uploader("Upload Word", type=['docx'])
        if up:
            extracted = "\n".join([p.text for p in docx.Document(up).paragraphs])

    if extracted: st.session_state['raw_content'] = extracted
    st.session_state['raw_content'] = st.text_area("Step 1: Raw Text Output", value=st.session_state['raw_content'], height=150)

    if st.button("Detect & Translate to Malay"):
        if st.session_state['raw_content']:
            with st.spinner("Processing language..."):
                lang = detect(st.session_state['raw_content'])
                if lang != 'ms':
                    st.session_state['translated_content'] = GoogleTranslator(source='auto', target='ms').translate(st.session_state['raw_content'])
                else:
                    st.session_state['translated_content'] = st.session_state['raw_content']
                    st.toast("Language is already Malay.")

with col2:
    st.subheader("🔍 3. Analysis & Results")
    review_text = st.text_area("Step 2: Review/Edit Malay Text", value=st.session_state['translated_content'], height=150)
    st.session_state['translated_content'] = review_text

    btn_row = st.columns(2)
    if btn_row[0].button("Run Prediction", type="primary"):
        if not review_text:
            st.warning("Please provide Malay text first.")
        else:
            with st.spinner("GNN Analysis in progress..."):
                cleaned = clean_malay_text(review_text)
                emb = torch.tensor(st_model.encode([cleaned]), dtype=torch.float)
                source_idx = le_source.transform([source_to_use])[0]
                
                # Setup Graph Object
                data = HeteroData()
                data['article'].x = emb
                data['source'].x = torch.eye(len(le_source.classes_))
                data['article','published_by','source'].edge_index = torch.tensor([[0],[source_idx]])
                
                with torch.no_grad():
                    out = gnn_model(data.x_dict, data.edge_index_dict)
                    prob = F.softmax(out['article'], dim=-1)
                    pred = out['article'].argmax(dim=-1).item()
                
                # Results Display
                st.markdown("### Verification Result")
                confidence = prob[0][pred].item() * 100
                
                if pred == 1:
                    st.success("✅ **LIKELY REAL NEWS**")
                else:
                    st.error("🚨 **LIKELY FAKE NEWS**")
                
                st.metric("Model Confidence", f"{confidence:.2f}%")
                st.progress(confidence / 100)

                # Suspicious Word Flagging (Informational)
                flagged = detect_suspicious_words(review_text)
                if flagged:
                    if pred == 0: # If Fake
                        st.warning(f"⚠️ Found {len(flagged)} sensationalist terms: " + ", ".join([f"`{w.upper()}`" for w in flagged]))
                    else: # If Real but sensational
                        with st.expander("Note on Language Style"):
                            st.write(f"Article contains emotional keywords ({', '.join(flagged)}) but verified as Real based on source/context.")

                with st.expander("Technical Deep-Dive"):
                    st.write(f"**Source used:** {source_to_use}")
                    st.write("**Cleaned for AI:**", cleaned)

    if btn_row[1].button("Reset Everything"):
        st.session_state['raw_content'] = ""
        st.session_state['translated_content'] = ""
        st.rerun()