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

# Initialize session state for content persistence
if 'raw_content' not in st.session_state:
    st.session_state['raw_content'] = ""
if 'translated_content' not in st.session_state:
    st.session_state['translated_content'] = ""

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    reader = easyocr.Reader(['ms', 'en'])
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Load data to get source mapping
    df = pd.read_csv('final_merged_fake_news_data.csv')
    le = LabelEncoder()
    unique_sources = df['site_url'].unique().tolist()
    le.fit(unique_sources)
    
    # Load GNN Architecture
    model = MultiRelationalGNN(128, 2, 384, len(le.classes_))
    checkpoint = torch.load('best_gnn_malay_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return reader, st_model, model, le, unique_sources

try:
    reader, st_model, gnn_model, le_source, known_sources = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")

# --- PREPROCESSING & UTILS ---

@st.cache_data
def get_malay_stopwords():
    """Fetch Malay stopwords from ISO repository."""
    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ms/master/stopwords-ms.txt"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return set(response.text.splitlines())
    except:
        pass
    return {'dan', 'yang', 'untuk', 'di', 'ke', 'dari', 'itu', 'ini'} # Fallback

def clean_malay_text(text):
    """Deep cleaning for Malay News Text."""
    text = text.lower()
    # Remove URLs, Emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    # Reduplication (bunga-bunga -> bunga)
    text = re.sub(r'(\w+)-\1', r'\1', text)
    text = re.sub(r'(\w+)2', r'\1', text)
    # Clean characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Stopwords
    iso_stop = get_malay_stopwords()
    words = [w for w in text.split() if w not in iso_stop and len(w) > 1]
    return " ".join(words)

def process_image(image):
    """Enhance image for OCR."""
    img = Image.open(image).convert('L') 
    img = ImageOps.autocontrast(img)     
    img = img.filter(ImageFilter.SHARPEN)
    return np.array(img)

# --- APP LAYOUT ---
st.title("🛡️ Malay News Graph-AI Verifier")
st.markdown("Verify news authenticity using Graph Neural Networks and NLP.")

with st.sidebar:
    st.header("1. Source Metadata")
    user_source = st.text_input("Enter News Source (URL/Name)", placeholder="e.g. hmetro.com.my")
    
    # Mapping logic for source
    source_to_use = "unknown"
    if user_source:
        clean_s = user_source.strip().lower()
        if clean_s in known_sources:
            source_to_use = clean_s
            st.success(f"Verified: {source_to_use}")
        else:
            st.warning("Unknown Source. Using fallback index.")
            source_to_use = "unknown" if "unknown" in known_sources else known_sources[0]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Extraction")
    mode = st.selectbox("Input Method", ["Manual", "Camera", "Image", "PDF", "Word"])
    
    extracted = ""
    if mode == "Camera":
        cam = st.camera_input("Scan Article")
        if cam:
            with st.spinner("OCR Processing..."):
                extracted = " ".join([t[1] for t in reader.readtext(process_image(cam))])
    elif mode == "Image":
        up = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        if up: extracted = " ".join([t[1] for t in reader.readtext(process_image(up))])
    elif mode == "PDF":
        up = st.file_uploader("Upload PDF", type=['pdf'])
        if up:
            pages = convert_from_bytes(up.read())
            extracted = " ".join([" ".join([t[1] for t in reader.readtext(np.array(p))]) for p in pages])
    elif mode == "Word":
        up = st.file_uploader("Upload Word", type=['docx'])
        if up: extracted = "\n".join([p.text for p in docx.Document(up).paragraphs])

    if extracted: st.session_state['raw_content'] = extracted

    st.session_state['raw_content'] = st.text_area("Step 1: Raw Text", value=st.session_state['raw_content'], height=150)

    if st.button("Detect & Translate to Malay"):
        if st.session_state['raw_content']:
            try:
                lang = detect(st.session_state['raw_content'])
                if lang != 'ms':
                    with st.spinner(f"Translating {lang.upper()}..."):
                        st.session_state['translated_content'] = GoogleTranslator(source='auto', target='ms').translate(st.session_state['raw_content'])
                else:
                    st.session_state['translated_content'] = st.session_state['raw_content']
                    st.info("Text is already in Malay.")
            except:
                st.session_state['translated_content'] = st.session_state['raw_content']

with col2:
    st.subheader("🔍 Analysis")
    # Users can edit the translation here
    review_text = st.text_area("Step 2: Review/Edit Malay Translation", value=st.session_state['translated_content'], height=150)
    st.session_state['translated_content'] = review_text

    btn_row = st.columns(2)
    if btn_row[0].button("Run Prediction", type="primary"):
        if not review_text:
            st.warning("No text to analyze.")
        else:
            with st.spinner("Preprocessing & GNN Prediction..."):
                # Clean text before embedding
                cleaned_text = clean_malay_text(review_text)
                emb = torch.tensor(st_model.encode([cleaned_text]), dtype=torch.float)
                
                # Setup Graph Data
                source_idx = le_source.transform([source_to_use])[0]
                data = HeteroData()
                data['article'].x = emb
                data['source'].x = torch.eye(len(le_source.classes_))
                data['article','published_by','source'].edge_index = torch.tensor([[0],[source_idx]])
                
                # Prediction
                with torch.no_grad():
                    out = gnn_model(data.x_dict, data.edge_index_dict)
                    prob = F.softmax(out['article'], dim=-1)
                    pred = out['article'].argmax(dim=-1).item()
                
                if pred == 1:
                    st.success(f"### ✅ LIKELY REAL\n**Confidence:** {prob[0][1]*100:.2f}%")
                else:
                    st.error(f"### 🚨 LIKELY FAKE\n**Confidence:** {prob[0][0]*100:.2f}%")
                
                with st.expander("See Cleaned Text used by AI"):
                    st.write(cleaned_text)

    if btn_row[1].button("Reset App"):
        st.session_state['raw_content'] = ""
        st.session_state['translated_content'] = ""
        st.rerun()