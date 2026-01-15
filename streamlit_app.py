import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import easyocr
import re
import docx
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
    le.fit(df['site_url'].unique())
    
    model = MultiRelationalGNN(128, 2, 384, len(le.classes_))
    checkpoint = torch.load('best_gnn_malay_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return reader, st_model, model, le

reader, st_model, gnn_model, le_source = load_resources()

# --- HELPER FUNCTIONS ---
def process_image(image):
    """Enhance image for better OCR results."""
    img = Image.open(image).convert('L') 
    img = ImageOps.autocontrast(img)     
    img = img.filter(ImageFilter.SHARPEN)
    return np.array(img)

# --- APP LAYOUT ---
st.title("🛡️ Unified Malay News Verification")

with st.sidebar:
    st.header("Optional Metadata")
    source_options = ["Unknown"] + list(le_source.classes_)
    selected_source = st.selectbox("Article Source", source_options)
    st.info("The GNN uses the source relationship to improve accuracy.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 1. Input & Extraction")
    mode = st.selectbox("Method", ["Manual", "Camera", "Image", "PDF", "Word"])
    
    # Input Logic
    text_out = ""
    if mode == "Camera":
        cam_image = st.camera_input("Scan Article")
        if cam_image:
            processed_img = process_image(cam_image)
            with st.spinner("OCR Processing..."):
                results = reader.readtext(processed_img)
                text_out = " ".join([t[1] for t in results])
    elif mode == "Image":
        up = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if up:
            processed_img = process_image(up)
            text_out = " ".join([t[1] for t in reader.readtext(processed_img)])
    elif mode == "PDF":
        up = st.file_uploader("Upload PDF", type=['pdf'])
        if up:
            pages = convert_from_bytes(up.read())
            for pg in pages:
                text_out += " ".join([t[1] for t in reader.readtext(np.array(pg))]) + " "
    elif mode == "Word":
        up = st.file_uploader("Upload Word", type=['docx'])
        if up:
            doc = docx.Document(up)
            text_out = "\n".join([p.text for p in doc.paragraphs])
    
    if text_out:
        st.session_state['raw_content'] = text_out

    # Raw Text Display
    st.session_state['raw_content'] = st.text_area("Extracted Raw Text:", value=st.session_state['raw_content'], height=150)

    # Translation Trigger
    if st.button("Detect Language & Translate to Malay"):
        if st.session_state['raw_content']:
            try:
                lang = detect(st.session_state['raw_content'])
                if lang != 'ms':
                    with st.spinner(f"Translating from {lang.upper()}..."):
                        translated = GoogleTranslator(source='auto', target='ms').translate(st.session_state['raw_content'])
                        st.session_state['translated_content'] = translated
                        st.toast(f"Translated from {lang.upper()} to Malay!")
                else:
                    st.session_state['translated_content'] = st.session_state['raw_content']
                    st.toast("Language is already Malay.")
            except:
                st.session_state['translated_content'] = st.session_state['raw_content']
                st.error("Could not detect language. Copying raw text to Malay field.")

with col2:
    st.subheader("🔍 2. Review & Predict")
    
    # Editable Translated Area
    final_malay_text = st.text_area(
        "Final Malay Text (Review/Edit here):", 
        value=st.session_state['translated_content'], 
        height=150,
        help="The model only predicts based on this Malay text."
    )
    st.session_state['translated_content'] = final_malay_text

    c1, c2 = st.columns(2)
    if c1.button("Predict News Validity", type="primary"):
        if not final_malay_text:
            st.warning("Please translate or provide Malay text first.")
        else:
            with st.spinner("GNN Analysis..."):
                # Preprocessing
                clean = " ".join(re.sub(r'[^a-zA-Z0-9\s]', ' ', final_malay_text).lower().split())
                emb = torch.tensor(st_model.encode([clean]), dtype=torch.float)
                
                # Source Metadata
                source_idx = 0 if selected_source == "Unknown" else le_source.transform([selected_source])[0]
                
                # GNN Forward Pass
                data = HeteroData()
                data['article'].x = emb
                data['source'].x = torch.eye(len(le_source.classes_))
                data['article','published_by','source'].edge_index = torch.tensor([[0],[source_idx]])
                
                with torch.no_grad():
                    out = gnn_model(data.x_dict, data.edge_index_dict)
                    prob = F.softmax(out['article'], dim=-1)
                    pred = out['article'].argmax(dim=-1).item()
                
                if pred == 1:
                    st.success(f"### ✅ LIKELY REAL\n**Confidence:** {prob[0][1]*100:.2f}%")
                else:
                    st.error(f"### 🚨 LIKELY FAKE\n**Confidence:** {prob[0][0]*100:.2f}%")

    if c2.button("Reset Everything"):
        st.session_state['raw_content'] = ""
        st.session_state['translated_content'] = ""
        st.rerun()