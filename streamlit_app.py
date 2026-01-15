import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import easyocr
import re
import unicodedata
import docx
from PIL import Image
from pdf2image import convert_from_bytes
from deep_translator import GoogleTranslator
from transformers import pipeline
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sage_model import MultiRelationalGNN

# --- UI CONFIG ---
st.set_page_config(page_title="Malay News AI Verification", layout="wide")

if 'content' not in st.session_state:
    st.session_state['content'] = ""

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    reader = easyocr.Reader(['ms', 'en'])
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    df = pd.read_csv('final_merged_fake_news_data.csv')
    le = LabelEncoder()
    le.fit(df['site_url'].unique())
    
    model = MultiRelationalGNN(128, 2, 384, len(le.classes_))
    checkpoint = torch.load('best_gnn_malay_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return reader, summarizer, st_model, model, le

try:
    reader, summarizer, st_model, gnn_model, le_source = load_resources()
except Exception as e:
    st.error(f"Initialization Error: {e}")

# --- APP LAYOUT ---
st.title("🛡️ Unified Malay News Verification")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input")
    mode = st.selectbox("Method", ["Manual", "Image", "PDF", "Word"])
    text_out = ""
    
    if mode == "Image":
        up = st.file_uploader("Upload Image")
        if up: text_out = " ".join([t[1] for t in reader.readtext(np.array(Image.open(up)))])
    elif mode == "PDF":
        up = st.file_uploader("Upload PDF")
        if up:
            for pg in convert_from_bytes(up.read()):
                text_out += " ".join([t[1] for t in reader.readtext(np.array(pg))]) + " "
    elif mode == "Word":
        up = st.file_uploader("Upload Word")
        if up: text_out = "\n".join([p.text for p in docx.Document(up).paragraphs])
    
    if text_out: st.session_state['content'] = text_out
    
    final_text = st.text_area("Review Content:", value=st.session_state['content'], height=300)
    st.session_state['content'] = final_text

with col2:
    st.subheader("🔍 Analysis")
    if final_text:
        if st.button("Run Prediction"):
            with st.spinner("Analyzing..."):
                # GNN Logic
                malay = GoogleTranslator(source='auto', target='ms').translate(final_text[:1000])
                clean = " ".join(re.sub(r'[^a-zA-Z0-9\s]', ' ', malay).lower().split())
                emb = torch.tensor(st_model.encode([clean]), dtype=torch.float)
                
                data = HeteroData()
                data['article'].x = emb
                data['source'].x = torch.eye(len(le_source.classes_))
                data['article','published_by','source'].edge_index = torch.tensor([[0],[0]])
                
                with torch.no_grad():
                    out = gnn_model(data.x_dict, data.edge_index_dict)
                    prob = F.softmax(out['article'], dim=-1)
                    pred = out['article'].argmax(dim=-1).item()
                
                if pred == 1: st.success(f"REAL NEWS ({prob[0][1]*100:.1f}%)")
                else: st.error(f"FAKE NEWS ({prob[0][0]*100:.1f}%)")
        
        if st.button("Reset"):
            st.session_state['content'] = ""
            st.rerun()