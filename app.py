import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import gdown
import os

MODEL_PATH = "seer_model.keras"  # ou .h5
VECTORIZER_PATH = "vectorizer.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1_DYLBo0fzko99hFWdYISbaPvNEd2Q9JH"
VECTORIZER_URL = "https://drive.google.com/uc?id=1FD04fRz4l9zdnfije4S8Z8kG2Z044o65"

def baixar_arquivo(url, caminho_destino):
    if not os.path.exists(caminho_destino):
        st.write(f"📥 Baixando arquivo: {caminho_destino} ...")
        gdown.download(url, caminho_destino, quiet=False)
        st.success(f"✅ {caminho_destino} baixado com sucesso!")
    else:
        st.info(f"📦 {caminho_destino} já existe.")

baixar_arquivo(MODEL_URL, MODEL_PATH)
baixar_arquivo(VECTORIZER_URL, VECTORIZER_PATH)

try:
    modelo = load_model(MODEL_PATH)
except Exception as e:
    st.warning(f"⚠️ Tentando carregar com safe_mode=False devido a erro: {e}")
    modelo = load_model(MODEL_PATH, safe_mode=False)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

st.success("✅ Modelo e vectorizer carregados com sucesso!")
