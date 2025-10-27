from tensorflow.keras.models import load_model
import pickle
import gdown
import os

MODEL_PATH = "seer_model.keras"
VECTORIZER_PATH = "vectorizer.pkl"

# Substitua com os IDs reais dos arquivos do Drive
MODEL_URL = "https://drive.google.com/file/d/1_DYLBo0fzko99hFWdYISbaPvNEd2Q9JH"
VECTORIZER_URL = "https://drive.google.com/file/d/1FD04fRz4l9zdnfije4S8Z8kG2Z044o65"

def baixar_arquivo(url, caminho_destino):
    if not os.path.exists(caminho_destino):
        st.write(f"📥 Baixando arquivo: {caminho_destino} ...")
        gdown.download(url, caminho_destino, quiet=False)
        st.success(f"✅ {caminho_destino} baixado com sucesso!")
    else:
        st.info(f"📦 {caminho_destino} já existe.")

# Baixa e carrega o modelo
baixar_arquivo(MODEL_URL, MODEL_PATH)
baixar_arquivo(VECTORIZER_URL, VECTORIZER_PATH)
