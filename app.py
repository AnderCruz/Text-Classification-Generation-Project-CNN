import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gdown
import os
import requests
import io

# ==============================
# CONFIGURAÇÕES DO APLICATIVO
# ==============================
st.set_page_config(
    page_title="🔮 Previsor de Próximas Palavras",
    page_icon="✨",
    layout="centered"
)

st.title("🔮 Previsor de Próximas Palavras com IA")
st.markdown("---")

# ==============================
# PARÂMETROS GERAIS
# ==============================
MODEL_PATH = "seer_model.keras"
VECTORIZER_PATH = "vectorizer.pkl"
MAX_SEQUENCE_LEN = 50
NUM_WORDS = 3

# URLs do Google Drive
MODEL_URL = "https://drive.google.com/file/d/1_DYLBo0fzko99hFWdYISbaPvNEd2Q9JH"  # substitua pelo ID real
VECTORIZER_URL = "https://drive.google.com/file/d/1FD04fRz4l9zdnfije4S8Z8kG2Z044o65"  # substitua pelo ID real


# ==============================
# FUNÇÕES DE DOWNLOAD
# ==============================
def download_file(url, output_path):
    """Baixa arquivo do Google Drive com fallback."""
    try:
        if not os.path.exists(output_path):
            st.info(f"📥 Baixando arquivo: {output_path} ...")
            gdown.download(url, output_path, quiet=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.success(f"✅ {os.path.basename(output_path)} baixado com sucesso!")
        else:
            st.error(f"❌ Falha no download de {output_path}")
    except Exception as e:
        st.error(f"❌ Erro ao baixar {output_path}: {e}")


# ==============================
# CARREGAMENTO DO MODELO
# ==============================
@st.cache_resource
def load_model_and_vectorizer():
    """Carrega modelo Keras e vectorizer a partir de disco ou Google Drive."""
    # Baixar arquivos se necessário
    if not os.path.exists(MODEL_PATH):
        download_file(MODEL_URL, MODEL_PATH)
    if not os.path.exists(VECTORIZER_PATH):
        download_file(VECTORIZER_URL, VECTORIZER_PATH)

    try:
        st.info("🔄 Carregando modelo e vectorizer...")
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        st.success("✅ Modelo e vectorizer carregados com sucesso!")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo/vectorizer: {e}")
        return None, None


# ==============================
# FUNÇÃO DE PREVISÃO
# ==============================
def predict_next_words(model, vectorizer, text_sequence, num_words=NUM_WORDS):
    """Prevê as próximas palavras mais prováveis em uma sequência de texto."""
    try:
        # Converte o texto em tokens
        token_list = vectorizer([text_sequence])[0].numpy()
        token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN - 1, padding='pre')

        # Faz a predição
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Obtém os índices das palavras mais prováveis
        top_indices = np.argpartition(predicted_probs, -num_words)[-num_words:]
        top_indices = top_indices[np.argsort(-predicted_probs[top_indices])]

        # Mapeia índices para palavras
        predicted_words = [vectorizer.get_vocabulary()[i] for i in top_indices]
        return predicted_words
    except Exception as e:
        st.error(f"Erro ao prever: {e}")
        return []


# ==============================
# INTERFACE PRINCIPAL
# ==============================
def main():
    st.sidebar.title("🧠 Sobre o App")
    st.sidebar.info("""
    Este aplicativo utiliza um modelo de **Rede Neural com Keras** para prever 
    as próximas palavras prováveis de uma frase digitada.
    """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Dica:")
    st.sidebar.write("Quanto maior o contexto do texto, mais precisas serão as previsões!")

    model, vectorizer = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique sua conexão ou o ID do arquivo.")
        return

    # Entrada do usuário
    input_text = st.text_input("✍️ Digite uma sequência de texto:")

    if st.button("🔮 Prever Próximas Palavras"):
        if not input_text.strip():
            st.warning("Por favor, insira algum texto.")
        else:
            with st.spinner("Gerando previsões..."):
                predicted = predict_next_words(model, vectorizer, input_text)
                if predicted:
                    st.success("✨ Palavras mais prováveis:")
                    for i, word in enumerate(predicted, 1):
                        st.write(f"**{i}.** {word}")
                else:
                    st.error("Não foi possível gerar previsões.")

    st.markdown("---")
    st.caption("Desenvolvido com ❤️ e TensorFlow.")


if __name__ == "__main__":
    main()
