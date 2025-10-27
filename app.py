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



def predict_next_words(model, vectorizer, text_sequence, num_words=3):
    token_list = vectorizer([text_sequence])[0].numpy()
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]

    # Encontra os índices das 3 maiores probabilidades
    top_indices = np.argpartition(predicted_probs, -num_words)[-num_words:]

    # Ordena os índices em ordem decrescente de probabilidade
    top_indices = top_indices[np.argsort(-predicted_probs[top_indices])]

    # Mapeia os índices para as palavras correspondentes
    predicted_words = [vectorizer.get_vocabulary()[index] for index in top_indices]

    return predicted_words


#loaded_model = tf.keras.models.load_model('modelo_vidente.h5')
    #st.success("Modelo carregado com sucesso!")

max_vocab_size = 20000
max_sequence_len = 50


# Tokenizar o texto
loaded_model, vectorizer = carrega_modelo()

# Interface Streamlit
st.title("🔮 Previsão de Próximas Palavras")

input_text = st.text_input("Digite uma sequência de texto:")

if st.button("Prever"):
    if input_text:
        try:
            predicted_words = predict_next_words(loaded_model, vectorizer, input_text) #predict_next_words(loaded_model, vectorizer, input_text)
            
            st.info("Palavras mais prováveis:")
            for word in predicted_words:
                st.success(word)  # Cada palavra em uma caixa de sucesso
        except Exception as e:
            st.error(f"Erro na previsão: {e}")
    else:
        st.warning("Por favor, insira algum texto.")
