import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

@st.cache_resource
def load_model():
    model_path = 'model_fp16_fast.tflite'
    
    # Check if file already exists, if not, download it
    if not os.path.exists(model_path):
        st.info("📥 Downloading model...")
        url = 'https://drive.google.com/file/d/1_DYLBo0fzko99hFWdYISbaPvNEd2Q9JH'
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model download completed!")
        except Exception as e:
            st.error(f"❌ Download error: {str(e)}")
            st.info("🔄 Trying alternative download method...")
            return download_model_alternative(model_path)
    
    # Verify the downloaded file
    try:
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            st.error("❌ Downloaded file is empty")
            os.remove(model_path)
            return download_model(model_path)
    except:
        pass
    
    try:
        # Load the model with error handling
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        st.success("✅ Model loaded successfully!")
        
        # Show model information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        st.sidebar.info(f"**Input shape:** {input_details[0]['shape']}")
        st.sidebar.info(f"**Output shape:** {output_details[0]['shape']}")
        
        return interpreter
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("🔄 Trying to re-download the model...")
        # Remove potentially corrupted file
        if os.path.exists(model_path):
            os.remove(model_path)
        return download_model(model_path)

def predict_next_words(model, vectorizer, text_sequence, num_words=3):
    """Prevê as próximas palavras mais prováveis em uma sequência de texto.

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
#vectorizer = TextVectorization(max_tokens=max_vocab_size, output_sequence_length=max_sequence_len, output_mode='int')

# Adaptar a camada ao corpus
#vectorizer.adapt(corpus)

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
