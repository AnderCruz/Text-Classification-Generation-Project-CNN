Aqui está a tradução do seu código para inglês, mantendo a funcionalidade e adaptando os comentários e strings para inglês:

```python
import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gdown
import os

# URLs and file paths
model_url = "https://drive.google.com/uc?id=1_DYLBo0fzko99hFWdYISbaPvNEd2Q9JH"
model_path = "seer_model.keras"
vectorizer_url = "https://drive.google.com/uc?id=1FD04fRz4l9zdnfije4S8Z8kG2Z044o65"
vectorizer_path = "vectorizer.pkl"

# Function to download files if they don't exist
def download_from_drive(url, path):
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)

# Function to load model and vectorizer
@st.cache_resource
def load_model():
    download_from_drive(model_url, model_path)
    download_from_drive(vectorizer_url, vectorizer_path)

    loaded_model = tf.keras.models.load_model(model_path)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return loaded_model, vectorizer

# Function to predict next words
def predict_next_words(model, vectorizer, text_sequence, num_words=3):
    token_list = vectorizer([text_sequence])[0].numpy()
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]

    top_indices = np.argpartition(predicted_probs, -num_words)[-num_words:]
    top_indices = top_indices[np.argsort(-predicted_probs[top_indices])]
    predicted_words = [vectorizer.get_vocabulary()[index] for index in top_indices]
    return predicted_words

# Model settings
max_vocab_size = 2000
max_sequence_len = 50

# Load model and vectorizer
loaded_model, vectorizer = load_model()

# Streamlit interface
st.title("🔮 Next Word Prediction")
input_text = st.text_input("Enter a text sequence:")

if st.button("Predict"):
    if input_text:
        try:
            predicted_words = predict_next_words(loaded_model, vectorizer, input_text)
            st.info("Most likely words:")
            for word in predicted_words:
                st.success(word)
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter some text.")
```

Se quiser, posso também **adaptar o código para inglês completo, incluindo nomes de funções e variáveis**, deixando-o totalmente “Pythonic” para uso internacional.

Quer que eu faça isso?
