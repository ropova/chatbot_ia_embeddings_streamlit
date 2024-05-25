import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import random
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import re

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Descargar los recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('wordnet')

# Cargar el modelo y otros datos necesarios
model = load_model('chatbot_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compilar el modelo para inicializar las métricas

with open('tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
    tokenizer = Tokenizer()
    tokenizer.word_index = tokenizer_data

with open('label_encoder.json', 'r') as f:
    label_encoder_data = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_data)

with open('word_index.json', 'r') as f:
    word_index = json.load(f)
embedding_matrix = np.load('embedding_matrix.npy')

max_length = 20

# Inicializar el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar intents
with open('data.json', encoding='utf-8') as f:
    intents = json.load(f)

# Función para normalizar texto eliminando puntuación y espacios innecesarios
def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)  # Eliminar espacios múltiples
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
    return text.strip().lower()

# Función para tokenizar y lematizar texto utilizando NLTK
def tokenize_and_lemmatize(text):
    text = normalize_text(text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return ' '.join(lemmatized_tokens)

import numpy as np

# Función para detectar la intención del usuario utilizando la red neuronal
def intent_detection(user_input):
    tokenized_input = tokenize_and_lemmatize(user_input)
    input_seq = tokenizer.texts_to_sequences([tokenized_input])
    input_seq = pad_sequences(input_seq, maxlen=max_length)
    prediction = model.predict(input_seq)
    encoded_response = np.argmax(prediction)
    response = label_encoder.inverse_transform([encoded_response])[0]
    score = prediction[0][encoded_response]
    return response, score

# Función para responder al usuario según la intención detectada
def respond_to_user(user_input, intents):
    # Verificar si el usuario envió un saludo
    saludos = ["hola", "hello", "saludos", "qué tal", "buenos días", "buenas tardes", "buenas noches"]
    if any(saludo in user_input.lower() for saludo in saludos):
        st.session_state["estado"] = "pregunta_como_estas"
        return "Hola, ¿cómo estás?", "saludo", 1.0  # Responder al saludo

    # Verificar si el usuario responde "sí"
    if "sí" in user_input.lower() or "si" in user_input.lower():
        return "¿En qué puedo ayudarte?", "si", 1.0

    # Verificar si el usuario responde "no"
    if "no" in user_input.lower():
        return "Espero verte pronto", "no", 1.0

    # Verificar el estado para continuar el flujo
    if st.session_state.get("estado") == "pregunta_como_estas":
        if "bien" in user_input.lower():
            st.session_state["estado"] = "interaccion_normal"
            return "Me alegra escuchar eso. ¿En qué puedo ayudarte?", "bien", 1.0
        elif "mal" in user_input.lower() or "no bien" in user_input.lower() or "no estoy bien" in user_input.lower():
            st.session_state["estado"] = "interaccion_normal"
            return "Lamento escuchar eso. ¿En qué puedo ayudarte?", "mal", 1.0

    
    # Detectar la intención normalmente si no es un saludo
    st.session_state["estado"] = "interaccion_normal"
    intent, score = intent_detection(user_input)
    print(f"Detected intent: {intent} with score: {score}")  # Imprimir las intenciones y puntajes en la consola
    threshold = 0.9  # Umbral para la confianza en la intención
    if score < threshold:
        return 'No entiendo. ¿Puedes reformular la pregunta?', intent, score
    for intent_obj in intents['intents']:
        if intent_obj['tag'] == intent:
            responses = intent_obj['responses']
            return random.choice(responses), intent, score
    return 'No entiendo. ¿Puedes reformular la pregunta?', intent, score

load_css()

# Interfaz de Streamlit
st.title("IntelliGen Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "estado" not in st.session_state:
    st.session_state.estado = "inicial"

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("image", None)):
        st.markdown(message["content"])

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="static/images/chatbot.png"):
        st.markdown("Hola, ¿cómo puedo ayudarte?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?", "image": "static/images/chatbot.png"})

if prompt := st.chat_input("Escribe aquí..."):
    with st.chat_message("user", avatar="static/images/gente.png"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "image": "static/images/gente.png"})     

    response, intent, score = respond_to_user(prompt, intents)

    with st.chat_message("assistant", avatar="static/images/chatbot.png"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response, "image": "static/images/chatbot.png"})

# Botón de limpiar
if st.button("Limpiar conversación"):
    st.session_state.messages = []
    st.session_state.estado = "inicial"
    st.experimental_rerun()  # Reiniciar la aplicación
