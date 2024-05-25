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
import unicodedata

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

# Función para detectar la intención del usuario utilizando el modelo
def intent_detection(user_input):
    tokenized_input = tokenize_and_lemmatize(user_input)
    input_seq = tokenizer.texts_to_sequences([tokenized_input])
    input_seq = pad_sequences(input_seq, maxlen=max_length)
    prediction = model.predict(input_seq)
    encoded_responses = np.argsort(prediction)[0][-3:][::-1]  # Obtener los índices de las 3 mayores predicciones
    top_intents = label_encoder.inverse_transform(encoded_responses)  # Convertir los índices en las intenciones
    top_scores = [prediction[0][idx] for idx in encoded_responses]  # Obtener los puntajes correspondientes
    return top_intents, top_scores

# Respuestas posibles para los saludos y "¿cómo estás?"
respuestas_saludo = ["Hola, ¿cómo estás?",
                    "¡Hola! ¿Cómo te va?",
                    "Saludos, ¿cómo te encuentras?",
                    "Hola, ¿cómo has estado?",
                    "¿Qué tal estás hoy?",
                    "¿Cómo te sientes?",
                    "Hola, ¿cómo va todo?",
                    "¿Cómo te encuentras?",
                    "¿Cómo va tu día?",
                    "Hola, ¿cómo te sientes hoy?",
                    "¿Cómo andas?"]

respuestas_bien = ["Me alegra escuchar eso. ¿En qué puedo ayudarte?", 
                    "¡Genial! ¿Cómo puedo asistirte?", 
                    "¡Qué bueno! ¿En qué te puedo ayudar?",
                    "Eso suena fantástico. ¿Hay algo específico en lo que pueda colaborar?",
                    "Qué bueno saberlo. ¿Necesitas algún tipo de ayuda o apoyo adicional?",
                    "Me alegra escuchar eso. ¿Hay algo en lo que pueda contribuir para que tu día sea aún mejor?",
                    "Maravilloso. ¿Hay algo en lo que necesites ayuda o asistencia?",
                    "Me alegra mucho escuchar eso. ¿Cómo puedo ayudarte hoy?",
                    "Excelente. ¿Hay algo en lo que necesites que te ayude o apoye?",
                    "Me alegra saberlo. ¿En qué puedo colaborar para que sigas sintiéndote así de bien?"]


respuestas_mal = ["Lamento escuchar eso. ¿En qué puedo ayudarte?", 
                "Lo siento, ¿cómo puedo asistirte?", 
                "Qué pena, ¿en qué te puedo ayudar?",
                "Oh, eso no suena bien. ¿Puedo hacer algo por ti?",
                "Me entristece escuchar eso. ¿Hay algo en lo que pueda colaborar?",
                "Lo siento mucho. ¿Hay algo que pueda hacer para ayudar?",
                "Qué lástima. ¿Hay algo en lo que pueda ser de ayuda?",
                "Vaya, eso no suena nada bien. ¿Necesitas algún tipo de apoyo?",
                "Oh, lo siento mucho. ¿Hay algo que pueda hacer para hacerte sentir mejor?",
                "Lo siento mucho por eso. ¿Puedo hacer algo para ayudar a mejorar tu día?",
                "Qué pena escuchar eso. ¿Cómo puedo brindarte apoyo en este momento?"]


# Palabras que indican que el usuario está bien
palabras_bien = ["bien", 
                "super", 
                "lo maximo", 
                "excelente", 
                "bendecido", 
                "genial", 
                "fantastico", 
                "maravilloso", 
                "increible", 
                "estupendo", 
                "fenomenal"]

# Palabras que indican que el usuario no está bien
palabras_mal = ["mal", 
                "regular", 
                "no me siento bien", 
                "no estoy bien", 
                "desanimado", 
                "decaido", 
                "apagado", 
                "mas o menos", 
                "bajo de nota", 
                "triste", 
                "abatido", 
                "preocupado", 
                "desalentado", 
                "cansado", 
                "agobiado"]


# Función para eliminar tildes y símbolos de una cadena
def remove_accents_and_symbols(text):
    text = unicodedata.normalize('NFD', text)  # Normalizar caracteres Unicode
    text = text.encode('ascii', 'ignore').decode('utf-8')  # Convertir caracteres Unicode a ASCII
    return text

# Función para responder al usuario según la intención detectada
def respond_to_user(user_input, intents):
    # Convertir la entrada del usuario a minúsculas
    user_input_lower = user_input.lower()

    # Eliminar tildes y símbolos de la entrada del usuario
    user_input_clean = remove_accents_and_symbols(user_input_lower)

    # Verificar si el usuario dice "si"
    if "si" in user_input_clean:
        return random.choice(["¿En qué puedo ayudarte?", "¿Qué puedo hacer por ti?", "¿Hay algo en lo que pueda colaborar?"]), "ayuda", 1.0

    # Verificar si el usuario dice "no"
    if "no" in user_input_clean:
        return random.choice(["¡Que tengas un excelente día!", "Espero verte nuevamente por aquí."]), "despedida", 1.0

    # Verificar si la palabra "hola" está presente en la entrada del usuario
    if "hola" in user_input_clean:
        st.session_state["estado"] = "pregunta_como_estas"
        return random.choice(respuestas_saludo), "saludo", 1.0  # Responder al saludo

    # Verificar el estado para continuar el flujo
    if st.session_state.get("estado") == "pregunta_como_estas":
        if any(palabra in user_input_clean for palabra in palabras_bien) and not any(neg_word in user_input_clean for neg_word in palabras_mal):
            st.session_state["estado"] = "interaccion_normal"
            return random.choice(respuestas_bien), "bien", 1.0
        elif any(neg_word in user_input_clean for neg_word in palabras_mal):
            st.session_state["estado"] = "interaccion_normal"
            return random.choice(respuestas_mal), "mal", 1.0
    
    # Detectar la intención normalmente si no es un saludo
    st.session_state["estado"] = "interaccion_normal"
    top_intents, top_scores = intent_detection(user_input)
    for intent, score in zip(top_intents, top_scores):
        print(f"Detected intent: {intent} with score: {score}")  # Imprimir las intenciones y puntajes en la consola
    threshold = 0.9  # Umbral para la confianza en la intención
    if top_scores[0] < threshold:
        return 'No entiendo. ¿Puedes reformular la pregunta?', top_intents[0], top_scores[0]
    for intent_obj in intents['intents']:
        if intent_obj['tag'] == top_intents[0]:
            responses = intent_obj['responses']
            return random.choice(responses), top_intents[0], top_scores[0]
    return 'No entiendo. ¿Puedes reformular la pregunta?', top_intents[0], top_scores[0]


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
