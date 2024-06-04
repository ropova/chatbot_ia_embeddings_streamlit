import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import random
import re
import unicodedata
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import time

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Función para incluir el script de JavaScript para mantener la aplicación activa
def load_keep_alive_script():
    keep_alive_script = """
    <script>
        function keepAlive() {
            fetch('/')
            setTimeout(keepAlive, 600000);  // 10 minutos en milisegundos
        }
        keepAlive();
    </script>
    """
    st.markdown(keep_alive_script, unsafe_allow_html=True)        

# Descargar los recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('wordnet')

# Cargar el modelo y otros datos necesarios
model = load_model('magnetica_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with open('tokenizer_mag.json', 'r') as f:
    tokenizer_data = json.load(f)
    tokenizer = Tokenizer()
    tokenizer.word_index = tokenizer_data

with open('label_encoder_mag.json', 'r') as f:
    label_encoder_data = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_data)

with open('word_index_mag.json', 'r') as f:
    word_index = json.load(f)
embedding_matrix = np.load('embedding_matrix.npy')

max_length = 20

# Inicializar el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar intents
with open('data_mag.json', encoding='utf-8') as f:
    intents = json.load(f)

# Listas adicionales de respuestas y palabras
palabras_bien = ["bien", "super", "lo maximo", "excelente", "bendecido", "genial", "fantastico", "maravilloso", "increible", "estupendo", "fenomenal"]
palabras_mal = ["mal", "regular", "no me siento bien", "no estoy bien", "desanimado", "decaido", "apagado", "mas o menos", "bajo de nota", "triste", "abatido", "preocupado", "desalentado", "cansado", "agobiado"]

respuestas_saludo = ["Hola, ¿cómo estás?", "¡Hola! ¿Cómo te va?", "Saludos, ¿cómo te encuentras?", "Hola, ¿cómo has estado?", "¿Qué tal estás hoy?", "¿Cómo te sientes?", "Hola, ¿cómo va todo?", "¿Cómo te encuentras?", "¿Cómo va tu día?", "Hola, ¿cómo te sientes hoy?", "¿Cómo andas?"]

respuestas_bien = ["Me alegra escuchar eso. ¿En qué puedo ayudarte?", "¡Genial! ¿Cómo puedo asistirte?", "¡Qué bueno! ¿En qué te puedo ayudar?", "Eso suena fantástico. ¿Hay algo específico en lo que pueda colaborar?", "Qué bueno saberlo. ¿Necesitas algún tipo de ayuda o apoyo adicional?", "Me alegra escuchar eso. ¿Hay algo en lo que pueda contribuir para que tu día sea aún mejor?", "Maravilloso. ¿Hay algo en lo que necesites ayuda o asistencia?", "Me alegra mucho escuchar eso. ¿Cómo puedo ayudarte hoy?", "Excelente. ¿Hay algo en lo que necesites que te ayude o apoye?", "Me alegra saberlo. ¿En qué puedo colaborar para que sigas sintiéndote así de bien?"]

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

# Función para normalizar texto eliminando puntuación y espacios innecesarios
def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def remove_accents_and_symbols(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

# Función para tokenizar y lematizar texto utilizando NLTK
def tokenize_and_lemmatize(text):
    text = normalize_text(text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return ' '.join(lemmatized_tokens)

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
    # Imprimir la entrada del usuario en la consola
    print(f"User input: {user_input}")

    user_input_lower = user_input.lower()
    user_input_clean = remove_accents_and_symbols(user_input_lower)

    # Procesar la entrada del usuario como de costumbre
    if "hola" in user_input_clean:
        print("Detected salutation.")
        response = random.choice(respuestas_saludo)
        print(f"Response: {response}")
        return response, "saludo", 1.0

    # Buscar la palabra "no" como una palabra independiente
    if "no" == user_input_clean.strip():
        print("Detected negative response.")
        response = "Espero verte pronto."
        print(f"Response: {response}")
        return response, "respuesta_no", 1.0

    # Buscar la palabra "si" como una palabra independiente
    if "si" == user_input_clean.strip() or "sí" == user_input_clean.strip():
        print("Detected affirmative response.")
        response = "¿En qué puedo ayudarte?"
        print(f"Response: {response}")
        return response, "respuesta_si", 1.0

    # Procesar palabras clave de estado emocional
    if any(palabra in user_input_clean for palabra in palabras_bien):
        print("Positive emotional state keywords detected.")
        response = random.choice(respuestas_bien)
        print(f"Response: {response}")
        return response, "bien", 1.0
    elif any(palabra in user_input_clean for palabra in palabras_mal):
        print("Negative emotional state keywords detected.")
        response = random.choice(respuestas_mal)
        print(f"Response: {response}")
        return response, "mal", 1.0

    # Detectar la intención normalmente
    intent, score = intent_detection(user_input)
    print(f"Detected intent: {intent} with score: {score}")  # Imprimir las intenciones y puntajes en la consola
    threshold = 0.9  # Umbral para la confianza en la intención
    if score < threshold:
        print('No entiendo. ¿Puedes reformular la pregunta?')  # Imprimir en la consola si no se entiende la pregunta
        return 'No entiendo. ¿Puedes reformular la pregunta?', intent, score
    for intent_obj in intents['intents']:
        if intent_obj['tag'] == intent:
            responses = intent_obj['responses']
            response = random.choice(responses)
            print(f"Response: {response}")
            return response, intent, score

def display_text_word_by_word(text):
    message_placeholder = st.empty()
    words = text.split()
    for i, word in enumerate(words):
        if i != len(words) - 1:
            message_placeholder.markdown(" ".join(words[:i]) + " " + word + " ▌", unsafe_allow_html=True)
        else:
            message_placeholder.markdown(" ".join(words[:i]) + " " + word + " ▌", unsafe_allow_html=True)  # Mostrar el marcador en la última palabra
        time.sleep(random.uniform(0.08, 0.2))  # Generar una pausa aleatoria entre 0.2 y 0.5 segundos por palabra

    # Eliminar el marcador después de 1 segundo
    time.sleep(0.1)
    message_placeholder.markdown(" ".join(words), unsafe_allow_html=True)


load_css()
load_keep_alive_script()  # Cargar el script para mantener la aplicación activa

# Interfaz de Streamlit
#st.title("IntelliGen Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "estado" not in st.session_state:
    st.session_state.estado = "inicial"

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("image", None)):
         st.markdown(message["content"], unsafe_allow_html=True)  # Asegurar que el contenido HTML se muestre correctamente

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="static/images/chatbot.png"):
        st.markdown("Hola, soy Magnétic Bot ¿cómo puedo ayudarte?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, soy Magnétic Bot ¿cómo puedo ayudarte?", "image": "static/images/chatbot.png"})

if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    with st.chat_message("user", avatar="static/images/gente.png"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "image": "static/images/gente.png"})     

    with st.spinner(""):
        response, intent, score = respond_to_user(prompt, intents)
        st.session_state.spinner = False  # Desactivar el spinner una vez que se completa el procesamiento

    with st.chat_message("assistant", avatar="static/images/chatbot.png"):
        display_text_word_by_word(response)
        st.session_state.messages.append({"role": "assistant", "content": response, "image": "static/images/chatbot.png"})


# Botón de limpiar
if st.button("Limpiar conversación"):
    st.session_state.messages = []
    st.session_state.estado = "inicial"
    st.experimental_rerun()  # Reiniciar la aplicación
