import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import unicodedata

nltk.download('punkt')
nltk.download('wordnet')

# Cargar modelos y datos del chatbot
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('dataset1.json', encoding='utf-8').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Verificar la estructura y contenido del archivo JSON
#print("Intenciones cargadas:")
#for intent in intents['intents']:
    #print(f"Tag: {intent['tag']}, Patterns: {intent['patterns']}, Responses: {intent['responses']}")

# Descargar stopwords en español
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('spanish'))

# Definir umbral de confianza
umbral_confianza = 0.8

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Funciones del chatbot
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words if word.lower() not in stop_words and word.isalpha()]
    # Eliminar tildes de las palabras
    sentence_words = [unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore').decode('utf-8') for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print("Resultados de predicción:", return_list)  # Depuración
    return return_list

def get_response(ints):
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    print(f"Buscando respuestas para la intención: {tag}")  # Depuración

    # Imprimir todas las intenciones para verificar
    #for i in list_of_intents:
        #print(f"Intención disponible: {i['tag']}")  # Depuración

    # Iterar sobre las intenciones
    for i in list_of_intents:
        if i['tag'].strip().lower() == tag.strip().lower():
            #print(f"Intención encontrada: {i}")  # Depuración
            return random.choice(i['responses'])
    
    # Si no se encuentra la intención
    print("Intención no encontrada")  # Depuración
    return "Lo siento, no tengo una respuesta para eso."

load_css()

# Interfaz de Streamlit
st.title("IntelliGen Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True    

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("image", None)):
        st.markdown(message["content"])

if st.session_state.first_message:
    with st.chat_message("assistant", avatar="static/images/chatbot.png"):
        st.markdown("Hola, ¿cómo puedo ayudarte?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?", "image": "static/images/chatbot.png"})
    st.session_state.first_message = False

if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    with st.chat_message("user", avatar="static/images/gente.png"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "image": "static/images/gente.png"})     

    intents_list = predict_class(prompt)
    print("Intenciones detectadas:", intents_list)  # Imprimir intenciones en la consola

    if intents_list:
        if float(intents_list[0]['probability']) < umbral_confianza:
            with st.chat_message("assistant", avatar="static/images/chatbot.png"):   
                st.markdown("Lo siento, no entendí tu pregunta. ¿Podrías reformularla?")
            st.session_state.messages.append({"role": "assistant", "content": "Lo siento, no entendí tu pregunta. ¿Podrías reformularla?", "image": "static/images/chatbot.png"})
        else:
            response = get_response(intents_list)
            with st.chat_message("assistant", avatar="static/images/chatbot.png"):   
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response, "image": "static/images/chatbot.png"})
    else:
        with st.chat_message("assistant", avatar="static/images/chatbot.png"):   
            st.markdown("Lo siento, no pude detectar ninguna intención en tu pregunta.")
        st.session_state.messages.append({"role": "assistant", "content": "Lo siento, no pude detectar ninguna intención en tu pregunta.", "image": "static/images/chatbot.png"})


# Botón de limpiar
if st.button("Limpiar conversación"):
    st.session_state.messages = []
    st.session_state.first_message = True
    st.experimental_rerun()  # Reiniciar la aplicación
