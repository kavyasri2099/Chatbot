import os
import base64
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
from transformers import DistilGPT2LMHeadModel, DistilGPT2Tokenizer
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init

# Initialize Float feature
float_init()

# Load LLM model and tokenizer
model_name = "distilgpt2"  # Use DistilGPT-2
tokenizer = DistilGPT2Tokenizer.from_pretrained(model_name)
model = DistilGPT2LMHeadModel.from_pretrained(model_name)

def get_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)
    return transcript

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    tts.save('temp_audio.mp3')
    return 'temp_audio.mp3'

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How may I assist you today?"}
        ]

initialize_session_state()

st.title("Speech-to-Speech Conversational Bot 🤖")

footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()

for message in st.session_state.messages:
    with st.chat_message(message["role"]
