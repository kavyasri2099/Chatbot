import os
import base64
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import speech_recognition as sr
from pyttsx3 import init
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init

# Initialize Float feature
float_init()

# Load LLM model and tokenizer
model_name = "t5-base"  # Use t5-base for demonstration
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

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
    engine = init()
    engine.say(text)
    engine.runAndWait()

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

st.title("Speech-to-Speech Conversational Bot ")

footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.wav"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_response = get_answer(st.session_state.messages[-1]["content"])
        with st.spinner("Generating audio response..."):    
            text_to_speech(final_response)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

footer_container.float("bottom: 0rem;")
