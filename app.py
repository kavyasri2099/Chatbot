import streamlit as st
import speech_recognition as sr
import pyttsx3
import cv2
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import numpy as np
from PIL import Image

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize Text-to-Speech
engine = pyttsx3.init()

def recognize_speech():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Could not understand audio")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")
    return None

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Streamlit Interface
st.title("Speech-to-Speech LLM Bot with Video")

# Upload a video file or use a webcam
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    st.video(uploaded_file)

if st.button("Start"):
    user_text = recognize_speech()
    if user_text:
        response_text = generate_response(user_text)
        st.write(f"Response: {response_text}")
        speak_text(response_text)
