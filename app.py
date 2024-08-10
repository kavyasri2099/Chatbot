import streamlit as st
import torch
import pyttsx3
import speech_recognition as sr
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import time

# Authenticate with Hugging Face
login('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', add_to_git_credential=True)

# Load models
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

tts_engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Define functions
def recognize_speech_microphone():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info("🎤 Listening...")
        audio = recognizer.listen(source, timeout=5)
        audio_data = audio.get_wav_data()
        inputs = whisper_processor(audio_data, return_tensors="pt", sampling_rate=16000)
        outputs = whisper_model.generate(inputs["input_ids"])
        text = whisper_processor.decode(outputs[0], skip_special_tokens=True)
        return text

def generate_response(input_text):
    inputs = llama_tokenizer(input_text, return_tensors="pt")
    outputs = llama_model.generate(inputs["input_ids"])
    response_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Main function
def main():
    st.title("🗣️ Speech-to-Speech Bot")
    st.markdown("## Simple and Intuitive Interface")

    input_method = st.sidebar.radio("Choose input method:", ("Microphone 🎤", "Webcam 📹", "Text Input 📝"))

    if input_method == "Microphone 🎤":
        if st.button("Start Recording"):
            start_time = time.time()
            spoken_text = recognize_speech_microphone()
            if spoken_text:
                response_text = generate_response(spoken_text)
                speak_text(response_text)
                st.success(f"💬 Recognized: {spoken_text}")
                st.success(f"🤖 Response: {response_text}")
            end_time = time.time()
            processing_time = end_time - start_time
            st.info(f"Processing Time: {processing_time:.2f} seconds")

    elif input_method == "Webcam 📹":
        st.warning("Webcam input not fully implemented yet.")
        # Placeholder for webcam functionality

    elif input_method == "Text Input 📝":
        user_input = st.text_area("Type your text below:")
        if st.button("Generate Response"):
            start_time = time.time()
            response_text = generate_response(user_input)
            speak_text(response_text)
            st.success(f"🤖 Response: {response_text}")
            end_time = time.time()
            processing_time = end_time - start_time
            st.info(f"Processing Time: {processing_time:.2f} seconds")

    st.sidebar.markdown("Developed with ❤️ by [Your Name]")

if __name__ == "__main__":
    main()
