import os
import base64
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
import numpy as np
import torchaudio

# Initialize Float feature
float_init()

# Load Wav2Vec2 model and tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Load GPT-2 model and tokenizer
gpt_model_name = "gpt2"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)

def get_answer(prompt):
    inputs = gpt_tokenizer(prompt, return_tensors="pt")
    outputs = gpt_model.generate(**inputs)
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

def speech_to_text(audio_file_path):
    # Load audio file
    waveform, _ = torchaudio.load(audio_file_path)
    inputs = tokenizer(waveform.squeeze().numpy(), return_tensors="pt")
    with torch.no_grad():
        logits = wav2vec2_model(input_values=inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return tokenizer.batch_decode(predicted_ids)[0]

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
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages[-1]["content"])
        with st.spinner("Generating audio response..."):
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

footer_container.float("bottom: 0rem;")
