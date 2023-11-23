import streamlit as st
import torch
from diffusion_model import SimpleUnet, get_image
from utils import embed_audio
import urllib.request

model_url = 'https://github.com/Smulemun/music-to-image/releases/download/model/diffusion_model_100e.pth'
model_path = 'models/diffusion_model_100.pth'

urllib.request.urlretrieve(model_url, model_path)

model = SimpleUnet()
model.load_state_dict(torch.load(model_path))

st.title('Music to Image Generator')

uploaded_file = st.file_uploader('Choose an audio file (currently only wav is supported)', type=['wav'])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write('Generating images...')
    audio = uploaded_file.getvalue()

    audio_path = 'tmp.wav'
    with open(audio_path, 'wb') as f:
        f.write(audio)
    audio_embedding = embed_audio(audio_path)

    columns = st.columns([1, 1], gap='medium')

    for i in range(10):
        image = get_image(model, audio_embedding)
        with columns[i % len(columns)]:
            st.image(image, caption=f'Generated image {i+1}', width=256)
    