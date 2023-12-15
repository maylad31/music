import streamlit as st
import librosa
import numpy as np
from key import Tonal_Fragment


def detect_key_and_bpm(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Estimate the tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    #chromagram = librosa.util.normalize(chromagram)

    # Calculate the mean chroma feature across time
    mean_chroma = np.mean(chromagram, axis=1)

    # Define the mapping of chroma features to keys
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Find the key by selecting the maximum chroma feature
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]


    
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    a=Tonal_Fragment(y_harmonic, sr, tstart=0, tend=120)
   

    return a.print_key(), estimated_key+" "+str(chroma_to_key), tempo



st.title('Key and tempo detection')

# File uploader for previous track
st.subheader('Upload a Track')
previous_track = st.file_uploader('Choose a mp3 file', type='mp3')
out_path=None
if previous_track is not None:
    with open('key.wav', 'wb') as f:
        a=previous_track.read()
        f.write(a)
    st.audio(a, format='audio/wav')
    out_path="key.wav"
if out_path is not None:
    key,key2,tempo=detect_key_and_bpm("key.wav")
    st.write("key: "+str(key))
    st.write("key using another method: "+str(key2))
    st.write("tempo: "+str(tempo))
 