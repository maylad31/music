
import streamlit as st
import librosa
import numpy as np
import librosa
import numpy as np

def extract_features(audio, sr):
    # Calculate various features - spectral centroid, tempo, beats, onsets
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onsets = librosa.frames_to_time(onset_frames, sr=sr)
    return spectral_centroid, tempo, onsets

def find_cue_points(audio1, audio2):
    cue_points = []
    window_size = 2048  # Adjust window size for longer or shorter transitions
    hop_length = 512  # Adjust hop length for smoother analysis

    # Load audio files and ensure they have the same sample rate
    y1, sr1 = librosa.load(audio1, sr=None)
    y2, sr2 = librosa.load(audio2, sr=None)

    # Resample to a common sample rate if different
    if sr1 != sr2:
        y2 = librosa.resample(y2, sr2, sr1)
        sr = sr1
    else:
        sr = sr1

    # Extract features for both tracks
    features1 = extract_features(y1, sr=sr)
    features2 = extract_features(y2, sr=sr)

    # Ensure the spectral centroid features have the same length
    min_len = min(len(features1[0]), len(features2[0]))
    spectral_centroid1 = features1[0][:min_len]
    spectral_centroid2 = features2[0][:min_len]

    # Calculate differences in spectral centroids between tracks
    diff_centroids = abs(spectral_centroid1 - spectral_centroid2)

    # Use tempo and beat information along with spectral centroid differences for cue point selection
    threshold = np.percentile(diff_centroids, 80)  # Adjust percentile as needed
    for i, val in enumerate(diff_centroids):
        if val > threshold:
            time_sec = librosa.frames_to_time(i * hop_length, sr=sr)
            # Consider tempo and beat information for cue point selection
            if time_sec > 5 and time_sec < (len(y1) / sr1 - 5):  # Skip initial and final 5 seconds
                # Allowing some flexibility by considering tempo difference within a range
                tempo_diff = abs(features1[1] - features2[1])
                if tempo_diff < 15:
                    # Check if the time is close to an onset in both tracks
                    if any(abs(time_sec - onset) < 0.5 for onset in features1[2]) and any(abs(time_sec - onset) < 0.5 for onset in features2[2]):
                        #time_sec=round(time_sec,-1)
                        cue_points.append((time_sec, val))  # Store time and difference as tuples

    # Select the top 10 cue points based on the magnitude of differences
    cue_points.sort(key=lambda x: x[1], reverse=True)
    cue_points = cue_points[:10]  # Select the top 10 points
    
    # Extract only the time values from the tuples
    cue_points = [point[0] for point in cue_points]
    
    return sorted(list(set(cue_points)))


st.subheader('Upload previous track')
prev = st.file_uploader('Choose mp3 file(prev track)', type='mp3')
out_path=False
if prev is not None:
    with open('previous.wav', 'wb') as f:
        a=prev.read()
        f.write(a)
    st.audio(a, format='audio/wav')
  
st.subheader('Upload next track')  
next= st.file_uploader('Choose a mp3 file(next track)', type='mp3')


if next is not None:
    with open('next.wav', 'wb') as f:
        a=next.read()
        f.write(a)
    st.audio(a, format='audio/wav')
    out_path=True

if out_path:
    cue_points = find_cue_points("previous.wav", "next.wav")
    
        
        
    st.write(cue_points)
    
