from spleeter.separator import Separator
from multiprocessing import freeze_support
import os
import spleeter
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup
#from pydub.effects import normalize
#from pydub.effects import reverb
import streamlit as st



# Path to your audio file
def main():
    separator = Separator('spleeter:4stems')

    # Define output directory for separated stems
    output_directory = 'stems'

    # Initialize Spleeter separator
    
    st.subheader('Upload a Track to separate stems')
    previous_track = st.file_uploader('Choose a mp3 file', type='mp3')
    out_path=None
    if previous_track is not None:
        with open('prev.wav', 'wb') as f:
            a=previous_track.read()
            f.write(a)
        st.audio(a, format='audio/wav')
        separator.separate_to_file('prev.wav', output_directory)
        out_path="stems/prev"
    if out_path is not None:
        for key in ["other","drums","other","vocals"]:
            st.title(f'{key}')
            saved_audio_path = f"{out_path}/{key}.wav"  
            saved_audio_file = open(saved_audio_path, 'rb')
            audio_bytes = saved_audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
    
# def apply_effects():
#     vocal_stem_path = '/Users/mayankladdha/Desktop/ai_dj/stems/Qismat_1/vocals.wav'
#     instrumental_stem_path = '/Users/mayankladdha/Desktop/ai_dj/stems/Qismat_1/accompaniment.wav'

#     # Load vocal and instrumental stems using pydub
#     vocal_stem = AudioSegment.from_file(vocal_stem_path)
#     instrumental_stem = AudioSegment.from_file(instrumental_stem_path)

    
    
#     manipulated_instrumental_stem = instrumental_stem + 4  # Increase volume by 4 dB

    
#     # Apply fade-in and fade-out effects to instrumental stem and vocal stem
#     fade_duration = 100  # milliseconds for fade-in and fade-out
#     manipulated_instrumental_stem = manipulated_instrumental_stem.fade_in(fade_duration).fade_out(fade_duration)
#     vocal_stem = vocal_stem.fade_in(fade_duration).fade_out(fade_duration)

#     # Merge the manipulated instrumental stem with the vocal stem
#     combined_audio = vocal_stem.overlay(manipulated_instrumental_stem)

#     # Set the path to save the output combined audio
#     output_path = 'res.wav'

#     # Export the combined audio
#     combined_audio.export(output_path, format='wav')
    
if __name__ == '__main__':
    freeze_support()
    main()
    #apply_effects()