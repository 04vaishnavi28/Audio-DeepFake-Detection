import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from glob import glob
import io
import librosa
import plotly.express as px 
import torch
import torch.nn.functional as F 
import torchaudio
import numpy as np 
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import webbrowser

#function to load the audio file
def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str): #if the input is the file path
        if audiopath.endswith('.wav'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, f"Unsupported audio format provided: {audiopath[-4:]}"
    elif isinstance(audiopath, io.BytesIO):  #if the input is file content
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0]  #remove any channel data

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1,1) 

    return audio.unsqueeze(0)

#function for classifier
def classify_audio_clip(clip):





    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():

    st.title("AI-Generated Audio Detection")

    if st.button("Insights"):
        webbrowser.open_new_tab("index.html")

        
    #file uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        if st.button("Analyze Audio"):
            st.info("YOUR RESULTS ARE BELOW")
            col1, col2 = st.columns(2)  # col3 to add disclaimer and col(3) to add the disclamier content col3
            with col1:
                
                #load and classify and audio file
                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip)
                result = result.item()
                st.info(f"Result Probability: {result}")
                if result >= .01:
                    st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI Generated.")
                else:
                    st.success(f"The uploaded audio is {(100 - (result)):.2f}% likely to be Real.")


            with col2:
                st.info("Your uploaded audio is below")
                st.audio(uploaded_file)
                
            

            # with col3:
            #     st.info("Disclaimer")
            #     st.warning("These classification or detection mechanisms are not always arrurate. They should be considered as a strong signal and not the ultimate decision makers.")
                


            #create a waveform
            fig = px.line()
            fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
            fig.update_layout(
                title="Waveform Plot",
                xaxis_title = "Time",
                yaxis_title = "Amplitude"
            )
            st.plotly_chart(fig, use_container_width=True) 


            # #create spectrogram
            # plt.figure(figsize=(13, 4))
            # plt.specgram(audio_clip.squeeze().numpy(), Fs=22000, cmap='viridis')
            # plt.xlabel('Time')
            # plt.ylabel('Frequency')
            # plt.title('Spectrogram')
            # st.pyplot()



            # # Create LFCC features
            # def compute_lfcc(audio_clip, sampling_rate=22000, n_mfcc=13):
            #     # Compute LFCC features using torchaudio
            #     mfcc_transform = torchaudio.transforms.MFCC(
            #         sample_rate=sampling_rate,
            #         n_mfcc=n_mfcc,
            #         melkwargs={'n_fft': 400, 'n_mels': 40, 'hop_length': 160, 'center': False}
            #     )
            #     lfcc_features = mfcc_transform(audio_clip)

            #     # Convert to numpy array and transpose
            #     lfcc_features = lfcc_features.squeeze().numpy().T
            #     return lfcc_features

            # lfcc_features = compute_lfcc(audio_clip)

            # # Plot LFCC features
            # plt.figure(figsize=(20, 5))
            # plt.imshow(lfcc_features, aspect='auto', origin='lower', cmap='viridis')
            # plt.colorbar()
            # plt.xlabel('Frame')
            # plt.ylabel('LFCC Coefficient')
            # plt.title('LFCC Features')
            # st.pyplot()

            # # Create MFCC visualization
            # audio_data, _ = librosa.load(uploaded_file, sr=22000)  # Load audio data
            # mfccs = librosa.feature.mfcc(y=audio_data, sr=22000, n_mfcc=13)  # Extract MFCC features
            # fig, ax = plt.subplots()
            # img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
            # fig.colorbar(img, ax=ax)
            # ax.set(title='MFCC')
            # st.pyplot(fig)


if __name__ == "__main__":
    main()