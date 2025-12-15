import streamlit as st
import torch
import timm
import librosa
import numpy as np
import json

SR = 32000
N_MELS = 128
N_FFT = 2048
HOP = 512
FMIN = 800
FMAX = 12000

SLICE_SECONDS = 5.0
SLICE_FRAMES = int(SLICE_SECONDS * SR / HOP)
HOP_FRAMES = SLICE_FRAMES // 2

DEVICE = "cpu"

with open("labels.json", "r") as f:
    idx2label = json.load(f)
idx2label = {int(k): v for k, v in idx2label.items()}
NUM_CLASSES = len(idx2label)


@st.cache_resource
def load_model():
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        in_chans=1,
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(
        torch.load("model/efficientnet_b3_best.pt", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def audio_to_mel(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-6)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP,
        fmin=FMIN, fmax=FMAX
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80.0) / 80.0
    mel = np.clip(mel, -1.0, 1.0)

    return mel.astype(np.float32)

def predict_species(audio_path):
    mel = audio_to_mel(audio_path)

    slices = []
    for s in range(0, max(1, mel.shape[1] - SLICE_FRAMES + 1), HOP_FRAMES):
        clip = mel[:, s:s + SLICE_FRAMES]
        if clip.shape[1] < SLICE_FRAMES:
            clip = np.pad(clip, ((0,0),(0,SLICE_FRAMES-clip.shape[1])))
        slices.append(clip)

    x = torch.tensor(np.stack(slices)).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits.mean(dim=0), dim=0)

    topk = torch.topk(probs, k=min(5, NUM_CLASSES))
    return [(idx2label[i.item()], p.item()) for i, p in zip(topk.indices, topk.values)]

st.title("Bird Species Classifier")
st.write("Upload an audio recording (wav / mp3).")

uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3", "ogg"])

if uploaded_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    with st.spinner("Predicting species..."):
        preds = predict_species("temp_audio.wav")

    st.subheader("Top predictions")
    for name, prob in preds:
        st.write(f"**{name}** — {prob:.2%}")
