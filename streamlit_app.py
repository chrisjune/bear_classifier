
from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

import streamlit as st

## 로딩
path = Path()
path.ls(file_exts = '.pkl')
learn_inf = load_learner(path/'export.pkl')


def load_image(image_file):
	img = Image.open(image_file)
	return img

st.title("어떤 곰인지 맞춰드립니다")
st.image("example1.jpg", width=100)
st.image("example2.jpg", width=100)
st.image("example3.jpg", width=100)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img_file = load_image(uploaded_file)
    st.image(img_file, width=224)

    pred, pred_idx, probs = learn_inf.predict(PILImage.create(uploaded_file))
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")
