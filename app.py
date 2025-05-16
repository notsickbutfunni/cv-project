import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

def preprocess_image(image_data):
    image = Image.open(image_data).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return thresh


def extract_text(image_np):
    image_pil = Image.fromarray(image_np)
    return pytesseract.image_to_string(image_pil)

def summarizer(text):
    response  = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this receipt or handwritten note."},
            {"role": "user", "content": text}
        ]
    )

    return response['choices'][0]['message']['content'] 

st.title("Receipt & Handwritten Note Summarizer")

uploaded_file = st.file_uploader("Upload receipt/note image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded file', use_column_width=True)
    preprocessed_image = preprocess_image(uploaded_file)
    st.image(uploaded_file, caption='Preprocessed', use_column_width=True)

    with st.spinner("Running OCR..."):
        extracted_text = extract_text(preprocessed_image)
        st.subheader("Extracted text: ")
        st.text_area("OCR Output", extracted_text, height=200)

    with st.spinner("Summarizing..."):
        summary = summarizer(extract_text)
        st.subheader('Summary: ')
        st.write(summary)