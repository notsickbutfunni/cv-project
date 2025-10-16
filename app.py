import streamlit as st
import cv2
import re
from PIL import Image
import numpy as np
import pytesseract
from transformers import pipeline

def preprocess_image(image_data):
    image = Image.open(image_data).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3) #denoise

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return thresh


def extract_text(image_np):
    custom_config = r'--oem 3 --psm 6'
    image_pil = Image.fromarray(image_np)
    text = pytesseract.image_to_string(image_pil, config=custom_config)
    return text

summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarizer(text):
    text = text[:2000]  
    summary = summarizer_model(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\%\$]', ' ', text)  # remove weird chars
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


st.title("Receipt & Handwritten Note Summarizer")

uploaded_file = st.file_uploader("Upload receipt/note image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded file', use_container_width =True)
    preprocessed_image = preprocess_image(uploaded_file)
    st.image(preprocessed_image, caption='Preprocessed', use_container_width =True)

    with st.spinner("Running OCR..."):
        extracted_text = extract_text(preprocessed_image)
        cleaned_text = clean_text(extracted_text)
        st.subheader("Extracted text: ")
        st.text_area("OCR Output", cleaned_text, height=200)


    if len(extracted_text.strip()) < 50:
        st.warning("⚠️ Could not detect enough readable text. Try uploading a clearer image.")
    else:
        with st.spinner("Summarizing..."):
            st.write("Extracted text preview:")
            st.write(extracted_text[:500])
        
            summary = summarizer(extracted_text)  
            st.subheader('Summary: ')
            st.write(summary)
