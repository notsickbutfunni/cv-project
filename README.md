# Receipt & Handwritten Note Summarizer

A simple **Streamlit** web app that extracts text from receipts or handwritten notes using **OCR (Tesseract)** and generates concise summaries using a **Transformer-based model** (BART).

---

## Features

- Upload **receipt** or **handwritten note** images (`.jpg`, `.jpeg`, `.png`)
- Automatic **OCR text extraction** with `pytesseract`
- **Text summarization** using `facebook/bart-large-cnn`
- Preprocessing & cleaning to improve OCR accuracy
- Real-time web UI built with **Streamlit**

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| OCR Engine | Tesseract OCR (`pytesseract`) |
| NLP Model | Hugging Face Transformers (`facebook/bart-large-cnn`) |
| Image Processing | OpenCV |
| Language | Python 3.10+ |

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/receipt-summarizer.git
cd receipt-summarizer
```
2. **Install dependencies**
```bash
git clone https://github.com/yourusername/receipt-summarizer.git
cd receipt-summarizer
```
3. **Install Tesseract OCR**

Windows

- Download and install from Tesseract OCR GitHub

```bash
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
```
Linux (Ubuntu/Debian)
```bash
sudo apt install tesseract-ocr
```

macOS
```bash
brew install tesseract
```

4. ▶️ **Run the App**
```bash
streamlit run app.py
```
5. Open your browser at
     👉 http://localhost:8501

---

### 📂 Project Structure
```
receipt-summarizer/
│
├── app.py
├── requirements.txt
├── README.md
└── sample_images/
    ├── receipt1.jpg
    └── note1.jpg
```
![image alt](https://github.com/notsickbutfunni/cv-project/blob/main/homepage.png?raw=true)

