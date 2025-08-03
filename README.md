# PDF Chatbot with Streamlit, LangChain, and Gemini

This project is a Streamlit web app that allows you to upload PDF files and ask questions about their content using Google Gemini and LangChain. The app processes your PDFs, creates embeddings, and answers your questions based on the content.

---

## Features
- Upload multiple PDF files
- Extracts and chunks text from PDFs
- Embeds text using Google Gemini embeddings
- Stores and retrieves embeddings with FAISS
- Answers questions using Gemini LLM via LangChain

---

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd pdf_chat
```

### 2. Create and Activate a Python Environment (Recommended)
```sh
conda create -n pdfchat python=3.10 -y
conda activate pdfchat
# OR use venv:
# python -m venv venv
# source venv/bin/activate
```

### 3. Install Required Packages
Install all dependencies:
```sh
pip install streamlit langchain langchain-community langchain-google-genai google-generativeai python-dotenv PyPDF2
```

### 4. Set Up Google Gemini API Key
- Get your Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 5. Run the App
Always use Streamlit to run the app:
```sh
streamlit run app.py
```

---

## Usage
1. **Upload PDFs:** Click "Upload PDF files" and select one or more PDF files.
2. **Process PDFs:** Click the "Process PDFs" button. Wait for the success message.
3. **Ask Questions:** Enter your question in the text box and click "Get Answer". The answer will be generated from the content of your uploaded PDFs.

---

## Troubleshooting
- **ModuleNotFoundError:** Ensure all required packages are installed in your active environment.
- **FAISS index error:** Always process PDFs before asking questions. If you see a missing index error, re-upload and process your PDFs.
- **GoogleGenerativeAIError:** Make sure your API key is correct and you are using the right model for embeddings (`models/embedding-001`).
- **Deserialization Warning:** The app uses `allow_dangerous_deserialization=True` when loading the FAISS index. This is safe as long as you trust your own generated index files.
- **Streamlit Warnings:** Always run the app with `streamlit run app.py`, not `python app.py`.

---

## Notes
- This app is for educational/demo purposes. Do not use with sensitive data.
- Only use your own `.env` and FAISS index files.
- For best results, use clear, well-formatted PDFs.

---

## License
MIT License
