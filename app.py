import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
# lOAD ENVIRONMENT VARIABLES
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
# for reading and extracting text from pdf files
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
            return text
        
# for splitting text into smaller chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

# for creating embeddings and loading the vector store
def get_vector_store(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(chunks,embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = (
    "Context:\n{context}\n\n"
    "Answer the question as detailed as possible from the provided context. "
    "If the answer is not in the context, say 'answer is not available in the context' and don't guess.\n"
    "Question: {question}"
)
                         
    
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    prompt=PromptTemplate(template=prompt_template, input_variables=["question"])
    chain=load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# for answering questions based on the context

def user_input(user_question):
    emmbeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index", emmbeddings, allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question, k=3)
    chain=get_conversational_chain()
    response=chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Answer: ", response["output_text"])

# main function to run the streamlit app
def main():
    st.title("PDF Chatbot")
    st.write("Upload your PDF files and ask questions about the content.")
    
    pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Process PDFs"):
        if pdf_docs:
            text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(text)
            get_vector_store(chunks)
            st.success("PDFs processed successfully!")
        else:
            st.error("Please upload at least one PDF file.")
    
    user_question = st.text_input("Ask a question about the content:")
    
    if st.button("Get Answer"):
        if user_question:
            user_input(user_question)
        else:
            st.error("Please enter a question.")
if __name__ == "__main__":
    main()


    



        


