import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
from PyPDF2.errors import PdfReadError
import chardet

# Load environment variables
load_dotenv('D:\\RAG query searcher\\apikey.env')

# API key configuration
key = os.getenv('GOOGLE_API_KEY')  # Ensure this is set in .env
if key:
    genai.configure(api_key=key)
else:
    raise ValueError("Google Gemini API key is missing. Please set GOOGLE_API_KEY in your .env file.")

def detect_encoding(file):
    """Detect file encoding using chardet."""
    raw_data = file.read()
    file.seek(0)  # Reset file pointer after reading
    result = chardet.detect(raw_data)
    return result['encoding']

def pdf_read(pdfs):
    """Read and extract text from PDF files."""
    text = ""
    for pdf in pdfs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text
                    else:
                        st.warning(f"No text extracted from page {page.page_number}")
        except Exception as e:
            st.error(f"Error reading PDF file {pdf.name}: {str(e)}")
    st.write(text[:1000])  # Display the first 1000 characters of extracted text for debugging
    return text

def read_csv(file):
    """Read and process CSV files with encoding fallback."""
    try:
        # Detect encoding
        encoding = detect_encoding(file)
        df = pd.read_csv(file, encoding=encoding)
        return df.to_string()
    except UnicodeDecodeError:
        st.error(f"Encoding issue detected in {file.name}. Attempting fallback encoding...")
        try:
            df = pd.read_csv(file, encoding='latin-1')
            return df.to_string()
        except Exception as e:
            st.error(f"Failed to process file {file.name}: {str(e)}")
            return ""
    except Exception as e:
        st.error(f"Error reading CSV file {file.name}: {str(e)}")
        return ""

def chunk_split(text):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    st.write(chunks[:5])  # Print first few chunks for debugging
    return chunks

def generate_embeddings(chunks):
    """Generate embeddings from text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        store = FAISS.from_texts(chunks, embedding=embeddings)
        store.save_local("faiss.index")
        st.info("Embeddings generated and saved successfully.")
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")

def get_chain():
    """Set up the question-answering chain."""
    prompt_template = '''
    Provide a detailed reply to the given question based on the given context. Your answer will only be based on the provided context and nothing else. If the answer to the question is not present in the context your reply should be 'Answer not in the provided context',\n\n
    Context : {context}\n\n
    Question: {question}
    '''
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    prompt = PromptTemplate(input_variables=['context', 'question'], template=prompt_template)
    return load_qa_chain(model, chain_type='stuff', prompt=prompt)

def user_input(query):
    """Process user query and return a response."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        data = FAISS.load_local("faiss.index", embeddings, allow_dangerous_deserialization=True)
        st.write(f"Performing similarity search for query: {query}")
        f_data = data.similarity_search(query)
        st.write(f"Relevant documents found: {f_data}")  # Display search results for debugging
        chain = get_chain()
        response = chain({'input_documents': f_data, 'question': query}, return_only_outputs=True)
        st.write("### Response:")
        st.write(response['output_text'])
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def process_file(files):
    """Process uploaded files (PDF or CSV)."""
    for file in files:
        if file.name.endswith('.pdf'):
            text = pdf_read([file])
        elif file.name.endswith('.csv'):
            text = read_csv(file)
        else:
            st.error(f"Unsupported file type: {file.name}")
            continue

        if text.strip():
            chunks = chunk_split(text)
            generate_embeddings(chunks)

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(page_title='PDF Reader', layout='wide')
    st.header('Generate Answers from PDFs Using Gemini')

    query = st.text_input("Ask your question")
    if query:
        user_input(query)

    with st.sidebar:
        st.title('Upload Files')
        files = st.file_uploader("Upload PDFs or CSVs", accept_multiple_files=True, type=['pdf', 'csv'])

        if st.button('Submit'):
            if not files:
                st.error("Please upload at least one file!")
            else:
                with st.spinner('Processing files...'):
                    process_file(files)
                    st.success("All files processed successfully!")

if __name__ == "__main__":
    main()
