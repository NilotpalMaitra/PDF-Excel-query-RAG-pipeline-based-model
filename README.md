# PDF and CSV Query System Using Gemini AI

This application allows users to upload PDF and CSV files, process them, and generate answers to queries based on the content of the files using Gemini AI.

## Features

- Upload and process multiple PDF and CSV files.
- Extract text from PDF files.
- Read and display content from CSV files.
- Split extracted text into manageable chunks.
- Generate embeddings for the text chunks using Gemini AI.
- Perform query-based searches on the processed content.

## Requirements

- Python 3.7 or higher
- Streamlit
- pandas
- PyPDF2
- pdfplumber
- langchain
- google-generativeai
- python-dotenv

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/pdf-csv-query-system.git
   cd pdf-csv-query-system

2. Create a virtual environment and activate it:

python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

3. Install the required packages:
   
 pip install -r requirements.txt
4. Set up the environment variables:

Create a .env file in the root directory.
Add your Google Gemini API key to the .env file:

GOOGLE_API_KEY=your_google_gemini_api_key

Usage
Run the Streamlit application:
streamlit run [app.py](http://_vscodecontentref_/1)

Open your web browser and go to http://localhost:8501.

Upload PDF and CSV files using the sidebar.

Click the "Submit" button to process the files.

Enter your query in the text input box and get answers based on the content of the uploaded files.



(Summarized version)
Create a file named .env and paste your Gemini API Key as:

GOOGLE_API_KEY = {your api key}

Command for installing required dependencies on you virtual environment:

pip install -r requirements.txt

Command for running your streamlit application:

streamlit run app.py
