# Rag_from_scratch
This repository contains a vanilla implementation of Retrieval-Augmented Generation (RAG) for document search and question answering. 

## Project Structure

- **chunked_text.txt**: Contains the text split into chunks for processing.
- **embeddings.csv**: Stores the embeddings generated from the text chunks.
- **preprocessed_pdf.txt**: Contains the preprocessed text extracted from the PDF.
- **get_embeddings.py**: Script to generate embeddings for the text chunks.
- **hello.py**: Script to perform search and question answering using the embeddings.
- **splitting.py**: Script to split the preprocessed text into chunks.
- **requirements.txt**: List of required packages for the project.
- **hellotest.py**: to run the app in streamlit.


## Setup Instructions
add your Gemini API key in hello.py file



replace **pdf_url** in splitting.py file  line(35) to your pdf and put your openai api in search_and_answer.py file


