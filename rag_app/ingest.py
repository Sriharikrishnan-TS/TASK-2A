# ingest_documents.py

import os
import pdfplumber
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directory containing PDFs
PDF_DIR = r"E:\rag_app\TASK 2A DATASET"
VECTOR_DB_DIR = "./chroma_db"

def load_papers_from_pdf(directory):
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            path = os.path.join(directory, file)
            with pdfplumber.open(path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                #It's designed to hold a piece of content (text) along with associated metadata
                documents.append(Document(page_content=text, metadata={"source": file}))
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    split_docs = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            split_docs.append(chunk)
    return split_docs

def create_vectorstore(split_docs):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(split_docs, embedding=embedding_model, persist_directory=VECTOR_DB_DIR)
    db.persist()
    return db


#to check is the script is the main programme being executed right now
'''
this is used to ensure that code should only run when the script is executed directly
and should not run when the script is imported as a module
'''

#__name__ = built in variable created by python interpreter
#__main__  if the script runs directly,the "__name__" variable will be set to "__main__"

if __name__ == "__main__":
    print("Loading and parsing documents...")
    docs = load_papers_from_pdf(PDF_DIR)

    print("Splitting documents into chunks...")
    split_docs = split_documents(docs)

    # Preview first 5 chunks for sanity check
    print("\n--- Preview of first 5 chunks ---")
    for chunk in split_docs[:5]:
        source = chunk.metadata.get("source", "N/A")
        idx = chunk.metadata.get("chunk_index", -1)
        print(f"Source: {source} | Chunk index: {idx}")
        print(chunk.page_content[:300] + "...\n")

    print("Creating and saving vector store...")
    vectorstore = create_vectorstore(split_docs)
    print("Done.")
