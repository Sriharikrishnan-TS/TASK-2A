import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

VECTOR_DB_DIR = "./chroma_db"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#to load an existing ChromaDB instance
def load_vectorstore(_embedding_model):
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=_embedding_model)

#trnsforms chroma database object "db" into a langchain retreiver object
def load_retriever(_db):
    return _db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@st.cache_resource
def load_llm():
    return Ollama(model="llama3",temperature=0.0)

def main():
    st.title("📄🔍 Research Paper Q&A (RAG System)")

    embedding_model = load_embedding_model()
    db = load_vectorstore(embedding_model)
    retriever = load_retriever(db)
    llm = load_llm()

    #this creates a complete question-answering chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True #this will also return the actual source 
    )

    query = st.text_input("Enter your question:", placeholder="e.g. What is the main innovation in 'Attention is All You Need'?")

    if st.button("Get Answer") and query:
        result = qa_chain({"query": query})
        st.subheader("Answer:")
        st.write(result["result"])

        st.subheader("Sources and Retrieved Chunks with Similarity Scores:")
        docs_with_scores = db.similarity_search_with_score(query, k=3)

        for i, (doc, score) in enumerate(docs_with_scores):
            st.markdown(f"**Chunk {i+1} | Similarity Score: {score:.4f} | Source: {doc.metadata.get('source', 'N/A')}**")
            st.code(doc.page_content[:1000] + "...")

if __name__ == "__main__":
    main()
