import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from src.core.config import settings

def get_hybrid_retriever(search_type="similarity", k=3):
    print(f"🔌 Menyiapkan Hybrid Retriever (Vektor + BM25, Mengambil {k} dokumen)...")
    
    # 1. Setup Konfigurasi Vektor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL, 
        model_kwargs={"device": device}
    )
    
    vector_store = Chroma(
        persist_directory=settings.CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    
    # 2. Vector Retriever
    vector_retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    
    # 3. Ambil data dari ChromaDB untuk membuat indeks BM25
    db_data = vector_store.get()
    
    if not db_data or not db_data.get('documents'):
        print("⚠️ Peringatan: Tidak ada dokumen untuk dibuatkan indeks BM25. Menggunakan Vector Retriever saja.")
        return vector_retriever
        
    docs = [
        Document(page_content=txt, metadata=meta or {}) 
        for txt, meta in zip(db_data['documents'], db_data.get('metadatas', []))
    ]
    
    # 4. BM25 Retriever (Pencarian Kata Kunci)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    
    # 5. Hybrid Retriever (Ensemble Vektor + BM25 dengan bobot seimbang 50:50)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return hybrid_retriever