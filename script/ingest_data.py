import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import sys

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.loader import load_data

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_docs") # Ubah nama folder agar lebih umum
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

def ingest_all_sources():
    print("Memulai proses Ingestion Data ke ChromaDB...")
    
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    if not files:
        print("❌ Tidak ada file di folder data/raw_docs/")
        return

    all_documents = []
    
    # 1. Gunakan loader dinamis milikmu
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        try:
            docs = load_data(file_path)
            all_documents.extend(docs)
        except Exception as e:
            print(f"⚠️ Gagal memuat {file}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="LazarusNLP/all-indo-e5-small-v4", model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
    
    print(f"🎉 Selesai! {len(chunks)} chunks tersimpan di ChromaDB.")

if __name__ == "__main__":
    ingest_all_sources()