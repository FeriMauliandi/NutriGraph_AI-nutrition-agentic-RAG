# File: src/utils/embeddings.py

import torch
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings
from src.core.config import settings

def get_cached_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Embedding inti (yang melakukan komputasi berat)
    underlying_embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL, 
        model_kwargs={"device": device}
    )
    
    # 2. Siapkan tempat penyimpanan lokal untuk cache vektor
    store = LocalFileStore("./.cache/embeddings_store")
    
    # 3. Bungkus embedding inti dengan CacheBackedEmbeddings
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embeddings,
        document_embedding_cache=store,
        namespace=settings.EMBEDDING_MODEL
    )
    
    return cached_embedder