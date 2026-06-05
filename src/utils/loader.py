import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader, CSVLoader

def load_data(source: str):
    print(f"Mencoba memuat data dari: {source}")
    
    if source.startswith("http://") or source.startswith("https://"):
        loader = WebBaseLoader(source)
        documents = loader.load()
        
    elif source.lower().endswith(".pdf"):
        if not os.path.exists(source): raise FileNotFoundError(f"File tidak ditemukan: {source}")
        loader = PyPDFLoader(source)
        documents = loader.load()
        
    elif source.lower().endswith(".md"):
        if not os.path.exists(source): raise FileNotFoundError(f"File tidak ditemukan: {source}")
        loader = TextLoader(source, encoding="utf-8")
        documents = loader.load()
        
    elif source.lower().endswith(".csv"):
        if not os.path.exists(source): raise FileNotFoundError(f"File tidak ditemukan: {source}")
        loader = CSVLoader(source, encoding="utf-8") 
        documents = loader.load()
        
    else:
        raise ValueError("Format tidak didukung. Harap masukkan URL web, .pdf, .md, atau .csv")
    
    return documents