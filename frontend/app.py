# frontend/app.py
import streamlit as st
import requests
import os
import base64 # Tambahkan library base64

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/analyze")

st.set_page_config(page_title="NutriGraph AI", page_icon="🥗", layout="centered")

st.title("🥗 NutriGraph AI: agentic nutrition tracker")
st.markdown("""
Asisten AI ini menggunakan Langchain advanced RAG (Vector + BM25) dan LangGraph multi-agent workflow untuk menganalisis asupan nutrisi Anda 
berdasarkan input teks natural, gambar makanan, dan literatur jurnal medis.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image"):
            st.image(message["image"], width=300)
        st.markdown(message["content"])

# --- Tambahkan UI Upload Gambar ---
uploaded_file = st.file_uploader("📸 Unggah foto makanan Anda (Opsional) dan tuliskan jam makan", type=["jpg", "jpeg", "png"])
image_base64 = None

if uploaded_file is not None:
    # Tampilkan preview gambar
    # st.image(uploaded_file, caption="Preview Gambar", width=300)
    # Konversi ke Base64
    image_base64 = base64.b64encode(uploaded_file.read()).decode("utf-8")

if prompt := st.chat_input("isi apa yang kamu makan hari ini. Jika ada foto makanan, sebutkan jam makan saja"):
    with st.chat_message("user"):
        if uploaded_file:
            st.image(uploaded_file, width=300)
        st.markdown(prompt)
    
    # Simpan riwayat user
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "image": uploaded_file if uploaded_file else None
    })
    
    with st.chat_message("assistant"):
        with st.spinner("Agen sedang menganalisis Gambar, Teks, & RAG..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={
                        "user_input": prompt,
                        "session_id": st.session_state.session_id,
                        "image_data": image_base64 # Kirim data base64 ke backend
                    },
                    timeout=90 # Sedikit dinaikkan karena VLM bisa butuh waktu proses lebih lama
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    extracted_items = data.get("extracted_items", [])
                    final_analysis = data.get("final_analysis", "")
                    needs_clarification = data.get("needs_clarification", False)
                    
                    if needs_clarification:
                         formatted_response = f"🔍 **Klarifikasi:**\n{final_analysis}"
                    else:
                         formatted_response = f"**Item terdeteksi:** {', '.join(extracted_items)}\n\n---\n\n**Analisis Gizi & Literatur:**\n{final_analysis}"
                    
                    st.markdown(formatted_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    
                else:
                    st.error(f"❌ Error dari server: {response.status_code}")
                    
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {str(e)}")