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

# --- State Management untuk Upload Gambar ---
if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = 0
if "has_uploaded" not in st.session_state:
    st.session_state.has_uploaded = False

image_base64 = None
image_bytes_for_history = None

# Widget upload gambar hanya muncul jika belum mengunggah gambar dan bukan dalam status klarifikasi item/porsi dari gambar
is_clarifying_image = False
for msg in reversed(st.session_state.messages):
    if msg["role"] == "assistant":
        if "Klarifikasi:" in msg["content"] or "mendeteksi item berikut dari gambar" in msg["content"]:
            is_clarifying_image = True
        break

if not st.session_state.has_uploaded and not is_clarifying_image:
    uploaded_file = st.file_uploader(
        "📸 Unggah foto makanan Anda (Opsional) dan tuliskan jam makan", 
        type=["jpg", "jpeg", "png"],
        key=f"file_uploader_{st.session_state.uploaded_file_key}"
    )
    
    if uploaded_file is not None:
        image_bytes_for_history = uploaded_file.read()
        image_base64 = base64.b64encode(image_bytes_for_history).decode("utf-8")
        st.session_state.temp_image_bytes = image_bytes_for_history
        st.session_state.temp_image_base64 = image_base64
        st.session_state.has_uploaded = True
        st.rerun()
elif st.session_state.has_uploaded:
    # Jika sudah mengunggah, ambil dari session state agar tetap ada saat dikirim
    image_bytes_for_history = st.session_state.get("temp_image_bytes")
    image_base64 = st.session_state.get("temp_image_base64")
    
    # Tampilkan preview gambar dan opsi untuk membatalkan/mengganti foto
    st.image(image_bytes_for_history, caption="Preview Gambar Makanan", width=300)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("📷 Foto makanan telah terlampir.")
    with col2:
        if st.button("Batalkan Foto"):
            st.session_state.has_uploaded = False
            st.session_state.uploaded_file_key += 1
            if "temp_image_bytes" in st.session_state:
                del st.session_state.temp_image_bytes
            if "temp_image_base64" in st.session_state:
                del st.session_state.temp_image_base64
            st.rerun()

if prompt := st.chat_input("isi apa yang kamu makan hari ini. Jika ada foto makanan, sebutkan jam makan saja"):
    with st.chat_message("user"):
        if image_bytes_for_history:
            st.image(image_bytes_for_history, width=300)
        st.markdown(prompt)
    
    # Simpan riwayat user
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "image": image_bytes_for_history if image_bytes_for_history else None
    })
    
    with st.chat_message("assistant"):
        spinner_text = "Menganalisis foto dan informasi gizi..." if image_base64 else "Memproses pesan..."
        with st.spinner(spinner_text):
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
                    try:
                        data = response.json()
                    except ValueError:
                        st.error("❌ Error: Respons dari server bukan format JSON yang valid.")
                        st.stop()
                    
                    extracted_items = data.get("extracted_items", [])
                    final_analysis = data.get("final_analysis", "")
                    needs_clarification = data.get("needs_clarification", False)
                    
                    if needs_clarification:
                         formatted_response = f"🔍 **Klarifikasi:**\n{final_analysis}"
                    else:
                         formatted_response = f"**Item terdeteksi:** {', '.join(extracted_items)}\n\n---\n\n**Analisis Gizi & Literatur:**\n{final_analysis}"
                    
                    st.markdown(formatted_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    
                    # Reset state setelah analisis selesai agar widget upload muncul kembali
                    st.session_state.has_uploaded = False
                    st.session_state.uploaded_file_key += 1
                    if "temp_image_bytes" in st.session_state:
                        del st.session_state.temp_image_bytes
                    if "temp_image_base64" in st.session_state:
                        del st.session_state.temp_image_base64
                    
                    st.rerun()
                    
                else:
                    error_detail = response.text
                    try:
                        err_json = response.json()
                        error_detail = err_json.get("detail", response.text)
                    except ValueError:
                        pass
                    st.error(f"❌ Error dari server (Status {response.status_code}): {error_detail}")
                    
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {str(e)}")