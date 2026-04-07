import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from huggingface_hub import InferenceClient

st.set_page_config(page_title="inapinep.id", layout="centered")

LOGO3 = "https://raw.githubusercontent.com/SonyaAudina/Chatbot-inapinep.id/main/stiker%20lucuk%203.png"
LOGO4 = "https://raw.githubusercontent.com/SonyaAudina/Chatbot-inapinep.id/main/stiker%20lucuk%204.png"
LOGO5 = "https://raw.githubusercontent.com/SonyaAudina/Chatbot-inapinep.id/main/stiker%20lucuk%205.png"
LOGO2 = "https://raw.githubusercontent.com/SonyaAudina/Chatbot-inapinep.id/main/stiker%20lucuk%202.png"

st.markdown("""<style>
@import url("https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600&display=swap");
html, body, .stApp { background-color: #fff0f5 !important; color: #4a1942 !important; font-family: Nunito, sans-serif !important; }
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stChatInput"] textarea { background: #fff5f8 !important; color: #4a1942 !important; border: 2px solid #e8a0b4 !important; border-radius: 20px !important; }
[data-testid="stChatMessage"] { background: #fff5f8 !important; border: 1px solid #f0c4d4 !important; border-radius: 20px !important; padding: 14px !important; margin: 6px 0 !important; }
.suggest-btn button {
    background: linear-gradient(135deg, #e8a0b4, #d4608a) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    min-height: 64px !important;
    height: auto !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    padding: 10px 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.suggest-btn button:hover { opacity: 0.88 !important; }
.stButton button {
    background: linear-gradient(135deg, #e8a0b4, #d4608a) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
}
div[data-testid="column"] .suggest-btn {
    height: 100% !important;
}
</style>""", unsafe_allow_html=True)

st.markdown(f"""<div style="text-align:center;padding:2rem 0 1rem;">
<div style="font-family:serif;font-size:2.4rem;font-weight:700;background:linear-gradient(135deg,#e8a0b4,#d4608a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">inapinep.id</div>
<div style="color:#b06080;font-size:0.9rem;margin-top:8px;display:flex;align-items:center;justify-content:center;gap:5px;flex-wrap:wrap;">
<img src="{LOGO3}" width="24" style="vertical-align:middle;border-radius:50%;"> Temukan hotel impianmu
· <img src="{LOGO4}" width="24" style="vertical-align:middle;border-radius:50%;"> Pulau Jawa
· <img src="{LOGO5}" width="24" style="vertical-align:middle;border-radius:50%;"> Rekomendasi fasilitas
· <img src="{LOGO2}" width="24" style="vertical-align:middle;border-radius:50%;"> Info harga &amp; rating
</div>
<div style="margin-top:12px;display:inline-block;background:#fff5f8;border:1px solid #e8a0b4;padding:4px 16px;border-radius:20px;font-size:0.75rem;color:#d4608a;">🟢 ONLINE</div>
</div>""", unsafe_allow_html=True)

@st.cache_resource
def load_all_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "dataset_hotel_indonesia (1).csv"))
    df = df.fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["facilities"] = df["facilities"].apply(lambda x: re.sub(r"[^a-zA-Z0-9 ,]", "", str(x)).strip())
    df["text"] = (df["hotel_name"] + " " + df["property_type"] + " " + df["city"] + " " + df["facilities"]).str.lower()
    return df

try:
    df = load_all_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def ask_hf(messages):
    models_to_try = ["mistralai/Mixtral-8x7B-Instruct-v0.1","Qwen/Qwen2.5-72B-Instruct","google/gemma-2-2b-it","HuggingFaceH4/zephyr-7b-beta"]
    client = InferenceClient(token=HF_TOKEN)
    for model in models_to_try:
        try:
            response = client.chat_completion(model=model, messages=messages, max_tokens=700, temperature=0.7)
            return response.choices[0].message.content
        except Exception:
            continue
    return "Aduh maaf, chatbot sedang tidak tersedia. Coba lagi beberapa menit yaa! 🌸"

def cari_hotel(query, kota=None, max_harga=None, min_rating=None):
    hasil = df.copy()
    if kota:
        hasil = hasil[hasil["city"].str.contains(kota, case=False, na=False, regex=False)]
    if max_harga:
        hasil = hasil[hasil["min_price"] <= max_harga]
    if min_rating:
        hasil = hasil[hasil["rating"] >= min_rating]
    if query:
        hasil = hasil[hasil["text"].str.contains(query.lower(), na=False, regex=False)]
    return hasil.sort_values("rating", ascending=False).head(5)

def format_hotel(row):
    return (
        "<div style='background:#fff5f8;border:1px solid #f0c4d4;border-radius:16px;padding:14px;margin:8px 0;box-shadow:0 2px 8px rgba(232,160,180,0.1);'>" +
        f"<div style='font-size:1rem;font-weight:700;color:#d4608a;'>🏩 {row['hotel_name']}</div>" +
        f"<div style='margin-top:6px;font-size:0.85rem;color:#7a4060;'>📍 {row['city']}<br>⭐ Rating: <b>{row['rating']}</b><br>💰 Harga: Rp {int(row['min_price']):,} - Rp {int(row['max_price']):,}<br>🏷️ Tipe: {row['property_type']}</div>" +
        "</div>"
    )

SYSTEM_PROMPT = """Kamu adalah inapinep.id — asisten rekomendasi hotel Indonesia yang cerdas, ramah, dan friendly khusus untuk perempuan.

ATURAN PENTING — WAJIB DIIKUTI:
- Kamu HANYA boleh menjawab pertanyaan yang berkaitan dengan hotel dan penginapan, seperti: rekomendasi hotel, fasilitas hotel, harga hotel, rating hotel, tips memilih hotel, lokasi hotel di kota-kota Pulau Jawa.
- Jika pengguna bertanya di luar topik hotel dan penginapan (seperti matematika, sejarah, sains, politik, bahasa, teknologi, atau topik umum lainnya), TOLAK dengan sopan dan arahkan kembali ke topik hotel.
- Contoh penolakan: "Maaf ya, aku hanya bisa membantu seputar hotel dan penginapan nih! 🌸 Ada hotel yang mau dicari atau ditanyakan?"
- Tetap friendly, hangat, dan encouraging meskipun menolak pertanyaan di luar topik.

Kamu membantu pengguna menemukan hotel terbaik di Indonesia berdasarkan kota, fasilitas, budget, dan rating.
Format jawaban: Bahasa Indonesia yang hangat, ramah, dan encouraging! Gunakan emoji yang sesuai 🌸✨💕"""

def proses_chat(prompt):
    kota_keywords = ["surabaya","jakarta","semarang","bandung","yogyakarta","malang","madiun","gresik","pacitan","magelang"]
    kota_found = next((k for k in kota_keywords if k in prompt.lower()), None)
    hotel_results = cari_hotel(prompt, kota=kota_found)
    context = ""
    hotel_cards = ""
    if len(hotel_results) > 0:
        hotel_cards = "".join([format_hotel(row) for _, row in hotel_results.iterrows()])
        context = f"\nData hotel relevan: {hotel_results[['hotel_name','city','rating','min_price','max_price','property_type']].to_string()}"
    ai_messages = [{"role": "system", "content": SYSTEM_PROMPT + context}]
    for m in st.session_state.messages:
        ai_messages.append({"role": m["role"], "content": m["content"]})
    ai_messages.append({"role": "user", "content": prompt})
    full_reply = ask_hf(ai_messages)
    return hotel_cards, full_reply

if "messages" not in st.session_state: st.session_state.messages = []

# Suggestion buttons (hanya saat belum ada chat)
if len(st.session_state.messages) == 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
        clicked1 = st.button("Hotel di Surabaya", key="btn1", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
        clicked2 = st.button("Hotel murah di Jakarta", key="btn2", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
        clicked3 = st.button("Hotel dengan spa dan kolam renang", key="btn3", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    prompt_from_btn = None
    if clicked1: prompt_from_btn = "Hotel di Surabaya"
    elif clicked2: prompt_from_btn = "Hotel murah di Jakarta"
    elif clicked3: prompt_from_btn = "Hotel dengan spa dan kolam renang"

    if prompt_from_btn:
        st.session_state.messages.append({"role": "user", "content": prompt_from_btn})
        with st.chat_message("user"):
            st.markdown(prompt_from_btn)
        with st.chat_message("assistant"):
            with st.spinner("Mencarikan hotel terbaik untukmu 🌸..."):
                hotel_cards, full_reply = proses_chat(prompt_from_btn)
                if hotel_cards:
                    st.markdown(hotel_cards, unsafe_allow_html=True)
                st.markdown(full_reply, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Penginapan terbaik khusus untukmu 🌸"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Mencarikan hotel terbaik untukmu 🌸..."):
            hotel_cards, full_reply = proses_chat(prompt)
            if hotel_cards:
                st.markdown(hotel_cards, unsafe_allow_html=True)
            st.markdown(full_reply, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_reply})

if st.session_state.messages:
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()
