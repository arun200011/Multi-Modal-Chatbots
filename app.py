# app.py (root)
import streamlit as st
from backend.inference import PlantGuard
import tempfile, os, json
from utils.speech_utils import transcribe_audio

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="PlantGuard", layout="centered")
st.title("🌿 PlantGuard — Multimodal Plant Disease Chatbot")

# --------------------------
# Load FAQ file
# --------------------------
FAQ_PATH = os.path.join("data", "faq.json")
faq_data = []
if os.path.exists(FAQ_PATH):
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

# --------------------------
# Initialize PlantGuard
# --------------------------
@st.cache_resource
def load_system():
    try:
        return PlantGuard(models_dir="models")
    except Exception as e:
        st.error(f"⚠️ Failed to load models: {e}")
        return None

pg = load_system()

# --------------------------
# Image-only
# --------------------------
st.header("🖼️ Image Diagnosis")
img_file = st.file_uploader("Upload leaf image", type=['jpg','jpeg','png'], key="image_only")
if img_file and pg:
    st.image(img_file, caption="Uploaded leaf", use_column_width=True)
    res = pg.predict_image(img_file)
    st.success(f"Prediction: **{res['label']}** (Confidence: {res['confidence']:.2f})")

# --------------------------
# Audio-only
# --------------------------
st.header("🎤 Voice Description")
audio_file = st.file_uploader("Upload voice (wav/mp3)", type=['wav','mp3'], key="audio_only")
if audio_file and pg:
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    t.write(audio_file.read()); t.flush(); t.close()

    transcript = transcribe_audio(t.name)
    st.info(f"Transcript: _{transcript}_")

    res = pg.predict_audio(t.name)
    st.success(f"Predicted disease from voice: **{res['label']}** ({res['confidence']:.2f})")

    os.unlink(t.name)

# --------------------------
# FAQ / Text-only
# --------------------------
st.header("💬 Ask a Question")
q = st.text_input("Ask plant-care question")
if st.button("Ask", key="faq"):
    answer = None
    # simple rule-based lookup
    for item in faq_data:
        if item["question"].lower() in q.lower():
            answer = item["answer"]
            break
    if answer:
        st.write("Answer:", answer)
    else:
        # fallback to QA model
        ans = pg.answer(q, context=" ".join([x["answer"] for x in faq_data]))
        st.write("Answer (AI):", ans.get("answer"))

# --------------------------
# Fusion Mode
# --------------------------
st.header("🌐 Fusion Mode (Image + Audio + Text)")
img_fusion = st.file_uploader("Upload leaf image (fusion)", type=['jpg','jpeg','png'], key="fusion_img")
audio_fusion = st.file_uploader("Upload voice (fusion)", type=['wav','mp3'], key="fusion_audio")
text_fusion = st.text_area("Describe symptoms or ask a question (fusion)")

if st.button("Run Fusion Prediction"):
    if img_fusion and audio_fusion and text_fusion.strip():
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        t.write(audio_fusion.read()); t.flush(); t.close()

        res = pg.predict_fusion(img_fusion, t.name, text_fusion)
        if "error" in res:
            st.error(res["error"])
        else:
            st.success(f"Fusion prediction: **{res['label']}** (confidence {res['confidence']:.2f})")
            st.bar_chart({"confidence": res["probs"]}, x=pg.label_map)

        os.unlink(t.name)
    else:
        st.warning("Please provide image, audio, and text for fusion prediction.")


