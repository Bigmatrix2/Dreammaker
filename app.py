import streamlit as st
import tempfile
import asyncio
import os
import time
from dotenv import load_dotenv
from backend import (
    transcribe_audio_func,
    detect_emotion_func,
    generate_image_prompt_func,
    generate_image_func
)

load_dotenv()
st.set_page_config(page_title="Synthétiseur de rêves", layout="centered")

st.title("🌙 Synthétiseur de Rêves")
st.markdown("Raconte un rêve à voix haute… et vois-le prendre vie ✨")

# -----------------------------------------------------
# Classes utilitaires
# -----------------------------------------------------
class DummyUploadFile:
    def __init__(self, path):
        self.filename = os.path.basename(path)
        ext = self.filename.split('.')[-1].lower()
        self.content_type = "audio/mpeg" if ext == "mp3" else "audio/wav" if ext == "wav" else "application/octet-stream"
        self._file = open(path, "rb")

    async def read(self):
        return self._file.read()

    def close(self):
        if not self._file.closed:
            self._file.close()

class Payload:
    def __init__(self, text_or_prompt):
        self.text = text_or_prompt
        self.prompt = text_or_prompt

# -----------------------------------------------------
# Wrappers synchrones
# -----------------------------------------------------
def transcribe_audio_sync(audio_path):
    async def inner():
        dummy_file = DummyUploadFile(audio_path)
        try:
            return await transcribe_audio_func(dummy_file)
        finally:
            dummy_file.close()
    return asyncio.run(inner())

def detect_emotion_sync(text):
    async def inner():
        return await detect_emotion_func(Payload(text))
    return asyncio.run(inner())

def generate_image_prompt_sync(text):
    async def inner():
        return await generate_image_prompt_func(Payload(text))
    return asyncio.run(inner())

def generate_image_sync(prompt):
    return generate_image_func(Payload(prompt))

# -----------------------------------------------------
# Upload de l’audio
# -----------------------------------------------------
uploaded_file = st.file_uploader("🎙️ Téléverse ton rêve (fichier .mp3 ou .wav)", type=["mp3", "wav"])

if uploaded_file:
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    try:
        # 1. Transcription
        with st.spinner("🔤 Transcription du rêve en cours..."):
            transcription_resp = transcribe_audio_sync(audio_path)
            transcription = transcription_resp.get("transcription", "") if isinstance(transcription_resp, dict) else ""
            if not transcription:
                st.warning("Aucune transcription détectée.")
            else:
                st.subheader("📝 Rêve transcrit :")
                st.success(transcription)

        if transcription:
            # 2. Détection émotion
            with st.spinner("🎭 Analyse de l’émotion du rêve..."):
                emotion_resp = detect_emotion_sync(transcription)
                emotion = emotion_resp.get("emotion", "").lower() if isinstance(emotion_resp, dict) else ""
                emojis = {
                    "heureux": "😊", "stressant": "😰", "neutre": "😐",
                    "cauchemar": "😱", "étrange": "🌌"
                }
                st.subheader(f"{emojis.get(emotion, '🌙')} Ambiance du rêve :")
                st.info(emotion.capitalize() if emotion else "Indéterminée")

            # 3. Génération prompt
            with st.spinner("🧠 Génération du prompt onirique..."):
                prompt_resp = generate_image_prompt_sync(transcription)
                prompt = prompt_resp.get("prompt") if isinstance(prompt_resp, dict) else None
                if prompt:
                    st.subheader("🧾 Prompt généré :")
                    st.code(prompt)
                else:
                    st.warning("Pas de prompt généré.")

            # 4. Génération image
            if prompt:
                with st.spinner("🖼️ Génération de l’image du rêve..."):
                    image_data_resp = generate_image_sync(prompt)
                    image_data = image_data_resp.get("image") if isinstance(image_data_resp, dict) else None
                    if image_data:
                        st.subheader("🌌 Image générée :")
                        st.image(image_data, use_column_width=True)
                    else:
                        st.warning("Aucun visuel généré.")

            # 5. Téléchargement transcription
            st.download_button(
                label="📄 Télécharger la transcription",
                data=transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )

            # 6. Recommencer
            if st.button("🔁 Recommencer"):
                st.experimental_rerun()

            # 7. Debug optionnel
            if st.checkbox("🔍 Voir les réponses brutes (debug)"):
                st.json({
                    "transcription_response": transcription_resp,
                    "emotion_response": emotion_resp,
                    "prompt_response": prompt_resp,
                    "image_response": image_data_resp if 'image_data_resp' in locals() else {}
                })

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")

    finally:
        # Nettoyage sécurisé
        if os.path.exists(audio_path):
            try:
                time.sleep(0.5)  # Pour s'assurer que le fichier n’est plus verrouillé
                os.remove(audio_path)
            except Exception as e:
                st.warning(f"⚠️ Impossible de supprimer le fichier temporaire : {e}")

else:
    st.info("🛌 Veuillez uploader un fichier audio de votre rêve pour commencer.")
