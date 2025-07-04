import io
import os
import base64
import tempfile
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from groq import Groq
from mistralai import Mistral

# Pour Streamlit secrets fallback
try:
    import streamlit as st
    use_streamlit_secrets = True
except ImportError:
    use_streamlit_secrets = False

# Constantes API
CLIPDROP_API_URL = "https://clipdrop-api.co/text-to-image/v1"
OPENAI_STT_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"

# App FastAPI
app = FastAPI()

# Modèles Pydantic
class TextPayload(BaseModel):
    text: str

class PromptPayload(BaseModel):
    prompt: str

# 🔐 Sécurité - Récupération des secrets
def get_secret(name: str) -> str:
    if use_streamlit_secrets and name in st.secrets:
        return st.secrets[name]
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Clé API '{name}' manquante.")
    return value

# 🗣️ Transcription audio via Whisper OpenAI
async def transcribe_audio_func(file: UploadFile) -> dict:
    api_key = get_secret("OPENAI_API_KEY")
    file_content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            file_bytes = f.read()

        files = {
            "file": (file.filename, io.BytesIO(file_bytes), file.content_type)
        }
        data = {"model": "whisper-1"}
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENAI_STT_ENDPOINT,
                headers=headers,
                files=files,
                data=data
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return {"transcription": response.json().get("text", "")}

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# 😶 Analyse d'émotion avec Groq
async def detect_emotion_func(text: str):
    api_key = get_secret("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    prompt = (
        "Lis ce rêve et classe-le dans l'une de ces catégories : heureux, stressant, neutre, cauchemar, étrange. "
        "Réponds uniquement par la catégorie.\nRêve :\n" + text
    )

    completion = await client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Tu es un assistant expert en analyse d'émotions de rêves."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )
    return {"emotion": completion.choices[0].message.content.strip().lower()}

# 🎨 Génération de prompt image avec Groq
async def generate_image_prompt_func(text: str):
    api_key = get_secret("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    prompt = (
        "Lis ce rêve et écris un prompt descriptif pour générer une image onirique qui l’illustre. "
        "Sois concis et imagé.\nRêve :\n" + text
    )

    completion = await client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Tu es un assistant expert en prompts d’images oniriques."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        temperature=0.7
    )
    return {"prompt": completion.choices[0].message.content.strip()}

# 🧠 Génération de prompt image avec Mistral
async def generate_mistral_prompt_func(text: str):
    api_key = get_secret("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    prompt = (
        "Tu es un assistant qui transforme des rêves en prompts artistiques pour la génération d’images oniriques. "
        "Utilise un français visuel, poétique et concis.\nVoici le rêve :\n" + text
    )

    completion = await client.chat.completions.create(
        model="mistral-tiny",
        messages=[
            {"role": "system", "content": "Assistant pour transformer des rêves en prompts artistiques."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        temperature=0.7
    )
    return {"prompt": completion.choices[0].message.content.strip()}

# 🖼️ Génération d’image avec ClipDrop
async def generate_image_func(prompt: str):
    api_key = get_secret("CLIPDROP_API_KEY")
    headers = {"x-api-key": api_key}

    async with httpx.AsyncClient() as client:
        response = await client.post(CLIPDROP_API_URL, json={"prompt": prompt}, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    base64_image = base64.b64encode(response.content).decode("utf-8")
    return {"image": f"data:image/png;base64,{base64_image}"}

# 🚀 Endpoints FastAPI
@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    return await transcribe_audio_func(file)

@app.post("/analyze-emotion")
async def analyze_emotion(payload: TextPayload):
    return await detect_emotion_func(payload.text)

@app.post("/generate-image-prompt")
async def generate_prompt(payload: TextPayload):
    return await generate_image_prompt_func(payload.text)

@app.post("/generate-image")
async def generate_image(payload: PromptPayload):
    return await generate_image_func(payload.prompt)

@app.post("/generate-mistral-prompt")
async def mistral_prompt(payload: TextPayload):
    return await generate_mistral_prompt_func(payload.text)

@app.post("/dream-to-image")
async def dream_to_image(file: UploadFile = File(...)):
    transcription = await transcribe_audio_func(file)
    text = transcription["transcription"]

    emotion = await detect_emotion_func(text)
    prompt_data = await generate_image_prompt_func(text)
    image_data = await generate_image_func(prompt_data["prompt"])

    return {
        "transcription": text,
        "emotion": emotion["emotion"],
        "prompt": prompt_data["prompt"],
        "image": image_data["image"]
    }
