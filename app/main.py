from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from app.core import Qwen3TTSEngine
from contextlib import asynccontextmanager
import io, os, uuid, soundfile as sf
import torch
import gc


engine = Qwen3TTSEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在載入 Qwen3TTS 模型至 GPU...")
    engine.load_model()
    yield
    print("正在關閉 API 並釋放顯存...")
    if engine.model is not None:
        engine.model = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # 進階清理：清理進程間通訊的顯存

app = FastAPI(lifespan=lifespan)


# -------------------------
# Voice Design
# 於text處輸入希望TTS模型生成的語句
# 於language輸入生成語言
# 於instruct填入說話者的語氣、情緒及性別等等，可以使用白話文。
# 由於是中國研發的TTS模型，Prompt建議使用簡體中文以避免發音錯誤。
# -------------------------

@app.post("/generate/voice_design")
async def voice_design(
    text: str = Form(
        ...,
        description="【必填】想要模型說出的文字內容，使用簡體中文以避免發音錯誤。",
        example="你好，我是一位虚拟助理，今天很高兴能够有这个机会认识各位，并和各位介绍功能。"
    ),
    language: str = Form(
        ....,
        description="【必填】想要模型生成的語言：Chinese, English...",
    )
    instruct: str = From(
        ....,
        description="【必填】想要模型生成的語言：Chinese, English...",
        example="中年女性，温雅中性的声音，音调偏高且起伏明显"
    )
):
    wav, sr = engine.generate(
            text=text,
            language=language,
            instruct=instruct
        )
    return wav_to_stream(wav, sr)


async def save_temp_file(upload_file: UploadFile):
    ext = os.path.splitext(upload_file.filename)[1]
    tmp_path = f"temp_{uuid.uuid4()}{ext}"
    with open(tmp_path, "wb") as buffer:
        buffer.write(await upload_file.read())
    return tmp_path

def wav_to_stream(wav, sr):
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format='WAV')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")
