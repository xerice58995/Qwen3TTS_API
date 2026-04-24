from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import Response
from app.core import Qwen3TTSEngine
from contextlib import asynccontextmanager
from typing import Optional
import io, os, uuid, soundfile as sf
import torch
import gc
import numpy as np
import tempfile


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
# Voice Clone
# 於text處輸入希望TTS模型生成的語句
# 於language輸入生成語言
# 於ref_audio上傳需要克隆的音檔
# 於ref_text上傳文字稿
# 由於是中國研發的TTS模型，Prompt建議使用簡體中文以避免發音錯誤
# -------------------------

@app.post("/generate/voice_clone")
async def voice_clone(
    text: str = Form(
        ...,
        description="【必填】想要模型說出的文字內容，使用簡體中文以避免發音錯誤。",
        examples=["你好，我是一位虚拟助理，今天很高兴能够有这个机会认识各位，并和各位介绍功能。"]
    ),
    language: str = Form(
        ...,
        description="【必填】想要模型生成的語言：Chinese, English...",
        examples=["Chinese"]
    ),
    ref_audio: UploadFile = File(
        ...,
        description="【必填】參考音檔（樣本），15秒左右，可從Samples資料夾選取。",
    ),
    ref_text: Optional[str] = Form(
        None,
        description="【選填】上傳音檔的文字稿，可留白。",
        example="【選填】上傳音檔的文字稿。",
    )
):
    wav, sr = engine.generate(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text
        )
    print(f"DEBUG: wav type: {type(wav)}, len: {len(wav) if wav is not None else 'None'}, sr: {sr}")
    if wav is None or len(wav) == 0:
        raise HTTPException(status_code=500, detail="模型未生成任何音訊數據")
    return wav_to_stream(wav, sr)


async def save_temp_file(upload_file: UploadFile):
    ext = os.path.splitext(upload_file.filename)[1]
    tmp_path = f"temp_{uuid.uuid4()}{ext}"
    with open(tmp_path, "wb") as buffer:
        buffer.write(await upload_file.read())
    return tmp_path

def wav_to_stream(wav, sr):
    # 如果 wav 是 [[...]] 這種格式，我們需要取出裡面的內容
    while isinstance(wav, list) and len(wav) == 1 and (isinstance(wav[0], list) or hasattr(wav[0], 'shape')):
        print("DEBUG: 偵測到嵌套結構，正在拆解...")
        wav = wav[0]

    # 1. 處理不同類型的輸入
    if isinstance(wav, list):
        # 如果是 list，先轉成 numpy
        wav = np.array(wav)
    elif hasattr(wav, 'cpu'):
        # 如果是 torch tensor，轉到 cpu 並轉成 numpy
        wav = wav.cpu().numpy()

    wav = wav.astype(np.float32).flatten()

    print(f"DEBUG: 最終音訊採樣數: {len(wav)}")

    # 數據正規化與防爆音
    if np.abs(wav).max() > 0:
        wav = wav / np.abs(wav).max()

    # 存成實體暫存檔 (Swagger UI 顯示 Bug )
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    sf.write(temp_file.name, wav, sr if sr else 24000, format='WAV', subtype='PCM_16')

    return FileResponse(
        path=temp_file.name,
        media_type="audio/wav",
        filename="qwen3_gen.wav"
    )
