## 快速啟動 Qwen3-tts

1. 建立環境:
   ```bash
   conda create -n qwen3-tts python=3.12 -y
   conda activate qwen3-tts
   ```

2. 安裝Qwen3-tts
    ```bash
    pip install -U qwen-tts
   ```

3. 安裝依賴:
    ```bash
    # 建議先安裝對應顯卡的 PyTorch，例如:
    pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
    # 安裝其餘依賴
    pip install -r requirements.txt
    ```

4. 啟動 API:
    ```bash
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

    啟動後請訪問：http://localhost:8000/docs 進入 Swagger UI 進行測試。

## 使用說明

Qwen3-tts的優勢為語調、音色及情緒生成，可以用白話描述需要的聲音型態及場景

Qwen3-tts的API先實裝了Voice Design一種功能，其餘功能相較VoxCPM2沒有明顯優勢：

1. Voice Design
    於text處輸入希望TTS模型生成的語句
    於language輸入生成語言
    於instruct填入說話者的語氣、情緒及性別等等，可以使用白話文。
    由於是中國研發的TTS模型，Prompt建議使用簡體中文以避免發音錯誤。

2. 其他功能
    模型支援非語言發音，如嘆氣、大笑...等等
    請參考原始模型[Github](https://github.com/QwenLM/Qwen3-TTS?tab=readme-ov-file#quickstart)
    如僅需進行簡單功能測試，可以直接線上使用[ Hugging Face Demo  ](https://huggingface.co/spaces/Qwen/Qwen3-TTS)
