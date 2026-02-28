import torch
import sys
import os
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

BASE_MODEL_ID = "openai/whisper-large-v3"
ADAPTER_DIR   = "AnvarMexmonov/uz-speech-adapter-v1"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE      = "uz"
TASK          = "transcribe"
AUDIO_FILE    = "your_audio_file.wav"   #


def load_model():
    print(f"Loading base model ({BASE_MODEL_ID})...")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL_ID, language=LANGUAGE, task=TASK
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    print(f"Loading LoRA adapter from {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    print(f"Model ready on {DEVICE}.\n")
    return model, processor


def transcribe_array(audio_array: np.ndarray, model, processor) -> str:
    inputs = processor.feature_extractor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features.to(DEVICE)
    attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=DEVICE)

    if DEVICE == "cuda":
        input_features = input_features.half()

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            language=LANGUAGE,
            task=TASK,
            max_new_tokens=225,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def transcribe_file(audio_path: str, model, processor) -> str:
    import librosa
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    print(f"Loading audio: {audio_path}")
    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Duration: {len(audio_array)/16000:.1f}s  |  Transcribing...")
    return transcribe_array(audio_array, model, processor)


if __name__ == "__main__":
    model, processor = load_model()

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    elif AUDIO_FILE:
        files = [AUDIO_FILE]
    else:
        files = []

    if files:
        for path in files:
            try:
                text = transcribe_file(path, model, processor)
                print(f"\n[{os.path.basename(path)}]\n→ {text}\n")
            except FileNotFoundError as e:
                print(f"  ✗ {e}")
    else:
        result = transcribe_array(np.zeros(16000, dtype=np.float32), model, processor)
        print(f"→ '{result}'")
