import torch
import os
import gc
import warnings
import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*SuppressTokens.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*deprecated.*")
warnings.filterwarnings("ignore", message=".*attention mask.*pad token.*")

# ==========================================
# CONFIGURATION
# ==========================================
HF_TOKEN      = "your_token"
AUDIO_FILE    = "your_audio_file.wav"

BASE_MODEL_ID = "openai/whisper-large-v3"
ADAPTER_DIR   = "AnvarMexmonov/uz-speech-adapter-v1"

LANGUAGE      = "uz"
TASK          = "transcribe"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def load_whisper(base_model_id, adapter_dir, device):
    print(f"  Loading base model ({base_model_id})...")
    processor = WhisperProcessor.from_pretrained(
        base_model_id, language=LANGUAGE, task=TASK
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    print(f"  Loading LoRA adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = None

    return model, processor


def transcribe_chunk(audio_array: np.ndarray, model, processor, device,
                     forced_ids) -> str:
    """Transcribe a float32 numpy array (16kHz mono)."""
    if len(audio_array) < 400:
        return ""

    inputs = processor.feature_extractor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)
    attention_mask = torch.ones(
        input_features.shape[:2], dtype=torch.long, device=device
    )
    if device == "cuda":
        input_features = input_features.half()

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_ids,
            max_new_tokens=225,
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    del input_features, attention_mask, predicted_ids
    if device == "cuda":
        torch.cuda.empty_cache()

    return text


def get_annotation(diarization_result):
    """Handle both old (Annotation) and new (DiarizeOutput) pyannote return types."""
    if hasattr(diarization_result, "speaker_diarization"):
        return diarization_result.speaker_diarization
    return diarization_result


def main():
    print("\n" + "=" * 65)
    print("  Uzbek Speaker Diarization + Transcription Pipeline")
    print("=" * 65)

    # ── 0. Convert audio ──────────────────────────────────────────
    print("\n[0/4] Converting audio to 16kHz mono WAV...")
    temp_wav = "_temp_input.wav"
    try:
        audio_seg = AudioSegment.from_file(AUDIO_FILE)
        audio_seg.export(temp_wav, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        print(f"      Duration: {len(audio_seg)/1000:.1f}s")
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")
        return

    # ── 1. Load models ────────────────────────────────────────────
    print("\n[1/4] Loading models...")
    print("  Loading pyannote diarization pipeline...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=HF_TOKEN
        ).to(torch.device(DEVICE))

    model, processor = load_whisper(BASE_MODEL_ID, ADAPTER_DIR, DEVICE)

    forced_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    print(f"  All models ready on {DEVICE}.")

    # ── 2. Diarization ────────────────────────────────────────────
    print(f"\n[2/4] Running speaker diarization on '{temp_wav}'...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diarization_result = diarization_pipeline(temp_wav)
    except Exception as e:
        print(f"  ✗ Diarization failed: {e}")
        os.remove(temp_wav)
        return

    annotation = get_annotation(diarization_result)

    del diarization_pipeline, diarization_result
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── 3. Load audio into RAM ────────────────────────────────────
    print("\n[3/4] Loading audio into RAM for segment slicing...")
    full_audio   = AudioSegment.from_file(temp_wav).set_frame_rate(16000).set_channels(1)
    full_samples = np.array(full_audio.get_array_of_samples()).astype(np.float32) / 32768.0
    print(f"      {len(full_samples)/16000:.1f}s loaded.")

    # ── 4. Transcribe ─────────────────────────────────────────────
    print(f"\n[4/4] Transcribing segments...\n")
    print("=" * 65)
    print(f"{'TIME':<16} | {'SPEAKER':<10} | TRANSCRIPT")
    print("=" * 65)

    results = []

    for segment, _, speaker in annotation.itertracks(yield_label=True):
        start_s = segment.start
        end_s   = segment.end

        if (end_s - start_s) < 0.3:
            continue

        start_idx = int(max(0, start_s - 0.1) * 16000)
        end_idx   = int(min(len(full_samples), (end_s + 0.1) * 16000))
        chunk     = full_samples[start_idx:end_idx]

        try:
            text = transcribe_chunk(chunk, model, processor, DEVICE, forced_ids)
        except Exception as e:
            text = f"[error: {e}]"

        if text:
            timestamp = f"{start_s:.1f}s–{end_s:.1f}s"
            print(f"{timestamp:<16} | {speaker:<10} | {text}")
            results.append({
                "start": start_s, "end": end_s,
                "speaker": speaker, "text": text,
            })

    print("=" * 65)

    # ── Save transcript ───────────────────────────────────────────
    out_txt = os.path.splitext(AUDIO_FILE)[0] + "_transcript.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[{r['start']:.1f}s-{r['end']:.1f}s] {r['speaker']}: {r['text']}\n")
    print(f"\n  Transcript saved → {out_txt}")

    os.remove(temp_wav)
    print("  Done! ✓\n")


if __name__ == "__main__":
    main()