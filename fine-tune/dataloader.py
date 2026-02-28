import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any
from datasets import load_dataset, Audio
from transformers import WhisperProcessor


class WhisperDataset(Dataset):


    def __init__(self, hf_dataset, processor):
        self.samples = []
        skipped = 0

        print(f"  Converting {len(hf_dataset)} samples to tensors (this happens once)...")
        for i, row in enumerate(hf_dataset):
            try:
                audio         = row["audio"]
                audio_array   = audio["array"]
                sampling_rate = audio["sampling_rate"]

                input_features = processor.feature_extractor(
                    audio_array, sampling_rate=sampling_rate
                ).input_features[0]   # shape: (80, 3000) float32

                transcription = row.get("text") or row.get("sentence") or "[empty]"
                labels        = processor.tokenizer(transcription).input_ids

                self.samples.append({
                    "input_features": input_features.tolist(),
                    "labels":         labels,
                })
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"    Skipping sample {i}: {e}")

            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{len(hf_dataset)} ...")

        print(f"  Done. {len(self.samples)} valid samples ({skipped} skipped).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_features": torch.tensor(s["input_features"], dtype=torch.float32),
            "labels":         torch.tensor(s["labels"],         dtype=torch.long),
        }


@dataclass
class DataCollator:
    processor: Any

    def __call__(self, features):
        valid = [f for f in features if f is not None and "input_features" in f]
        if not valid:
            return None

        inputs = [{"input_features": f["input_features"]} for f in valid]
        batch  = self.processor.feature_extractor.pad(inputs, return_tensors="pt")

        labels = [{"input_ids": f["labels"]} for f in valid]
        labels = self.processor.tokenizer.pad(labels, return_tensors="pt")

        batch["labels"] = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )
        return batch


def get_dataset_and_collator(
    model_id="openai/whisper-large-v3",
    dataset_id="yakhyo/mozilla-common-voice-uzbek",
    max_train_samples=3000,
    max_eval_samples=200,
):
    processor = WhisperProcessor.from_pretrained(
        model_id, language="uz", task="transcribe"
    )

    print(f"  Downloading slices (train={max_train_samples}, eval={max_eval_samples})...")
    raw_train = load_dataset(dataset_id, split=f"train[:{max_train_samples}]")
    raw_eval  = load_dataset(dataset_id, split=f"test[:{max_eval_samples}]")

    raw_train = raw_train.cast_column("audio", Audio(sampling_rate=16000))
    raw_eval  = raw_eval.cast_column("audio",  Audio(sampling_rate=16000))

    print("  Building train dataset...")
    train_dataset = WhisperDataset(raw_train, processor)

    print("  Building eval dataset...")
    eval_dataset  = WhisperDataset(raw_eval,  processor)

    del raw_train, raw_eval

    return train_dataset, eval_dataset, DataCollator(processor), processor