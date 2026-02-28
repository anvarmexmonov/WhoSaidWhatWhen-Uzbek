from datasets import load_dataset, Audio

DATASET_ID = "yakhyo/mozilla-common-voice-uzbek"

print("=" * 50)
print(" Dataset Downloader & Verifier")
print("=" * 50)


print("\nChecking dataset structure (first 3 samples)...")
sample = load_dataset(DATASET_ID, split="train[:3]")
sample = sample.cast_column("audio", Audio(sampling_rate=16000))

print(f"\nColumns available : {sample.column_names}")
print(f"Samples loaded    : {len(sample)}")
print()

for i, row in enumerate(sample):
    text  = row.get("sentence") or row.get("text") or "N/A"
    audio = row.get("audio", {})
    dur   = len(audio.get("array", [])) / audio.get("sampling_rate", 16000)
    print(f"  [{i}] text     : {text}")
    print(f"       duration : {dur:.2f}s")
    print(f"       sr       : {audio.get('sampling_rate')} Hz")
    print()

# Check split sizes without downloading audio
print("Checking split sizes (metadata only)...")
train_meta = load_dataset(DATASET_ID, split="train")
test_meta  = load_dataset(DATASET_ID, split="test")
print(f"  Train split : {len(train_meta):,} samples")
print(f"  Test split  : {len(test_meta):,} samples")
print(f"\n  We will use : train[:3000] + test[:200] for fine-tuning")
print("\nDataset OK - ready to train.")