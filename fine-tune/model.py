import torch
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model


def get_peft_whisper_model(model_id="openai/whisper-large-v3"):
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise