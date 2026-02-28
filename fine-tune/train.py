import os
import glob
import time
import torch
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from model import get_peft_whisper_model
from dataloader import get_dataset_and_collator

MODEL_ID    = "openai/whisper-large-v3"
OUTPUT_DIR  = "./uzbek-whisper"
MAX_STEPS   = 1000
EVAL_EVERY  = 100
SAVE_EVERY  = 50

def find_latest_checkpoint(output_dir):
    """Return the path to the latest Trainer checkpoint, or None."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.split("-")[-1])
    )
    return checkpoints[-1] if checkpoints else None


class PeftSeq2SeqTrainer(Seq2SeqTrainer):

    def floating_point_ops(self, inputs):
        return 0

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result



class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        resumed = state.global_step > 0
        print(f"\n{'='*60}")
        if resumed:
            print(f"RESUMING from step {state.global_step}")
        print(f"TRAINING START  |  max_steps={args.max_steps}  |  "
              f"effective_batch={args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"LR={args.learning_rate}  |  bf16={args.bf16}")
        print(f"{'='*60}\n")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 25 == 0 and state.global_step > 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            elapsed    = time.time() - self.start_time
            steps_done = state.global_step
            steps_left = args.max_steps - steps_done
            eta_s      = (elapsed / steps_done) * steps_left if steps_done else 0
            eta_min, eta_sec = divmod(int(eta_s), 60)

            loss = state.log_history[-1].get("loss", 0) if state.log_history else 0
            lr   = state.log_history[-1].get("learning_rate", 0) if state.log_history else 0
            pct  = steps_done / args.max_steps * 100

            print(f"  Step {steps_done:>4}/{args.max_steps} | {pct:5.1f}% | "
                  f"loss={loss:.4f} | lr={lr:.2e} | ETA {eta_min:02d}:{eta_sec:02d}")

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self.start_time
        m, s  = divmod(int(total), 60)
        h, m  = divmod(m, 60)
        print(f"\n{'='*60}")
        print(f"DONE  |  {state.global_step} steps  |  Total: {h:02d}:{m:02d}:{s:02d}")
        print(f"{'='*60}\n")


class CheckpointCallback(TrainerCallback):


    def __init__(self, output_dir):
        self.best_loss = float("inf")
        self.best_dir  = os.path.join(output_dir, "best_model")
        self.last_dir  = os.path.join(output_dir, "last_model")
        os.makedirs(self.best_dir, exist_ok=True)
        os.makedirs(self.last_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        step         = state.global_step
        current_loss = (metrics or {}).get("eval_loss", float("inf"))

        print(f"  [Step {step}] Saving last_model  (eval_loss={current_loss:.4f}) ...")
        model.save_pretrained(self.last_dir)

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            print(f"  [Step {step}] ★ New best! Saving best_model ...")
            model.save_pretrained(self.best_dir)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Final save regardless of eval schedule
        print(f"\n  Saving final last_model ...")
        model.save_pretrained(self.last_dir)
        print(f"  Best model → {self.best_dir}  (eval_loss={self.best_loss:.4f})")
        print(f"  Last model → {self.last_dir}")




def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        resume_from = find_latest_checkpoint(OUTPUT_DIR)
        if resume_from:
            print(f"\n★ Found checkpoint — resuming from: {resume_from}")
        else:
            print("\n  No checkpoint found — starting from scratch.")

        # ── Dataset ──────────────────────────────────────────────────────────
        print("\nStep 1: Loading dataset...")
        train_dataset, test_dataset, data_collator, _ = get_dataset_and_collator(
            model_id=MODEL_ID,
            max_train_samples=3000,
            max_eval_samples=200,
        )
        print(f"  Train: {len(train_dataset)}  |  Eval: {len(test_dataset)}")

        # ── Model ────────────────────────────────────────────────────────────
        print("\nStep 2: Loading model...")
        model = get_peft_whisper_model(model_id=MODEL_ID)
        model.config.use_cache = False


        last_adapter = os.path.join(OUTPUT_DIR, "last_model", "adapter_model.safetensors")
        if resume_from and os.path.exists(last_adapter):
            print(f"  Loading LoRA weights from last_model/ ...")
            from peft import set_peft_model_state_dict
            import safetensors.torch as st
            state = st.load_file(last_adapter)
            set_peft_model_state_dict(model, state)
            print("  LoRA weights restored.")

        # ── Training args ────────────────────────────────────────────────────
        print("\nStep 3: Configuring training...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,

            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,      #
            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            max_steps=MAX_STEPS,
            warmup_steps=50,

            learning_rate=5e-4,
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,

            predict_with_generate=False,
            eval_strategy="steps",
            eval_steps=EVAL_EVERY,
            save_strategy="steps",
            save_steps=SAVE_EVERY,
            save_total_limit=3,
            load_best_model_at_end=False,

            logging_steps=25,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = PeftSeq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback(), CheckpointCallback(OUTPUT_DIR)],
        )

        print(f"\nStep 4: Training (target: {MAX_STEPS} steps)...")
        trainer.train(resume_from_checkpoint=resume_from)

        print("\nStep 5: Saving final model...")
        model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n✓ Training completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())