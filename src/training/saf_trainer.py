import os
import torch
from transformers import Trainer


class SAFTrainer(Trainer):
    """
    SAF-RAG Trainer (OPTIMIZED VERSION)

    Improvements:
    - Explicit LM vs SAF balancing
    - Prevents over-regularization
    - Better logging
    - Stable training
    """

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    # =========================
    # LOSS
    # =========================
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):
        # Remove unsupported keys
        inputs.pop("decoder_input_ids", None)
        inputs.pop("decoder_attention_mask", None)

        outputs = model(**inputs)

        # -----------------------------
        # Extract components
        # -----------------------------
        if isinstance(outputs, dict):
            saf_loss = outputs["loss"]
            lm_loss = outputs.get("lm_loss", saf_loss)
            risk = outputs.get("risk", torch.tensor(0.0))
        else:
            saf_loss = outputs.loss
            lm_loss = saf_loss
            risk = torch.tensor(0.0)

        if saf_loss is None:
            raise ValueError("Model did not return a valid loss.")

        saf_loss = saf_loss.float()
        lm_loss = lm_loss.float()

        saf_loss = torch.nan_to_num(saf_loss, nan=0.0)
        lm_loss = torch.nan_to_num(lm_loss, nan=0.0)

        # -----------------------------
        # 🔥 DYNAMIC BALANCING (KEY FIX)
        # -----------------------------
        # If risk is high → apply SAF strongly
        # If risk is low → focus on LM learning

        if isinstance(risk, torch.Tensor):
            avg_risk = risk.mean().item()
        else:
            avg_risk = float(risk)

        # Adaptive weighting
        alpha = 0.7 + 0.6 * avg_risk   # SAF strength
        beta = 1.0                     # LM always important

        loss = beta * lm_loss + alpha * (saf_loss - lm_loss)

        # -----------------------------
        # Logging
        # -----------------------------
        log_dict = {
            "loss": loss.detach(),
            "lm_loss": lm_loss.detach(),
            "saf_loss": saf_loss.detach(),
            "alpha": torch.tensor(alpha),
        }

        if isinstance(outputs, dict):
            if "lambda" in outputs:
                log_dict["lambda"] = outputs["lambda"].detach()
            if "risk" in outputs:
                log_dict["risk"] = outputs["risk"].detach()

        self.log({k: v.mean().item() for k, v in log_dict.items()})

        return (loss, outputs) if return_outputs else loss

    # =========================
    # SAVE MODEL
    # =========================
    def save_model(self, output_dir=None, _internal_call=False):

        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        model = self.model

        if hasattr(model, "module"):
            model = model.module

        # HuggingFace model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin")
            )
            print(f"✅ Saved state_dict → {output_dir}/pytorch_model.bin")

        # Save tokenizer
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Save training args
        torch.save(
            self.args,
            os.path.join(output_dir, "training_args.bin")
        )