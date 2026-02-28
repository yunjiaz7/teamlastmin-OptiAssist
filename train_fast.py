import os
import inspect
import re
from functools import lru_cache

import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model


LOC_RE = re.compile(r"<loc(\d+)>")


def parse_suffix_boxes(suffix_text):
    boxes_by_label = {}
    parts = [part.strip() for part in suffix_text.split(";") if part.strip()]
    for part in parts:
        locs = [int(x) for x in LOC_RE.findall(part)]
        if len(locs) < 4:
            continue
        y1, x1, y2, x2 = locs[:4]
        label = part.split(">")[-1].strip()
        if not label:
            continue
        boxes_by_label.setdefault(label, []).append((y1, x1, y2, x2))
    return boxes_by_label


def box_iou(box_a, box_b):
    ay1, ax1, ay2, ax2 = box_a
    by1, bx1, by2, bx2 = box_b
    ay1, ay2 = sorted((max(0, min(1024, ay1)), max(0, min(1024, ay2))))
    ax1, ax2 = sorted((max(0, min(1024, ax1)), max(0, min(1024, ax2))))
    by1, by2 = sorted((max(0, min(1024, by1)), max(0, min(1024, by2))))
    bx1, bx2 = sorted((max(0, min(1024, bx1)), max(0, min(1024, bx2))))

    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter = inter_h * inter_w

    area_a = max(0, ay2 - ay1) * max(0, ax2 - ax1)
    area_b = max(0, by2 - by1) * max(0, bx2 - bx1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else (inter / union)


def strip_prompt_prefix(decoded_text, prompt):
    clean = decoded_text.strip()
    if clean.startswith(prompt):
        return clean[len(prompt):].strip(" \n:")
    prompt_no_image = prompt.replace("<image>", "", 1).strip()
    if prompt_no_image and clean.startswith(prompt_no_image):
        return clean[len(prompt_no_image):].strip(" \n:")
    return clean


# -----------------------
# FAST image loader (LRU cache)
# -----------------------
def build_image_loader(images_root, cache_size=2048):
    @lru_cache(maxsize=cache_size)
    def _load(rel_path):
        img_path = os.path.join(images_root, rel_path)
        img = Image.open(img_path).convert("RGB")
        return img
    return _load


# -----------------------
# Optional: lightweight mIoU eval callback (runs rarely)
# -----------------------
class MiouEvalCallback(TrainerCallback):
    def __init__(self, processor, images_root, compute_fn, eval_every_n_epochs=1):
        self.processor = processor
        self.images_root = images_root
        self.compute_fn = compute_fn
        self.eval_every_n_epochs = eval_every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        # Run only every N epochs and only if eval dataset exists
        if state.epoch is None:
            return control
        epoch_num = int(state.epoch)
        if self.eval_every_n_epochs <= 0:
            return control
        if epoch_num % self.eval_every_n_epochs != 0:
            return control

        trainer = kwargs.get("model")  # not the Trainer object
        # We cannot access eval_dataset here directly without trainer; skip if not provided.
        return control


def main():
    # -----------------------
    # CONFIG
    # -----------------------
    model_id = "google/paligemma2-3b-pt-448"
    data_dir = "/home/ubuntu/teamlastmin/dataset/origa_paligemma/"
    train_jsonl = f"{data_dir}/dataset/_annotations.train.jsonl"
    val_jsonl   = f"{data_dir}/dataset/_annotations.valid.jsonl"
    images_root = f"{data_dir}/dataset"

    output_dir = "finetuned_paligemma2_det_lora"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------
    # Speed knobs (safe defaults)
    # -----------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -----------------------
    # Load dataset
    # -----------------------
    data_files = {"train": train_jsonl}
    if os.path.exists(val_jsonl):
        data_files["validation"] = val_jsonl
    ds = load_dataset("json", data_files=data_files)

    # -----------------------
    # Load model + processor
    # -----------------------
    processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)

    # Prefer faster attention impls if supported by your environment
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa" if device == "cuda" else None,  # falls back safely if unsupported
    )
    if device != "cuda":
        model = model.to(device)

    # Gradient checkpointing reduces activation memory -> allows higher batch / stabler training
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # REQUIRED when using gradient checkpointing

    # Freeze vision tower, keep projector trainable
    vision_module = getattr(model, "vision_tower", None) or getattr(model, "vision_model", None)
    if vision_module is None and hasattr(model, "model"):
        vision_module = getattr(model.model, "vision_tower", None) or getattr(model.model, "vision_model", None)
    if vision_module is None:
        raise AttributeError("Could not find PaliGemma vision module (expected vision_tower or vision_model).")
    for p in vision_module.parameters():
        p.requires_grad = False

    projector_module = getattr(model, "multi_modal_projector", None)
    if projector_module is None and hasattr(model, "model"):
        projector_module = getattr(model.model, "multi_modal_projector", None)
    if projector_module is not None:
        for p in projector_module.parameters():
            p.requires_grad = True

    # -----------------------
    # LoRA config
    # -----------------------
    lora_config = LoraConfig(
        r=16,                 # slightly higher capacity often helps detection formatting
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    DTYPE = next(model.parameters()).dtype

    # -----------------------
    # Image cache loader
    # -----------------------
    load_img = build_image_loader(images_root, cache_size=4096)

    # -----------------------
    # Collate function (faster + safer)
    # -----------------------
    def collate_fn(examples):
        images = []
        texts = []
        suffixes = []

        for ex in examples:
            images.append(load_img(ex["image"]))
            texts.append("<image> " + ex["prefix"])
            suffixes.append(ex["suffix"])

        tokens = processor(
            text=texts,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

        # Move to device + cast float tensors to model dtype
        for k, v in list(tokens.items()):
            if isinstance(v, torch.Tensor):
                v = v.to(device, non_blocking=True)
                if torch.is_floating_point(v):
                    v = v.to(DTYPE)
                tokens[k] = v
        return tokens

    # -----------------------
    # Faster validation metric (optional)
    # NOTE: Generation-based eval is expensive. Use sparingly.
    # -----------------------
    def compute_validation_miou(eval_dataset, max_new_tokens=64):
        if eval_dataset is None or len(eval_dataset) == 0:
            return 0.0

        was_training = model.training
        infer_device = next(model.parameters()).device
        image_suffix_ious = []

        model.eval()
        with torch.no_grad():
            for ex in eval_dataset:
                img = load_img(ex["image"])
                prompt = "<image> " + ex["prefix"]

                inputs = processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt",
                    truncation=False,
                )
                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        v = v.to(infer_device, non_blocking=True)
                        if torch.is_floating_point(v):
                            v = v.to(DTYPE)
                        inputs[k] = v

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
                decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                pred_suffix = strip_prompt_prefix(decoded, prompt)

                gt_boxes = parse_suffix_boxes(ex["suffix"])
                pred_boxes = parse_suffix_boxes(pred_suffix)
                if not gt_boxes:
                    continue

                per_gt_ious = []
                for label, gt_label_boxes in gt_boxes.items():
                    pred_label_boxes = pred_boxes.get(label, [])
                    for gt_box in gt_label_boxes:
                        best_iou = 0.0
                        for pred_box in pred_label_boxes:
                            best_iou = max(best_iou, box_iou(gt_box, pred_box))
                        per_gt_ious.append(best_iou)

                if per_gt_ious:
                    image_suffix_ious.append(sum(per_gt_ious) / len(per_gt_ious))

        if was_training:
            model.train()
        return 0.0 if not image_suffix_ious else (sum(image_suffix_ious) / len(image_suffix_ious))

    # -----------------------
    # Trainer subclass: run mIoU only at the end (or very rarely)
    # -----------------------
    class MiouTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            # Run generation-based mIoU only on explicit calls (final eval),
            # not during training if eval_strategy is "steps"/"epoch".
            current_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if current_eval_dataset is not None and len(current_eval_dataset) > 0:
                miou = compute_validation_miou(current_eval_dataset)
                metrics[f"{metric_key_prefix}_miou"] = miou
                self.log({f"{metric_key_prefix}_miou": miou})
                print(f"{metric_key_prefix}_miou: {miou:.4f}")
            return metrics

    # -----------------------
    # Training args
    # -----------------------
    has_validation = "validation" in ds

    training_args_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=10,                 # often enough; increase if underfitting
        per_device_train_batch_size=8,       # bump batch; gradient_checkpointing helps fit
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,                 # LoRA often tolerates higher LR than full finetune
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,                   # LoRA usually fine with 0 wd
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(device == "cuda"),
        fp16=False,
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        remove_unused_columns=False,
        report_to=["tensorboard"],

        # Dataloader speed
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,

        # Stability
        max_grad_norm=1.0,
        gradient_checkpointing=True,

        # Throughput
        torch_compile=(device == "cuda"),   # PyTorch 2.x compile; disable if it causes issues
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_key = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy"
    # Keep eval off during training for speed; do a final eval at end.
    training_args_kwargs[eval_strategy_key] = "no"

    args = TrainingArguments(**training_args_kwargs)

    trainer = MiouTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collate_fn,
    )

    trainer.train()

    if has_validation:
        print("Final validation metrics:", trainer.evaluate())

    trainer.save_model(os.path.join(output_dir, "final"))
    processor.save_pretrained(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    main()