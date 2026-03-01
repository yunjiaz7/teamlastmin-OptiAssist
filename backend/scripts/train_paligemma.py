import os
import inspect
import re
import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
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
    if union <= 0:
        return 0.0
    return inter / union


def strip_prompt_prefix(decoded_text, prompt):
    clean = decoded_text.strip()
    if clean.startswith(prompt):
        return clean[len(prompt):].strip(" \n:")
    prompt_no_image = prompt.replace("<image>", "", 1).strip()
    if prompt_no_image and clean.startswith(prompt_no_image):
        return clean[len(prompt_no_image):].strip(" \n:")
    return clean


def main():
    # -----------------------
    # CONFIG (edit these)
    # -----------------------
    model_id = "google/paligemma2-3b-pt-448"  # same as the post
    data_dir = "/home/ubuntu/teamlastmin/dataset/origa_paligemma/"            # <-- your root
    train_jsonl = f"{data_dir}/dataset/_annotations.train.jsonl"
    val_jsonl   = f"{data_dir}/dataset/_annotations.valid.jsonl"  # optional
    images_root = f"{data_dir}/dataset"

    output_dir = "finetuned_paligemma2_det_lora"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------
    # Load dataset
    # -----------------------
    data_files = {"train": train_jsonl}
    if os.path.exists(val_jsonl):
        data_files["validation"] = val_jsonl

    ds = load_dataset("json", data_files=data_files)

    # Ensure image paths are absolute and load as PIL in collate_fn
    # (We’ll read images in collate_fn to keep dataset lightweight)

    # -----------------------
    # Load model + processor
    # -----------------------
    processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    if device != "cuda":
        model = model.to(device)

    # Freeze vision + projector (attribute names vary by transformers version)
    vision_module = getattr(model, "vision_tower", None)
    if vision_module is None:
        vision_module = getattr(model, "vision_model", None)
    if vision_module is None and hasattr(model, "model"):
        # Some checkpoints nest the core modules under model.model
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
            p.requires_grad = False

    # -----------------------
    # LoRA config (same spirit as post)
    # -----------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    DTYPE = model.dtype  # important: tokens cast to same dtype

    # -----------------------
    # Collate function (key part)
    # -----------------------
    def collate_fn(examples):
        """
        Each example has:
          - image: relative path like "train/xxx.jpg"
          - prefix: "detect a ; b ; c"
          - suffix: "<loc..> a ; <loc..> b ; ..."
        We must feed:
          text = "<image>" + prefix
          suffix = detection string
        """
        images = []
        texts = []
        suffixes = []

        for ex in examples:
            # image path
            img_path = os.path.join(images_root, ex["image"])
            img = Image.open(img_path).convert("RGB")
            images.append(img)

            # IMPORTANT: include <image> token
            texts.append("<image> " + ex["prefix"])

            # detection targets
            suffixes.append(ex["suffix"])

        tokens = processor(
            text=texts,
            images=images,
            suffix=suffixes,              # <- labels are appended like the post
            return_tensors="pt",
            padding="longest",
            # PaliGemma expands <image> into many special tokens (e.g., 1024).
            # Truncating can cut these tokens and crash with image-token mismatch.
            truncation=False,
        )

        # Move + cast
        tokens = {k: v.to(device) for k, v in tokens.items()}
        # Cast float tensors to model dtype (bf16 on GPU)
        for k, v in tokens.items():
            if torch.is_floating_point(v):
                tokens[k] = v.to(DTYPE)

        return tokens

    def compute_validation_miou(eval_dataset, max_new_tokens=96):
        if eval_dataset is None or len(eval_dataset) == 0:
            return 0.0

        was_training = model.training
        infer_device = next(model.parameters()).device
        image_suffix_ious = []

        model.eval()
        with torch.no_grad():
            for ex in eval_dataset:
                img_path = os.path.join(images_root, ex["image"])
                img = Image.open(img_path).convert("RGB")
                prompt = "<image> " + ex["prefix"]

                inputs = processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt",
                    truncation=False,
                )
                inputs = {k: v.to(infer_device) for k, v in inputs.items()}
                for k, v in inputs.items():
                    if torch.is_floating_point(v):
                        inputs[k] = v.to(DTYPE)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
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

        if not image_suffix_ious:
            return 0.0
        return sum(image_suffix_ious) / len(image_suffix_ious)

    class MiouTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
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
        num_train_epochs=8,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        warmup_steps=50,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(device == "cuda"),
        fp16=False,  # keep false if using bf16
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        remove_unused_columns=False,  # REQUIRED for vision+text
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_key = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy"
    # Disable periodic evaluation; run evaluation once after training completes.
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

    # Save final
    trainer.save_model(os.path.join(output_dir, "final"))
    processor.save_pretrained(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    main()