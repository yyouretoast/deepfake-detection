import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, ".")
from scripts.run_release1_kaggle import (
    DatasetResolver,
    DomainClassifier,
    FaceCropDataset,
    HybridDeepfakeDetector,
    ModelEvaluator,
    fit_temperature_log,
    get_differential_param_groups,
    get_transforms,
    resolve_splits_path,
)
from src.evaluation.metrics import compute_eer


def run_single_fold(fold_name, holdout_key, display_name, data_root, splits, device, epochs=3, batch_size=32):
    train_samples = splits["train"]
    test_samples = splits["test"]

    # Filter train split: pure FaceForensics++ (exclude Celeb-DF completely and exclude held-out fakes)
    retained_train = []
    for s in train_samples:
        p, lbl = s[0], s[1]
        norm = p.replace("\\", "/").lower()
        parts = norm.split("/")
        # Exclude Celeb-DF completely from training
        if "celeb" in norm or (len(parts) > 1 and "id" in parts[1]):
            continue
        # Exclude held-out manipulation method fakes
        if lbl == 1.0 and DomainClassifier.matches_holdout(p, holdout_key):
            continue
        retained_train.append(s)

    # Held-out test fakes
    held_fakes = [
        s for s in test_samples
        if s[1] == 1.0 and DomainClassifier.matches_holdout(s[0], holdout_key)
    ]

    # Matched authentic FF++ test reals (3-digit folder names)
    all_reals = [
        s for s in test_samples
        if s[1] == 0.0 and s[0].replace("\\", "/").split("/")[1].isdigit() and len(s[0].replace("\\", "/").split("/")[1]) == 3
    ]

    # Canonical 1:1 balanced evaluation cohort (Rossler et al., ICCV 2019)
    rng = random.Random(42)
    n_eval = min(len(held_fakes), len(all_reals))
    eval_fakes = held_fakes[:n_eval]
    eval_reals = rng.sample(all_reals, n_eval)
    eval_samples = eval_fakes + eval_reals

    print("\n=======================================================", flush=True)
    print(f"=== {fold_name}: {display_name} ===", flush=True)
    print(f"  Retained Train: {len(retained_train)} (Pure FF++ minus {display_name})", flush=True)
    print(f"  Eval Set (1:1 Balanced): {len(eval_samples)} ({len(eval_fakes)} Fakes, {len(eval_reals)} Reals)", flush=True)
    print("=======================================================", flush=True)

    train_transform, eval_transform = get_transforms(img_size=256, hardened=True)

    labels = [int(s[1]) for s in retained_train]
    w_real = 1.0 / max(1, len(labels) - sum(labels))
    w_fake = 1.0 / max(1, sum(labels))
    weights = [w_fake if y == 1 else w_real for y in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        FaceCropDataset(retained_train, data_root, is_train=True, transform=train_transform),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        FaceCropDataset(eval_samples, data_root, is_train=False, transform=eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = HybridDeepfakeDetector(pretrained=True, frequency_backbone="resse").to(device)
    optimizer = torch.optim.AdamW(get_differential_param_groups(model, lr_backbone=1e-5, lr_head=1e-4, weight_decay=1e-2), weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_samples = 0
        for step, (images, b_labels, v_flags) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            b_labels = b_labels.to(device, non_blocking=True).unsqueeze(1)
            v_mask = v_flags.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs, aux_outputs = model(images, return_aux=True)
                loss_main = (criterion(outputs, b_labels) * v_mask).sum() / torch.clamp(v_mask.sum(), min=1.0)
                loss_aux = (criterion(aux_outputs, b_labels) * v_mask).sum() / torch.clamp(v_mask.sum(), min=1.0)
                loss = loss_main + 0.30 * loss_aux

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * len(images)
            n_samples += len(images)
            if step % 150 == 0:
                print(f"  [{ep+1}/{epochs}] Step {step}/{len(train_loader)} Loss: {running_loss/n_samples:.4f}", flush=True)

        elapsed = time.time() - t0
        print(f"  Epoch {ep+1} completed in {elapsed:.1f}s | Avg Loss: {running_loss/n_samples:.4f}", flush=True)

    # Evaluate zero-shot on 1:1 balanced held-out cohort
    model.eval()
    evaluator = ModelEvaluator(model, device=device)
    logits, targets, valid = evaluator.predict_loader(eval_loader)
    mask = valid > 0.0
    logits = logits[mask]
    targets = targets[mask]

    # Zero-shot evaluation using uncalibrated logits to prevent test-set temperature leakage
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.50).astype(int)
    t_fold = fit_temperature_log(logits, targets)  # Diagnostic post-hoc oracle check only

    auc = float(roc_auc_score(targets, probs))
    bacc = float(balanced_accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, zero_division=0))
    prec = float(precision_score(targets, preds, zero_division=0))
    eer, _ = compute_eer(targets, probs)

    print(f"\n--- {fold_name} ({display_name}) FINAL VERIFIED RESULTS ---", flush=True)
    print(f"  Holdout Samples: {len(eval_fakes)} (Eval Cohort: {len(eval_samples)})", flush=True)
    print(f"  Fitted Platt T*: {t_fold:.4f}", flush=True)
    print(f"  Zero-Shot ROC AUC: {auc:.4f}", flush=True)
    print(f"  Balanced Accuracy: {bacc:.4f}", flush=True)
    print(f"  Fake F1: {f1:.4f}", flush=True)
    print(f"  Precision: {prec:.4f}", flush=True)
    print(f"  Equal Error Rate (EER): {eer*100:.2f}%", flush=True)

    return {
        "fold_name": fold_name,
        "display_name": display_name,
        "holdout_samples": len(eval_fakes),
        "fitted_t_star": float(t_fold),
        "zero_shot_auc": auc,
        "balanced_accuracy": bacc,
        "zero_shot_f1": f1,
        "zero_shot_precision": prec,
        "eer": eer,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}", flush=True)

    data_root = DatasetResolver.find_dataset_root("deepfake_crops_512")
    splits_path = resolve_splits_path(data_root)
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    folds = [
        ("Fold 1", "deepfakes", "FF++ Deepfakes (Pairs 0-99)"),
        ("Fold 2", "face2face", "FF++ Face2Face (Pairs 100-399)"),
        ("Fold 3", "faceswap", "FF++ FaceSwap (Pairs 400-599)"),
        ("Fold 4", "neuraltextures", "FF++ NeuralTextures (Pairs 600-799)"),
    ]

    results = {}
    output_dir = "results/release1_run"
    checkpoint_file = os.path.join(output_dir, "lomo_canonical_results.json")

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Found existing checkpoint with {len(results)} folds completed.", flush=True)
        except Exception:
            results = {}

    for fold_name, holdout_key, display_name in folds:
        if holdout_key in results:
            print(f"Skipping {fold_name} ({display_name}) - already completed.", flush=True)
            continue

        res = run_single_fold(fold_name, holdout_key, display_name, data_root, splits, device, epochs=3, batch_size=32)
        results[holdout_key] = res
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\n=======================================================", flush=True)
    print("=== CANONICAL 4-FOLD LOMO BENCHMARK SUMMARY ===", flush=True)
    print("=======================================================", flush=True)
    aucs = [r["zero_shot_auc"] for r in results.values()]
    baccs = [r["balanced_accuracy"] for r in results.values()]
    f1s = [r["zero_shot_f1"] for r in results.values()]
    precs = [r["zero_shot_precision"] for r in results.values()]
    eers = [r["eer"] for r in results.values()]

    print(f"Macro-Average ROC AUC:       {np.mean(aucs):.4f}", flush=True)
    print(f"Macro-Average Balanced Acc:  {np.mean(baccs):.4f}", flush=True)
    print(f"Macro-Average Fake F1:       {np.mean(f1s):.4f}", flush=True)
    print(f"Macro-Average Precision:     {np.mean(precs):.4f}", flush=True)
    print(f"Macro-Average EER:           {np.mean(eers)*100:.2f}%", flush=True)

if __name__ == "__main__":
    main()
