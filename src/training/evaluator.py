import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve
from scipy.interpolate import interp1d

from src.dataset.loader import extract_video_id

@dataclass
class EvaluationResult:
    frame_auc: float
    frame_eer: float
    frame_accuracy: float
    frame_macro_f1: float
    frame_threshold: float
    video_auc: float
    video_eer: float
    video_accuracy: float
    video_threshold: float
    ece_uncalibrated: float
    ece_calibrated: float
    optimal_temperature: float
    classification_report_str: str

    def summary_markdown(self) -> str:
        return f"""| Metric | Frame-Level | Video-Level |
|---|---|---|
| AUC | {self.frame_auc:.4f} | {self.video_auc:.4f} |
| EER | {self.frame_eer:.4f} | {self.video_eer:.4f} |
| Accuracy | {self.frame_accuracy:.4f} | {self.video_accuracy:.4f} |
| Threshold | {self.frame_threshold:.4f} | {self.video_threshold:.4f} |
| Macro F1 | {self.frame_macro_f1:.4f} | - |

| Calibration | Value |
|---|---|
| ECE (Uncalibrated) | {self.ece_uncalibrated:.4f} |
| ECE (Calibrated) | {self.ece_calibrated:.4f} |
| Optimal Temp | {self.optimal_temperature:.4f} |
"""

def calculate_crash_proof_eer(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Calculate EER using sklearn.metrics.roc_curve. First try interpolation, and fallback to vector search."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5
    
    y_prob = np.nan_to_num(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    fnr = 1 - tpr
    
    try:
        # Interpolation
        from scipy.optimize import brentq
        unique_fpr, indices = np.unique(fpr, return_index=True)
        unique_tpr = tpr[indices]
        unique_thresholds = thresholds[indices]
        eer = brentq(lambda x: 1. - x - interp1d(unique_fpr, unique_tpr)(x), 0., 1.)
        thresh = interp1d(unique_fpr, unique_thresholds)(eer)
        return float(eer), float(thresh)
    except Exception:
        # Vector search fallback
        idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        return float(eer), float(thresholds[idx])

def aggregate_video_predictions(samples: list, probs: np.ndarray, aggregation_mode: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """Group frame predictions by video ID using extract_video_id and take probability per video based on mode."""
    video_map = {}
    for sample, prob in zip(samples, probs):
        path, label = sample[0], sample[1]
        first_path = path[0] if isinstance(path, (list, tuple)) else path
        vid_id = extract_video_id(first_path)
        
        if vid_id not in video_map:
            video_map[vid_id] = {'probs': [], 'label': label}
        video_map[vid_id]['probs'].append(prob)
        
    vid_labels = []
    vid_probs = []
    for vid_id in sorted(video_map.keys()):
        if aggregation_mode == "mean":
            vid_probs.append(np.mean(video_map[vid_id]['probs']))
        elif aggregation_mode == "median":
            vid_probs.append(np.median(video_map[vid_id]['probs']))
        elif aggregation_mode == "max":
            vid_probs.append(np.max(video_map[vid_id]['probs']))
        vid_labels.append(video_map[vid_id]['label'])
        
    return np.array(vid_labels), np.array(vid_probs)

def calculate_adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error using equal-frequency (quantile) binning."""
    if len(y_prob) == 0:
        return 0.0
    bin_boundaries = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bin_boundaries[-1] += 1e-8
    bin_boundaries[0] -= 1e-8
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def tune_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Find scalar T > 0 minimizing BCE loss on validation set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logits_t = torch.as_tensor(val_logits, dtype=torch.float32, device=device).view(-1)
    labels_t = torch.as_tensor(val_labels, dtype=torch.float32, device=device).view(-1)
    
    alpha = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([alpha], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()
    
    def closure():
        optimizer.zero_grad()
        T = torch.clamp(torch.exp(alpha), min=0.05)
        loss = criterion(logits_t / T, labels_t)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    return float(torch.clamp(torch.exp(alpha), min=0.05).item())

def evaluate_full_suite(model: nn.Module, loader: DataLoader, device: torch.device, val_loader: Optional[DataLoader] = None, is_sequence: bool = False) -> EvaluationResult:
    """Evaluates the full suite of metrics for the model."""
    model.eval()
    all_logits = []
    all_probs = []
    all_targets = []
    
    use_amp = torch.cuda.is_available()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    if hasattr(torch.amp, "autocast"):
        autocast_ctx = torch.amp.autocast(device_type, enabled=use_amp)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
        
    def _forward(batch):
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            imgs, labels, padding_mask = batch[0], batch[1], batch[2]
            fwd_input = (imgs.to(device), padding_mask.to(device))
            fwd_input_flip = (torch.flip(imgs.to(device), dims=[-1]), padding_mask.to(device))
        else:
            imgs, labels = batch[0], batch[1]
            fwd_input = (imgs.to(device), None)
            fwd_input_flip = (torch.flip(imgs.to(device), dims=[-1]), None)
            
        with autocast_ctx:
            if fwd_input[1] is not None:
                logits = model(fwd_input[0], padding_mask=fwd_input[1])
                logits_flip = model(fwd_input_flip[0], padding_mask=fwd_input_flip[1])
            else:
                logits = model(fwd_input[0])
                logits_flip = model(fwd_input_flip[0])
                
        logits_avg = (logits + logits_flip) / 2.0
        probs = torch.sigmoid(logits_avg)
        return logits_avg, probs, labels
        
    with torch.no_grad():
        for batch in loader:
            logits, probs, labels = _forward(batch)
            logits = logits.view(-1)
            probs = probs.view(-1)
            labels = labels.view(-1)
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    all_logits = np.array(all_logits)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Frame metrics
    frame_auc = float(roc_auc_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.5
    frame_eer, frame_eer_th = calculate_crash_proof_eer(all_targets, all_probs)
    
    # Macro F1 Threshold Calibration
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, opt_thresh = 0.0, 0.5
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = float(f1_score(all_targets, preds, average='macro', zero_division=0))
        if f1 > best_f1:
            best_f1, opt_thresh = f1, float(t)
            
    frame_preds = (all_probs >= opt_thresh).astype(int)
    frame_acc = float(np.mean(frame_preds == all_targets))
    
    report_str = classification_report(all_targets, frame_preds, zero_division=0, target_names=["Real", "Fake"])
    
    # ECE
    ece_uncal = calculate_adaptive_ece(all_targets, all_probs)
    
    # Temp scaling
    temp = tune_temperature_scaling(all_logits, all_targets)
        
    calibrated_probs = 1.0 / (1.0 + np.exp(-all_logits / temp))
    ece_cal = calculate_adaptive_ece(all_targets, calibrated_probs)
    
    # Video metrics
    if hasattr(loader.dataset, 'samples'):
        samples = loader.dataset.samples
    elif hasattr(loader.dataset, 'video_samples'):
        samples = loader.dataset.video_samples
    else:
        # Fallback dummy samples
        samples = [("dummy", int(t)) for t in all_targets]
        
    vid_targets, vid_probs = aggregate_video_predictions(samples, all_probs)
    
    vid_auc = float(roc_auc_score(vid_targets, vid_probs)) if len(np.unique(vid_targets)) > 1 else 0.5
    vid_eer, vid_eer_th = calculate_crash_proof_eer(vid_targets, vid_probs)
    
    vid_preds = (vid_probs >= vid_eer_th).astype(int)
    vid_acc = float(np.mean(vid_preds == vid_targets))
    
    return EvaluationResult(
        frame_auc=frame_auc,
        frame_eer=frame_eer,
        frame_accuracy=frame_acc,
        frame_macro_f1=best_f1,
        frame_threshold=opt_thresh,
        video_auc=vid_auc,
        video_eer=vid_eer,
        video_accuracy=vid_acc,
        video_threshold=vid_eer_th,
        ece_uncalibrated=ece_uncal,
        ece_calibrated=ece_cal,
        optimal_temperature=temp,
        classification_report_str=report_str
    )
