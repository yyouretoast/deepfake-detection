#!/usr/bin/env python3
import os
import argparse
import json
import torch
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hybrid_detector import build_model
from torchvision import transforms

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/deepfake_convnext_v2.pth')
    parser.add_argument('--skip_train', action='store_true', help="Skip training for zero-shot evaluation")
    parser.add_argument('--celebdf_path', type=str, default='/kaggle/input/celeb-df-v2')
    parser.add_argument('--output', type=str, default='results/celebdf_cross_eval.json')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(pretrained=False)
    if os.path.exists(args.checkpoint):
        try:
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            from app import clean_state_dict
            state_dict = clean_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {args.checkpoint}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}, proceeding with untrained model.")
    
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_list_path = os.path.join(args.celebdf_path, 'List_of_testing_videos.txt')
    if not os.path.exists(test_list_path):
        print(f"Could not find {test_list_path}. Please check your dataset path.")
        return

    with open(test_list_path, 'r') as f:
        lines = f.readlines()

    video_preds = []
    video_labels = []
    frame_preds = []
    frame_labels = []

    print(f"Found {len(lines)} testing videos.")

    for line in tqdm(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        label = int(parts[0])
        # In Celeb-DF, 1=Real, 0=Fake. Map Fake to 1 and Real to 0 for binary classification.
        binary_label = 1 if label == 0 else 0
        
        video_path = os.path.join(args.celebdf_path, parts[1])
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            continue

        indices = np.linspace(0, frame_count - 1, 10, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                frames.append(transform(img))
        cap.release()

        if not frames:
            continue

        batch = torch.stack(frames).to(device)
        with torch.no_grad():
            outputs = model(batch)
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        vid_prob = float(np.mean(probs))
        video_preds.append(vid_prob)
        video_labels.append(binary_label)

        for p in probs:
            frame_preds.append(float(p))
            frame_labels.append(binary_label)

    if not video_labels:
        print("No valid testing data was successfully parsed. Ensure videos exist at the specified path.")
        return

    frame_auc = roc_auc_score(frame_labels, frame_preds)
    video_auc = roc_auc_score(video_labels, video_preds)
    eer = compute_eer(video_labels, video_preds)
    
    video_preds_bin = (np.array(video_preds) >= 0.5).astype(int)
    macro_f1 = f1_score(video_labels, video_preds_bin, average='macro')

    results = {
        'Zero-Shot Frame AUC': frame_auc,
        'Zero-Shot Video AUC': video_auc,
        'Zero-Shot EER': eer,
        'Macro F1': macro_f1
    }

    print("\nResults on Celeb-DF v2:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {args.output}")

if __name__ == '__main__':
    main()
