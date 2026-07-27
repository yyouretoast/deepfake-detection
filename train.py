from typing import Optional
import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset.loader import build_dataloaders, get_eval_transforms
from src.models.hybrid_detector import build_model
from src.models.onnx_exporter import export_to_onnx
from src.training.trainer import TwoPhaseTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake Detector Training Engine (PyTorch 2.x)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path to cropped frames dataset")
    parser.add_argument("--epochs_p1", type=int, default=None, help="Phase 1 head warmup epochs")
    parser.add_argument("--epochs_p2", type=int, default=None, help="Phase 2 LLRD fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--sequence", action="store_true", help="Enable 5D video sequence dataset loading")
    parser.add_argument("--save_path", type=str, default="models/deepfake_convnext_v2.pt", help="Checkpoint save path")
    parser.add_argument("--export_onnx", action="store_true", help="Export model to ONNX format after training")

    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config.setdefault("dataset", {})["cropped_frames_dir"] = args.data_dir
    if args.epochs_p1:
        config.setdefault("training", {})["epochs_phase1"] = args.epochs_p1
    if args.epochs_p2:
        config.setdefault("training", {})["epochs_phase2"] = args.epochs_p2
    if args.batch_size:
        config.setdefault("dataset", {})["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Training on Device: %s", device)

    # 1. Build DataLoaders via Graph-Connected Identity Split
    dataloaders = build_dataloaders(config=config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 2. Build Model Architecture
    model = build_model(use_fft=True, device=device, pretrained=True)

    # 3. Execute Two-Phase Training Pipeline
    trainer = TwoPhaseTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device)
    checkpoint_data, opt_thresh, metrics = trainer.train()

    # 4. Save Metadata-Embedded Checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save(checkpoint_data, args.save_path)
    logger.info("Saved verified checkpoint to: %s | Opt Threshold T*: %.4f | Val AUC: %.4f",
                args.save_path, opt_thresh, metrics.get("best_val_auc", 0.0))

    # 5. Optional ONNX Export
    if args.export_onnx:
        onnx_path = os.path.splitext(args.save_path)[0] + ".onnx"
        export_to_onnx(model=model, save_path=onnx_path, img_size=config.get("dataset", {}).get("img_size", 256))
        logger.info("Exported ONNX model to: %s", onnx_path)

if __name__ == "__main__":
    main()
