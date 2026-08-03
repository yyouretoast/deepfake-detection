import os
import logging
import torch
from src.models.hybrid_detector import build_model
from src.models.onnx_exporter import export_to_onnx, quantize_onnx_model
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)

def main():
    model_path = '/kaggle/working/deepfake_convnext_v2.pth'
    onnx_path = '/kaggle/working/deepfake_convnext_v2.onnx'
    quant_path = '/kaggle/working/deepfake_convnext_v2_int8.onnx'
    
    if not os.path.exists(model_path):
        logging.warning(f'Model checkpoint {model_path} not found.')
        return
        
    model = build_model(use_fft=True, pretrained=False)
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    export_to_onnx(model, save_path=onnx_path, img_size=512)
    quantize_onnx_model(onnx_path, quant_path)
    logging.info('ONNX INT8 export complete.')

if __name__ == '__main__':
    main()
