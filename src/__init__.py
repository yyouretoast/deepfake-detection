from src.config import load_config
from src.models.hybrid_detector import HybridDeepfakeDetector, build_model

try:
    from src.models.onnx_exporter import ONNXDeepfakePredictor, export_to_onnx, quantize_onnx_model
except ImportError:
    ONNXDeepfakePredictor, export_to_onnx, quantize_onnx_model = None, None, None

try:
    from src.dataset.loader import DeepfakeDataset, group_video_split, extract_video_id, extract_identities, perform_graph_split, create_dataloaders
except ImportError:
    DeepfakeDataset, group_video_split, extract_video_id, extract_identities, perform_graph_split, create_dataloaders = None, None, None, None, None, None

    DynamicFaceCropper = None

try:
    from src.explainability.gradcam import PyTorchGradCAM, overlay_cam
except ImportError:
    PyTorchGradCAM, overlay_cam = None, None

__all__ = [
    "load_config",
    "HybridDeepfakeDetector",
    "build_model",
    "ONNXDeepfakePredictor",
    "export_to_onnx",
    "quantize_onnx_model",
    "DeepfakeDataset",
    "group_video_split",
    "extract_video_id",
    "extract_identities",
    "perform_graph_split",
    "create_dataloaders",
    "DynamicFaceCropper",
    "PyTorchGradCAM",
    "overlay_cam",
]
