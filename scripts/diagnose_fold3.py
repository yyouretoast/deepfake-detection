import faulthandler, json, logging, os, sys, time, traceback
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(line_buffering=True)
try: faulthandler.enable()
except Exception: pass
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fold3_diag')
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)

print('=' * 70, flush=True)
print('[PROBE 1/8] Checking Hardware and Environment...', flush=True)
import torch
print(f'  PyTorch Version: {torch.__version__}', flush=True)
print(f'  CUDA Available: {torch.cuda.is_available()}', flush=True)
if torch.cuda.is_available():
    print(f'  Device Name: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'  Device Count: {torch.cuda.device_count()}', flush=True)
    print(f'  VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB', flush=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('  [PASS] Hardware probe completed successfully.', flush=True)

print('=' * 70, flush=True)
print('[PROBE 2/8] Resolving Dataset & Splits Manifest...', flush=True)
from src.dataset.resolver import find_dataset_root, resolve_splits_path
data_root = find_dataset_root()
splits_path = resolve_splits_path(data_root=data_root)
print(f'  Dataset Root: {data_root}', flush=True)
print(f'  Splits Path:  {splits_path}', flush=True)
with open(splits_path, 'r') as f: splits = json.load(f)
train_samples = splits['train']
val_samples = splits['val']
test_samples = splits.get('test', [])
print(f'  Total Splits -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}', flush=True)
print('  [PASS] Dataset manifest loaded successfully.', flush=True)

print('=' * 70, flush=True)
print('[PROBE 3/8] Filtering Fold 3 (FaceSwap) Partitions...', flush=True)
from src.dataset.domains import DomainClassifier
train_loto_samples = [s for s in train_samples if s and len(s) >= 2 and s[0] and not DomainClassifier.matches_holdout(s[0], 'faceswap')]
val_loto_samples = [s for s in val_samples if s and len(s) >= 2 and s[0] and not DomainClassifier.matches_holdout(s[0], 'faceswap')]
eval_target_samples = [s for s in test_samples if s and len(s) >= 2 and s[0] and DomainClassifier.matches_holdout(s[0], 'faceswap')]
if not eval_target_samples:
    eval_target_samples = [s for s in val_samples if s and len(s) >= 2 and s[0] and DomainClassifier.matches_holdout(s[0], 'faceswap')]
real_test_samples = [s for s in test_samples if s and len(s) >= 2 and s[1] == 0]
eval_target_samples.extend(real_test_samples[: min(len(real_test_samples), max(500, len(eval_target_samples)))])
num_fake_train = sum(1 for s in train_loto_samples if s[1] == 1)
num_real_train = len(train_loto_samples) - num_fake_train
num_fake_eval = sum(1 for s in eval_target_samples if s[1] == 1)
num_real_eval = len(eval_target_samples) - num_fake_eval
print(f'  Fold 3 Train: {len(train_loto_samples)} (Fake: {num_fake_train}, Real: {num_real_train})', flush=True)
print(f'  Fold 3 Val:   {len(val_loto_samples)}', flush=True)
print(f'  Fold 3 Eval:  {len(eval_target_samples)} (Held-out FaceSwap: {num_fake_eval}, Real: {num_real_eval})', flush=True)
print('  [PASS] Fold 3 partitioning completed successfully.', flush=True)

print('=' * 70, flush=True)
print('[PROBE 4/8] Testing Direct Image Decoding from Disk...', flush=True)
from src.dataset.datasets import FaceCropDataset
from src.dataset.loader import get_transforms
train_transform, eval_transform = get_transforms(img_size=256, hardened=True)
train_ds = FaceCropDataset(train_loto_samples, data_root, is_train=True, transform=train_transform)
for idx in range(min(10, len(train_ds))):
    img_t, lbl_t, val_t = train_ds[idx]
    print(f'    Train Sample {idx:02d}: shape={tuple(img_t.shape)}, label={lbl_t.item()}, valid={val_t.item()}', flush=True)
eval_ds = FaceCropDataset(eval_target_samples, data_root, is_train=False, transform=eval_transform)
for idx in range(min(5, len(eval_ds))):
    img_t, lbl_t, val_t = eval_ds[idx]
    print(f'    Eval Sample {idx:02d}: shape={tuple(img_t.shape)}, label={lbl_t.item()}, valid={val_t.item()}', flush=True)
print('  [PASS] Direct disk image loading completed without errors.', flush=True)

print('=' * 70, flush=True)
print('[PROBE 5/8] Testing DataLoader Batch 0 Assembly...', flush=True)
from torch.utils.data import DataLoader
fold_seed = 42 + sum(ord(c) for c in 'faceswap')
g_train = torch.Generator()
g_train.manual_seed(fold_seed)
train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=0, drop_last=True, generator=g_train)
batch_iter = iter(train_loader)
b_images, b_labels, b_valid = next(batch_iter)
print(f'  Batch 0 successfully yielded!', flush=True)
print(f'    Images: shape={tuple(b_images.shape)}, min={b_images.min():.2f}, max={b_images.max():.2f}', flush=True)
print(f'    Labels: shape={tuple(b_labels.shape)}, sum_fake={b_labels.sum():.0f}', flush=True)
print(f'    Valid:  shape={tuple(b_valid.shape)}, valid_count={b_valid.sum():.0f}/12', flush=True)
print('  [PASS] DataLoader Batch 0 yielded cleanly.', flush=True)

print('=' * 70, flush=True)
print('[PROBE 6/8] Testing Model Initialization & Forward Pass...', flush=True)
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.training.loss import FocalLossWithLogits
model = HybridDeepfakeDetector(frequency_backbone='resse').to(device)
pos_weight = torch.tensor([min(float(num_real_train / max(1, num_fake_train)), 3.0)], device=device)
criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pos_weight)
b_images = b_images.to(device)
b_labels = b_labels.to(device).unsqueeze(1)
b_valid = b_valid.to(device).unsqueeze(1)
model.train()
with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
    outputs, aux_outputs = model(b_images, return_aux=True)
    loss_main = criterion(outputs, b_labels)
    loss_aux = criterion(aux_outputs, b_labels)
    loss = (loss_main * b_valid).sum() / b_valid.sum().clamp(min=1.0) + 0.3 * (loss_aux * b_valid).sum() / b_valid.sum().clamp(min=1.0)
print(f'  Forward Pass: outputs={tuple(outputs.shape)}, aux={tuple(aux_outputs.shape)}, loss={loss.item():.4f}', flush=True)
print('  [PASS] Model forward pass completed cleanly.', flush=True)

print('=' * 70, flush=True)
print('[PROBE 7/8] Testing Autograd Backward Pass and Optimizer Step...', flush=True)
from src.training.optimization import get_differential_param_groups
optimizer = torch.optim.AdamW(get_differential_param_groups(model))
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
print('  [PASS] Autograd backward pass & optimizer step completed cleanly.', flush=True)

print('=' * 70, flush=True)
print('[PROBE 8/8] Testing Holdout (FaceSwap) Zero-Shot Evaluation Loop...', flush=True)
eval_loader = DataLoader(eval_ds, batch_size=12, shuffle=False, num_workers=0)
model.eval()
all_logits, all_targets = [], []
with torch.no_grad():
    for e_imgs, e_lbls, _ in eval_loader:
        e_imgs = e_imgs.to(device)
        out = model(e_imgs)
        all_logits.extend(out.cpu().reshape(-1).tolist())
        all_targets.extend(e_lbls.reshape(-1).tolist())
        if len(all_logits) >= 24: break
print(f'  Evaluated {len(all_logits)} zero-shot FaceSwap samples successfully!', flush=True)
print('  [PASS] Holdout evaluation loop functions cleanly.', flush=True)

print('=' * 70, flush=True)
print('ALL 8 PROBE CHECKS PASSED WITH ZERO CRASHES!', flush=True)
print('=' * 70, flush=True)