import torch

from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector

print("== dual_stream_calibrated.pth ==")
ckpt = torch.load("models/dual_stream_calibrated.pth", map_location="cpu", weights_only=False)
if isinstance(ckpt, dict):
    meta = {k: (f"dict[{len(v)}]" if isinstance(v, dict) else v) for k, v in ckpt.items()}
    for k, v in meta.items():
        print(f"  {k}: {v}")
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
else:
    sd = ckpt.state_dict()

m = HybridDeepfakeDetector(pretrained=False, use_fft_branch=True)
missing, unexpected = m.load_state_dict(sd, strict=False)
print("missing:", len(missing), missing[:5])
print("unexpected:", len(unexpected), unexpected[:5])
n = sum(p.numel() for p in m.parameters())
spatial = (sum(p.numel() for p in m.spatial_backbone.parameters())
           + sum(p.numel() for p in m.spatial_norm.parameters())
           + sum(p.numel() for p in m.spatial_fc.parameters()))
tower = sum(p.numel() for p in m.freq_tower.parameters())
print(f"TOTAL dual-stream params: {n:,} ({n/1e6:.2f}M)")
print(f"spatial stream: {spatial:,} ({spatial/1e6:.2f}M)")
print(f"ResSE tower: {tower:,} ({tower/1e6:.2f}M)")
print(f"rest (gate/classifier/srm/bayar): {n - spatial - tower:,} ({(n-spatial-tower)/1e6:.2f}M)")

print("\n== temporal_head_best.pth ==")
t = torch.load("models/temporal_head_best.pth", map_location="cpu", weights_only=False)
if isinstance(t, dict):
    for k, v in t.items():
        if isinstance(v, dict):
            print(f"  {k}: dict[{len(v)}]")
        else:
            print(f"  {k}: {v}")
    tsd = t.get("state_dict", t.get("model_state_dict", t))
else:
    tsd = t.state_dict()
head = BiGRUTemporalDetector(embed_dim=512, hidden_dim=256, dropout=0.2, use_deltas=True, use_max_pool=True)
try:
    head.load_state_dict(tsd, strict=True)
    ok = "strict OK"
except Exception as e:
    print("strict tempered:", e)
    r = head.load_state_dict(tsd, strict=False)
    ok = f"non-strict missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}"
n2 = sum(p.numel() for p in head.parameters())
print(f"temporal head params: {n2:,} ({n2/1e6:.2f}M) | load: {ok}")
