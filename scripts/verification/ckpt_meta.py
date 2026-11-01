import torch

for name in ["models/dual_stream_calibrated.pth", "models/temporal_head_best.pth"]:
    print("==", name, "==")
    ckpt = torch.load(name, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if torch.is_tensor(v):
                print(f"  TENSOR {k}: shape={tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"  dict {k}[{len(v)}]")

                for k2, v2 in v.items():
                    if torch.is_tensor(v2):
                        pass
                    else:
                        print(f"    {k2}: {v2!r}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: <{type(v).__name__}> {v!r}"[:200])
    else:
        print(" type:", type(ckpt))
