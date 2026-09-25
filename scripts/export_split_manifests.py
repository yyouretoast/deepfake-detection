"""Export open-science split manifests (CSV and JSON) from deepfake_crops_512/splits.json.

Guarantees full open-science reproducibility for the benchmarks presented in the paper.
Exports:
  splits/train_split.csv
  splits/val_split.csv
  splits/test_split.csv
  splits/splits_manifest_summary.json
"""

import csv
import json
import re
from pathlib import Path


def classify_crop(relative_path: str, label: int) -> tuple[str, str, str, int]:
    norm = relative_path.replace("\\", "/").strip("/")
    parts = norm.split("/")
    video_id = parts[1] if len(parts) > 1 else parts[0]
    frame_name = parts[-1]
    
    # Extract frame index
    m_idx = re.search(r"(\d+)", frame_name)
    frame_idx = int(m_idx.group(1)) if m_idx else 0

    if label == 0:
        if re.match(r"^\d{3}$", video_id):
            domain = "faceforensics"
            generator = "authentic_ffpp"
        elif re.match(r"^\d{5}$", video_id):
            domain = "youtube"
            generator = "authentic_youtube"
        else:
            domain = "celeb_df_v2"
            generator = "authentic_celeb"
    else:
        if "__" in video_id:
            domain = "google_dfd"
            generator = "deepfake_detection"
        elif re.match(r"^\d{3}_\d{3}$", video_id):
            domain = "faceforensics"
            pair_num = int(video_id.split("_")[0])
            if pair_num < 100:
                generator = "deepfakes"
            elif pair_num < 400:
                generator = "face2face"
            elif pair_num < 600:
                generator = "faceswap"
            elif pair_num < 800:
                generator = "neuraltextures"
            else:
                generator = "ffpp_extended"
        else:
            domain = "celeb_df_v2"
            generator = "celeb_synthesis"

    return domain, generator, video_id, frame_idx


def main() -> None:
    splits_path = Path("deepfake_crops_512/splits.json")
    out_dir = Path("splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading splits from {splits_path}...")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    manifest_summary = {}

    for split_name in ["train", "val", "test"]:
        samples = splits.get(split_name, [])
        csv_path = out_dir / f"{split_name}_split.csv"
        print(f"[+] Exporting {split_name} ({len(samples)} crops) -> {csv_path}...")

        counts = {
            "total_crops": len(samples),
            "authentic": 0,
            "synthetic": 0,
            "domains": {},
            "generators": {},
            "unique_videos": set(),
        }

        with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["relative_path", "label", "label_name", "domain", "generator", "video_id", "frame_idx"])

            for rel_path, lbl in samples:
                label_int = int(lbl)
                label_name = "synthetic" if label_int == 1 else "authentic"
                domain, generator, vid_id, f_idx = classify_crop(rel_path, label_int)

                if label_int == 1:
                    counts["synthetic"] += 1
                else:
                    counts["authentic"] += 1

                counts["domains"][domain] = counts["domains"].get(domain, 0) + 1
                counts["generators"][generator] = counts["generators"].get(generator, 0) + 1
                counts["unique_videos"].add(vid_id)

                writer.writerow([rel_path, label_int, label_name, domain, generator, vid_id, f_idx])

        manifest_summary[split_name] = {
            "total_crops": counts["total_crops"],
            "authentic_crops": counts["authentic"],
            "synthetic_crops": counts["synthetic"],
            "unique_videos": len(counts["unique_videos"]),
            "domains": counts["domains"],
            "generators": counts["generators"],
            "csv_manifest": str(csv_path).replace("\\", "/"),
        }

    summary_path = out_dir / "splits_manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(manifest_summary, f_sum, indent=2)

    print(f"[+] Successfully exported split manifests to {out_dir}/")
    print(f"[+] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
