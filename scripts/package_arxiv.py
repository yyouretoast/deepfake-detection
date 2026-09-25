"""Package manuscript files into an arXiv-ready tar.gz / zip archive."""

import os
import tarfile
import zipfile


def package_arxiv():
    manuscript_dir = "manuscript"
    figures_dir = os.path.join(manuscript_dir, "figures")
    output_tar = "manuscript_arxiv.tar.gz"
    output_zip = "manuscript_arxiv.zip"

    figure_names = [
        "lifecycle_flaws.pdf",
        "system_architecture.pdf",
        "radial_spectral_profiles.pdf",
        "calibration_reliability.pdf",
        "bayesian_decision_zones.pdf",
        "roc_pr_curves.pdf",
        "qualitative_attention.pdf",
        "robustness_curves.pdf",
    ]

    files_to_pack = [
        ("main.tex", os.path.join(manuscript_dir, "main.tex")),
    ] + [
        (os.path.join("figures", f), os.path.join(figures_dir, f))
        for f in figure_names
    ]

    for _arcname, filepath in files_to_pack:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing required file: {filepath}")

    # Build tar.gz (preferred by arXiv AutoTeX)
    with tarfile.open(output_tar, "w:gz") as tar:
        for arcname, filepath in files_to_pack:
            tar.add(filepath, arcname=arcname)
    print(f"[+] Created {output_tar} ({os.path.getsize(output_tar):,} bytes)")

    # Build zip (convenient for Overleaf inspection)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, filepath in files_to_pack:
            z.write(filepath, arcname=arcname)
    print(f"[+] Created {output_zip} ({os.path.getsize(output_zip):,} bytes)")

    print("\nArchive contents:")
    with tarfile.open(output_tar, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"  - {member.name} ({member.size:,} bytes)")

if __name__ == "__main__":
    package_arxiv()
