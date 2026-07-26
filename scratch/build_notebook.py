import json
import os

NOTEBOOK_PATH = "deepfake_detection_v2_pytorch.ipynb"

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        raise FileNotFoundError(f"Notebook {NOTEBOOK_PATH} not found.")

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source_lines = cell.get("source", [])
            new_lines = []
            for line in source_lines:
                if "torch.fft.rfft2(gray)" in line and 'norm="ortho"' not in line:
                    line = line.replace("torch.fft.rfft2(gray)", 'torch.fft.rfft2(gray, norm="ortho")')
                    updated = True
                new_lines.append(line)
            cell["source"] = new_lines

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    if updated:
        print(f"Successfully updated {NOTEBOOK_PATH} with norm='ortho'.")
    else:
        print(f"No changes required or already updated for {NOTEBOOK_PATH}.")

if __name__ == "__main__":
    update_notebook()
