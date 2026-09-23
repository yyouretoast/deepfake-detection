import sys

terms = ['static', 'snr', 'delta', 'table 5', 'ablat', 'baseline', 'gate']

output_lines = []
with open('manuscript/main.tex', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        low = line.lower()
        matched = [t for t in terms if t in low]
        if matched:
            output_lines.append(f"{i}: {matched} -> {line.strip()}\n")

with open('results/dangling_report.txt', 'w', encoding='utf-8') as out:
    out.writelines(output_lines)

print(f"Done. Found {len(output_lines)} matches.")
