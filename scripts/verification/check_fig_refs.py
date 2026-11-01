import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAN = os.path.join(ROOT, "manuscript")

def check(fname):
    p = os.path.join(MAN, fname)
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    # strip LaTeX comments (unescaped %) before scanning
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"(?<!\\)%.*", "", line)
        lines.append(cleaned)
    text_nc = "\n".join(lines)
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    print(f"== {fname} ==")
    n = 0
    missing = 0
    for m in pattern.finditer(text_nc):
        tgt = m.group(1).strip()
        n += 1
        cands = [os.path.join(ROOT, tgt), os.path.join(MAN, tgt)]
        ok = any(os.path.exists(c) for c in cands)
        if not ok:
            # try matching by stem (different extension)
            for c in cands:
                d = os.path.dirname(c)
                b = os.path.basename(c)
                if os.path.isdir(d):
                    for f2 in os.listdir(d):
                        if os.path.splitext(f2)[0] == b:
                            ok = True
                            cands.append(os.path.join(d, f2))
                            break
                if ok:
                    break
        res = "OK  " if ok else "MISS"
        if not ok:
            missing += 1
        print(f"{res} {tgt}")
    print(f"-- {fname}: {n} refs, {missing} missing")
    return n, missing

tot = 0
tm = 0
for f in ("main.tex", "supplementary.tex"):
    n, m = check(f)
    tot += n
    tm += m
print(f"TOTAL: {tot} refs, {tm} missing")
