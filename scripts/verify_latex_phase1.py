import re
import sys

def verify_latex(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []
    
    # 1. Check environment balances
    begins = re.findall(r"\\begin\{([a-zA-Z*]+)\}", content)
    ends = re.findall(r"\\end\{([a-zA-Z*]+)\}", content)
    
    if len(begins) != len(ends):
        errors.append(f"Environment count mismatch: {len(begins)} begins vs {len(ends)} ends")
    
    stack = []
    for line_idx, line in enumerate(content.splitlines(), 1):
        # strip comments
        line_clean = re.sub(r"(?<!\\)%.*", "", line)
        
        b_matches = re.findall(r"\\begin\{([a-zA-Z*]+)\}", line_clean)
        e_matches = re.findall(r"\\end\{([a-zA-Z*]+)\}", line_clean)
        
        for b in b_matches:
            stack.append((b, line_idx))
        for e in e_matches:
            if not stack:
                errors.append(f"Line {line_idx}: unexpected \\end{{{e}}} with empty stack")
            else:
                last_b, last_line = stack.pop()
                if last_b != e:
                    errors.append(f"Line {line_idx}: \\end{{{e}}} does not match \\begin{{{last_b}}} from line {last_line}")

    if stack:
        for b, l in stack:
            errors.append(f"Unclosed \\begin{{{b}}} from line {l}")

    # 2. Check curly brace matching outside comments
    brace_depth = 0
    in_math_dollar = False
    for line_idx, line in enumerate(content.splitlines(), 1):
        line_clean = re.sub(r"(?<!\\)%.*", "", line)
        for char_idx, char in enumerate(line_clean):
            if char == '{' and (char_idx == 0 or line_clean[char_idx-1] != '\\'):
                brace_depth += 1
            elif char == '}' and (char_idx == 0 or line_clean[char_idx-1] != '\\'):
                brace_depth -= 1
                if brace_depth < 0:
                    errors.append(f"Line {line_idx}: Extra closing brace '}}'")
                    brace_depth = 0

    if brace_depth > 0:
        errors.append(f"Unbalanced curly braces: {brace_depth} unclosed '{{'")

    # 3. Check citations
    cites = set()
    for m in re.findall(r"\\cite\{([^}]+)\}", content):
        for k in m.split(","):
            cites.add(k.strip())
            
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", content))
    
    missing_citations = cites - bibitems
    if missing_citations:
        errors.append(f"Missing bibitems for citations: {missing_citations}")

    # 4. Check references
    refs = set(re.findall(r"\\ref\{([^}]+)\}", content))
    labels = set(re.findall(r"\\label\{([^}]+)\}", content))
    missing_labels = refs - labels
    if missing_labels:
        errors.append(f"Missing labels for references: {missing_labels}")

    # 5. Check mandatory numbers from the ledger
    ledger = [
        "1,258", "38", "91", "13,444", "1,120",
        "0.8719", "0.8818", "18.98", "5.75",
        "0.8966", "4.2880", "0.2597", "0.0785", "69.8",
        "60.9", "16.41", "0.4678"
    ]
    missing_numbers = [num for num in ledger if num not in content]
    if missing_numbers:
        errors.append(f"Missing mandatory ledger numbers: {missing_numbers}")

    print("Verification Summary:")
    print(f"Total lines: {len(content.splitlines())}")
    print(f"Total words: {len(content.split())}")
    print(f"Total citations found: {len(cites)} (all defined: {len(cites - missing_citations)}/{len(cites)})")
    print(f"Total labels found: {len(labels)}")
    print(f"Total references checked: {len(refs)}")
    print(f"Ledger numbers checked: {len(ledger) - len(missing_numbers)}/{len(ledger)} found")
    
    if errors:
        print("\nERRORS DETECTED:")
        for err in errors:
            print(f"  [!] {err}")
        return False
    else:
        print("\nALL INVARIANTS PASSED PERFECTLY!")
        return True

if __name__ == "__main__":
    success = verify_latex("manuscript/main.tex")
    sys.exit(0 if success else 1)
