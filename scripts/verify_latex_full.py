import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def verify_latex_full(file_path, results_data=None):
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
        line_clean = re.sub(r"(?<!\\)%.*", "", line)
        
        # Tokenize \begin and \end preserving character order on the line
        tokens = []
        for m in re.finditer(r"\\(begin|end)\{([a-zA-Z*]+)\}", line_clean):
            tokens.append((m.start(), m.group(1), m.group(2)))
        tokens.sort(key=lambda x: x[0])
        
        for _, action, env in tokens:
            if action == "begin":
                stack.append((env, line_idx))
            else:
                if not stack:
                    errors.append(f"Line {line_idx}: unexpected \\end{{{env}}} with empty stack")
                else:
                    last_env, last_line = stack.pop()
                    if last_env != env:
                        errors.append(f"Line {line_idx}: \\end{{{env}}} does not match \\begin{{{last_env}}} from line {last_line}")

    if stack:
        for b, l in stack:
            errors.append(f"Unclosed \\begin{{{b}}} from line {l}")

    # 2. Check curly brace matching outside comments
    brace_depth = 0
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
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", content))
    labels = set(re.findall(r"\\label\{([^}]+)\}", content))
    missing_labels = refs - labels
    if missing_labels:
        errors.append(f"Missing labels for references: {missing_labels}")

    # 5. Check tables against generator and numbers against results JSON
    if results_data:
        import difflib

        from scripts.generate_manuscript_tables import (
            extract_table,
            generate_table1,
            generate_table2,
            generate_table3,
            generate_table4,
            generate_table_ablations,
            generate_table_latency,
            normalize_tex_table,
        )

        table_generators = [
            ("tab:main_benchmark", generate_table1),
            ("tab:subdomain_breakdown", generate_table2),
            ("tab:robustness_benchmarks", generate_table4),
            ("tab:ablations", generate_table_ablations),
            ("tab:latency", generate_table_latency),
        ]
        if results_data.get("loto_cross_generator_benchmark"):
            table_generators.append(("tab:loto_results", generate_table3))

        print("\n--- Verifying LaTeX tables against generator output ---")
        for label, gen_fn in table_generators:
            expected = gen_fn(results_data)
            actual = extract_table(content, label)
            if actual is None:
                errors.append(f"Table environment with label '{label}' not found in {file_path}")
                continue

            norm_expected = normalize_tex_table(expected)
            norm_actual = normalize_tex_table(actual)
            if norm_expected != norm_actual:
                diff = list(difflib.unified_diff(
                    norm_expected.splitlines(keepends=True),
                    norm_actual.splitlines(keepends=True),
                    fromfile=f"expected_{label}",
                    tofile=f"manuscript_{label}",
                ))
                print(f"\n[DIFF DETECTED in table '{label}']:")
                for d in diff[:25]:
                    print(d, end="")
                if len(diff) > 25:
                    print(f"... ({len(diff) - 25} more diff lines)")
                errors.append(
                    f"Table '{label}' in manuscript does not match table generated from results JSON! "
                    f"Run 'python scripts/generate_manuscript_tables.py --update' to synchronize."
                )
            else:
                print(f"  [+] Table '{label}' matches generator exactly.")
    else:
        # Static ledger fallback check
        ledger = [
            "37,104", "35,016", "26,981", "7,609", "19,372", "2,250",
            "3,092", "2,918", "99,101",
            "0.8719", "0.8818", "18.98", "5.75",
            "0.8966", "4.2880", "0.2597", "0.0785", "69.8",
            "60.9", "16.41", "0.4678",
            "0.8248", "0.8627", "0.7834", "0.7172", "0.7449", "0.5946"
        ]
        missing_numbers = [num for num in ledger if num not in content]
        if missing_numbers:
            print(f"  [i] Note: {len(missing_numbers)} static ledger numbers not in text (use --results to verify live run numbers): {missing_numbers}")

    # 6. Check Section headings
    required_sections = [
        r"\section{Introduction}",
        r"\section{Related Work}",
        r"\section{Proposed Forensic Architecture}",
        r"\section{Experiments and Empirical Benchmarks}",
        r"\section{Discussion, Forensic Degradation Dynamics, and Limitations}",
        r"\section{Conclusion}"
    ]
    missing_sections = [s for s in required_sections if s not in content]
    if missing_sections:
        errors.append(f"Missing required sections: {missing_sections}")

    print("\nFull Manuscript Verification Summary:")
    print(f"Total lines: {len(content.splitlines())}")
    print(f"Total words: {len(content.split())}")
    print(f"Total citations found: {len(cites)} (all defined: {len(cites - missing_citations)}/{len(cites)})")
    print(f"Total labels found: {len(labels)}")
    print(f"Total references checked: {len(refs)} (all defined: {len(refs - missing_labels)}/{len(refs)})")
    print(f"Required sections checked: {len(required_sections) - len(missing_sections)}/{len(required_sections)} found")
    
    if errors:
        print("\nERRORS DETECTED:")
        for err in errors:
            print(f"  [!] {err}")
        return False
    else:
        print("\nALL MANUSCRIPT INVARIANTS PASSED PERFECTLY!")
        return True

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Verify full LaTeX manuscript invariants and results.")
    parser.add_argument("target", nargs="?", default="manuscript/main.tex", help="Path to main.tex")
    parser.add_argument("--results", type=str, default=None, help="Path to results JSON file")
    args = parser.parse_args()

    results_data = None
    if args.results and os.path.exists(args.results):
        with open(args.results, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    elif os.path.exists("results/release1_results.json"):
        with open("results/release1_results.json", "r", encoding="utf-8") as f:
            results_data = json.load(f)

    success = verify_latex_full(args.target, results_data=results_data)
    sys.exit(0 if success else 1)
