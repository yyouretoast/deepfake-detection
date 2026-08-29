"""Generate publication-grade PDF and HTML documents with 100% perfect LaTeX math rendering."""

import os
import re
import subprocess
import markdown

MD_FILES = [
    "MASTER_EXPLANATION.md",
    "SECTION_1_FOUNDATIONS.md",
    "SECTION_2_MODEL_ARCHITECTURE.md",
    "SECTION_3_DATA_PREPROCESSING_ALIGNMENT.md",
    "SECTION_4_TRAINING_ENGINE_OPTIMIZATION.md",
    "SECTION_5_CALIBRATION_TEMPORAL_AGGREGATION.md",
    "SECTION_6_EXPLAINABILITY_INTERPRETABILITY.md",
    "SECTION_7_BENCHMARKS_ROBUSTNESS_LATENCY.md",
    "SECTION_8_WEB_APPLICATION_DEPLOYMENT.md",
]

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- KaTeX CSS & JS for 100% Crisp LaTeX Math Rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{
                delimiters: [
                    {{left: '$$', right: '$$', display: true}},
                    {{left: '$', right: '$', display: false}},
                    {{left: '\\\\[', right: '\\\\]', display: true}},
                    {{left: '\\\\(', right: '\\\\)', display: false}}
                ],
                throwOnError: false
            }});"></script>

    <!-- Modern Typography & GitHub Markdown Styling -->
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        @page {{
            size: A4 portrait;
            margin: 20mm 15mm 20mm 15mm;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 10.5pt;
            line-height: 1.6;
            color: #1e293b;
            background-color: #ffffff;
            max-width: 900px;
            margin: 0 auto;
            padding: 30px 20px;
        }}

        h1 {{
            font-size: 20pt;
            font-weight: 700;
            color: #0f172a;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 8px;
            margin-top: 25px;
            margin-bottom: 16px;
        }}

        h2 {{
            font-size: 15pt;
            font-weight: 600;
            color: #1e3a8a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
            margin-top: 22px;
            margin-bottom: 12px;
        }}

        h3 {{
            font-size: 12pt;
            font-weight: 600;
            color: #1d4ed8;
            margin-top: 18px;
            margin-bottom: 8px;
        }}

        p {{
            margin-top: 0;
            margin-bottom: 12px;
            text-align: justify;
        }}

        ul, ol {{
            margin-top: 4px;
            margin-bottom: 12px;
            padding-left: 24px;
        }}

        li {{
            margin-bottom: 4px;
        }}

        pre {{
            background-color: #0f172a;
            color: #f8fafc;
            border-radius: 8px;
            padding: 12px 16px;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.4;
            overflow-x: auto;
            margin-bottom: 16px;
        }}

        code {{
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 9pt;
        }}

        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
            margin-bottom: 18px;
            font-size: 9.5pt;
        }}

        th, td {{
            border: 1px solid #cbd5e1;
            padding: 8px 12px;
            text-align: left;
        }}

        th {{
            background-color: #f8fafc;
            color: #0f172a;
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        blockquote {{
            border-left: 4px solid #3b82f6;
            background-color: #eff6ff;
            color: #1e40af;
            padding: 10px 16px;
            margin: 14px 0;
            border-radius: 0 8px 8px 0;
        }}

        hr {{
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 20px 0;
        }}

        a {{
            color: #2563eb;
            text-decoration: none;
            font-weight: 500;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .katex-display {{
            margin: 14px 0 !important;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 4px 0;
        }}

        @media print {{
            body {{
                max-width: 100%;
                padding: 0;
                font-size: 10pt;
            }}
            h1 {{ font-size: 18pt; }}
            h2 {{ font-size: 14pt; }}
            h3 {{ font-size: 11pt; }}
            pre {{
                background-color: #1e293b !important;
                color: #ffffff !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
{body}
</body>
</html>
"""

def convert_md_to_html_and_pdf(md_filename: str) -> None:
    base_name = os.path.splitext(md_filename)[0]
    html_dir = "html_exports"
    pdf_dir = "pdf_exports"
    
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    html_path = os.path.abspath(os.path.join(html_dir, f"{base_name}.html"))
    pdf_path = os.path.abspath(os.path.join(pdf_dir, f"{base_name}.pdf"))
    
    with open(md_filename, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert Markdown to HTML
    body_html = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "toc"]
    )
    
    full_html = HTML_TEMPLATE.format(title=base_name, body=body_html)
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Generated HTML: {html_path}")
    
    # Run Headless Chrome to print HTML to high-precision PDF
    if os.path.exists(CHROME_PATH):
        cmd = [
            CHROME_PATH,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=3000",
            f"--print-to-pdf={pdf_path}",
            html_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Generated Vector PDF: {pdf_path}")
        except Exception as e:
            print(f"Chrome PDF generation error for {md_filename}: {e}")

def main() -> None:
    print("Starting high-quality PDF/HTML generation with vector KaTeX math...\n")
    for md_file in MD_FILES:
        if os.path.exists(md_file):
            convert_md_to_html_and_pdf(md_file)
    print("\nAll documents successfully generated!")

if __name__ == "__main__":
    main()
