"""Convert all project explanatory markdown files to publication-styled PDF documents."""

import os
import re
import markdown
from xhtml2pdf import pisa

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

CSS_STYLES = """
@page {
    size: a4 portrait;
    margin-top: 1.8cm;
    margin-bottom: 1.8cm;
    margin-left: 1.5cm;
    margin-right: 1.5cm;
}

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.45;
    color: #1e293b;
}

h1 {
    font-size: 16pt;
    color: #0f172a;
    border-bottom: 1.5px solid #2563eb;
    padding-bottom: 3px;
    margin-top: 12px;
    margin-bottom: 8px;
}

h2 {
    font-size: 13pt;
    color: #1e3a8a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    margin-top: 12px;
    margin-bottom: 6px;
}

h3 {
    font-size: 10.5pt;
    color: #1e40af;
    margin-top: 8px;
    margin-bottom: 4px;
}

p {
    margin-bottom: 6px;
}

ul, ol {
    margin-top: 3px;
    margin-bottom: 6px;
    padding-left: 18px;
}

li {
    margin-bottom: 2px;
}

pre {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    padding: 6px;
    font-family: Courier, monospace;
    font-size: 7.5pt;
    line-height: 1.2;
    margin-bottom: 8px;
}

code {
    font-family: Courier, monospace;
    background-color: #f1f5f9;
    color: #0f172a;
    font-size: 8pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    margin-bottom: 10px;
    font-size: 8pt;
}

th, td {
    border: 0.5px solid #94a3b8;
    padding: 4px 6px;
    text-align: left;
}

th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: bold;
}

blockquote {
    border-left: 2.5px solid #6366f1;
    background-color: #eef2ff;
    padding: 5px 10px;
    margin: 6px 0;
    font-style: italic;
    color: #3730a3;
}

hr {
    border-top: 0.5px solid #e2e8f0;
    margin: 10px 0;
}
"""

def clean_markdown_for_pdf(text: str) -> str:
    # Remove emojis or characters that may cause font glyph issues in basic Helvetica
    # Strip local file URLs to clean text
    text = re.sub(r'\[([^\]]+)\]\(file:///[^\)]+\)', r'**\1**', text)
    text = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'**\1**', text)
    # Simplify raw math $$...$$ blocks into clean pre blocks for PDF rendering
    return text

def convert_file(md_filename: str, output_dir: str = "pdf_exports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    pdf_filename = os.path.join(output_dir, os.path.splitext(md_filename)[0] + ".pdf")
    
    if not os.path.exists(md_filename):
        print(f"File not found: {md_filename}")
        return ""

    with open(md_filename, "r", encoding="utf-8") as f:
        raw_md = f.read()

    cleaned_md = clean_markdown_for_pdf(raw_md)

    # Convert Markdown to HTML
    html_body = markdown.markdown(
        cleaned_md,
        extensions=["tables", "fenced_code", "toc"]
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{os.path.splitext(md_filename)[0]}</title>
    <style>
    {CSS_STYLES}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(pdf_filename, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        print(f"Error converting {md_filename} to PDF: {pisa_status.err}")
        return ""
    else:
        print(f"Successfully generated: {pdf_filename}")
        return pdf_filename

def main() -> None:
    output_dir = "pdf_exports"
    generated = []
    for md_file in MD_FILES:
        pdf_path = convert_file(md_file, output_dir=output_dir)
        if pdf_path:
            generated.append(pdf_path)

    print(f"\nCompleted! Generated {len(generated)} PDF documents in '{output_dir}/'.")

if __name__ == "__main__":
    main()
