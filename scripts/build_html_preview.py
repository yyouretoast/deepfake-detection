"""
Builds an interactive, publication-quality HTML preview of manuscript/main.tex.
Includes MathJax for equation rendering, embedded PNG figures, styled IEEE tables,
section navigation, and print stylesheet for browser 'Save as PDF'.
"""
import re
from pathlib import Path


def extract_braced(text, start_idx):
    """Extracts content inside the first { } starting at or after start_idx, handling nested braces."""
    brace_start = text.find('{', start_idx)
    if brace_start == -1:
        return "", start_idx
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == '{' and (i == 0 or text[i-1] != '\\'):
            depth += 1
        elif text[i] == '}' and (i == 0 or text[i-1] != '\\'):
            depth -= 1
            if depth == 0:
                return text[brace_start+1:i], i + 1
    return "", len(text)

def build_html():
    tex_path = Path("manuscript/main.tex")
    html_path = Path("manuscript/paper_preview.html")
    
    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    # Preserve \% before stripping comments
    tex = tex.replace(r'\%', '__PERCENT_ESCAPED__')
    tex = re.sub(r'(?<!\\)%.*?\n', '\n', tex)
    tex = tex.replace('__PERCENT_ESCAPED__', '%')

    # Extract Title
    title_idx = tex.find(r'\title')
    title_raw, _ = extract_braced(tex, title_idx) if title_idx != -1 else ("", 0)
    title = re.sub(r'\\textbf\{\\Large\s*(.*?)\}', r'\1', title_raw, flags=re.DOTALL)
    title = title.replace('\\\\', '<br/>').strip()

    # Extract Authors
    author_idx = tex.find(r'\author')
    author_raw, _ = extract_braced(tex, author_idx) if author_idx != -1 else ("", 0)
    author_clean = re.sub(r'\\textbf\{(.*?)\}', r'<strong>\1</strong>', author_raw)
    author_clean = re.sub(r'\\textsuperscript\{(.*?)\}', r'<sup>\1</sup>', author_clean)
    author_clean = re.sub(r'\\texttt\{(.*?)\}', r'<code>\1</code>', author_clean)
    author_clean = re.sub(r'\\\[.*?\]', '<br/>', author_clean)
    author_clean = author_clean.replace('\\\\', '<br/>').strip()

    # Extract Abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.DOTALL)
    abstract_text = abstract_match.group(1).strip() if abstract_match else ""

    # Extract Keywords
    keywords_match = re.search(r'\\textbf\{\\textit\{Keywords\}\}---(.*?)(?=\n\n|\n\\section)', tex, re.DOTALL)
    keywords_text = keywords_match.group(1).strip() if keywords_match else ""

    # Bibliography items
    bib_items = {}
    bib_list = []
    bib_counter = 1
    for match in re.finditer(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})', tex, re.DOTALL):
        key = match.group(1).strip()
        raw_entry = match.group(2).strip()
        clean_entry = re.sub(r'\\textit\{(.*?)\}', r'<em>\1</em>', raw_entry)
        clean_entry = re.sub(r'\\textbf\{(.*?)\}', r'<strong>\1</strong>', clean_entry)
        clean_entry = re.sub(r'\\url\{([^}]+)\}', r'<a href="\1" target="_blank">\1</a>', clean_entry)
        clean_entry = clean_entry.replace('~', ' ')
        bib_items[key] = bib_counter
        bib_list.append((bib_counter, key, clean_entry))
        bib_counter += 1

    # Extract Body (between \maketitle and \begin{thebibliography})
    body_match = re.search(r'\\maketitle(.*?)\\begin\{thebibliography\}', tex, re.DOTALL)
    body = body_match.group(1) if body_match else tex

    # Remove abstract, footnotes, keywords from body
    body = re.sub(r'\\def\\thefootnote.*?\\def\\thefootnote\{\\arabic\{footnote\}\}', '', body, flags=re.DOTALL)
    body = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '', body, flags=re.DOTALL)
    body = re.sub(r'\\textbf\{\\textit\{Keywords\}\}.*?(?=\\section)', '', body, flags=re.DOTALL)

    # Convert Figures
    def convert_figure(fig_match):
        code = fig_match.group(0)
        img_match = re.search(r'\\includegraphics(?:\[.*?\])?\{figures/(.*?)\.(?:pdf|png)\}', code)
        fig_name = img_match.group(1) if img_match else ""
        
        caption_idx = code.find(r'\caption')
        caption_raw, _ = extract_braced(code, caption_idx) if caption_idx != -1 else ("", 0)
        caption = caption_raw.replace('\n', ' ')
        caption = re.sub(r'\\textbf\{(.*?)\}', r'<strong>\1</strong>', caption)
        caption = re.sub(r'\\textit\{(.*?)\}', r'<em>\1</em>', caption)
        caption = re.sub(r'\\texttt\{(.*?)\}', r'<code>\1</code>', caption)
        caption = re.sub(r'\\ref\{(.*?)\}', r'<a href="#\1">\1</a>', caption)
        caption = re.sub(r'\\etal\b', r'<em>et al.</em>', caption)
        
        return f"""
        <figure class="paper-figure" id="{fig_name}">
            <img src="figures/{fig_name}.png" alt="{fig_name}" class="figure-img" />
            <figcaption><strong>Figure:</strong> {caption}</figcaption>
        </figure>
        """

    body = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', convert_figure, body, flags=re.DOTALL)

    # Convert Tables
    def convert_table(tab_match):
        code = tab_match.group(0)
        caption_idx = code.find(r'\caption')
        caption_raw, _ = extract_braced(code, caption_idx) if caption_idx != -1 else ("", 0)
        caption = caption_raw.replace('\n', ' ')
        caption = re.sub(r'\\textbf\{(.*?)\}', r'<strong>\1</strong>', caption)
        caption = re.sub(r'\\texttt\{(.*?)\}', r'<code>\1</code>', caption)
        
        tabular_match = re.search(r'\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}', code, re.DOTALL)
        if not tabular_match:
            return ""
        
        tab_code = tabular_match.group(2).strip()
        rows = [r.strip() for r in tab_code.split('\\\\') if r.strip()]
        
        html_rows = []
        is_first = True
        for row in rows:
            if not row or row.startswith(('\\toprule', '\\bottomrule')):
                continue
            row = row.replace('\\midrule', '').strip()
            if not row:
                continue
            cells = [c.strip() for c in row.split('&')]
            tag = "th" if is_first else "td"
            cell_htmls = []
            for c in cells:
                c_clean = re.sub(r'\\textbf\{(.*?)\}', r'<strong>\1</strong>', c)
                c_clean = re.sub(r'\\textit\{(.*?)\}', r'<em>\1</em>', c_clean)
                c_clean = re.sub(r'\\texttt\{(.*?)\}', r'<code>\1</code>', c_clean)
                c_clean = re.sub(r'\\multicolumn\{(\d+)\}\{([^}]+)\}\{(.*?)\}', r'\3', c_clean)
                cell_htmls.append(f"<{tag}>{c_clean}</{tag}>")
            html_rows.append(f"<tr>{''.join(cell_htmls)}</tr>")
            if is_first:
                is_first = False
        
        return f"""
        <div class="table-container">
            <div class="table-caption">{caption}</div>
            <table class="academic-table">
                <tbody>
                    {''.join(html_rows)}
                </tbody>
            </table>
        </div>
        """

    body = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', convert_table, body, flags=re.DOTALL)

    # Convert Citations \cite{key1, key2}
    def replace_cite(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        nums = [f'<a href="#ref-{k}" class="cite-ref">[{bib_items.get(k, "?")}]</a>' for k in keys]
        return "<sup>" + ",".join(nums) + "</sup>"
    
    body = re.sub(r'\\cite\{([^}]+)\}', replace_cite, body)

    # Sequential Section, Subsection, and Subsubsection Replacement
    current_sec = 0
    current_subsec = 0
    current_subsubsec = 0

    def replace_heading(m):
        nonlocal current_sec, current_subsec, current_subsubsec
        h_type = m.group(1)
        h_title = m.group(2).strip()

        if h_type == 'section':
            current_sec += 1
            current_subsec = 0
            current_subsubsec = 0
            return f'<h2 id="sec-{current_sec}" class="section-title"><span class="sec-num">{current_sec}.</span> {h_title}</h2>'
        elif h_type == 'subsection':
            current_subsec += 1
            current_subsubsec = 0
            return f'<h3 id="subsec-{current_sec}-{current_subsec}" class="subsection-title"><span class="sec-num">{current_sec}.{current_subsec}</span> {h_title}</h3>'
        elif h_type == 'subsubsection':
            current_subsubsec += 1
            return f'<h4 class="subsubsection-title"><span class="sec-num">{current_sec}.{current_subsec}.{current_subsubsec}</span> {h_title}</h4>'
        return m.group(0)

    body = re.sub(r'\\(section|subsection|subsubsection)\*?\{([^}]+)\}', replace_heading, body)

    # Paragraphs, formatting, links
    body = re.sub(r'\\paragraph\{([^}]+)\}', r'<p class="paragraph-header"><strong>\1</strong> ', body)
    body = re.sub(r'\\textbf\{([^}]+)\}', r'<strong>\1</strong>', body)
    body = re.sub(r'\\textit\{([^}]+)\}', r'<em>\1</em>', body)
    body = re.sub(r'\\texttt\{([^}]+)\}', r'<code>\1</code>', body)
    body = re.sub(r'\\etal\b', r'<em>et al.</em>', body)
    body = re.sub(r'\\eg\b', r'<em>e.g.</em>', body)
    body = re.sub(r'\\ie\b', r'<em>i.e.</em>', body)
    body = re.sub(r'\\url\{([^}]+)\}', r'<a href="\1" target="_blank">\1</a>', body)
    body = re.sub(r'\\ref\{([^}]+)\}', r'<a href="#\1" class="in-ref">\1</a>', body)
    body = re.sub(r'\\label\{([^}]+)\}', r'<a id="\1"></a>', body)

    # Clean environments
    body = re.sub(r'\\begin\{itemize\}', '<ul class="paper-list">', body)
    body = re.sub(r'\\end\{itemize\}', '</ul>', body)
    body = re.sub(r'\\begin\{enumerate\}', '<ol class="paper-list">', body)
    body = re.sub(r'\\end\{enumerate\}', '</ol>', body)
    body = re.sub(r'\\item\s*(.*?)(?=\\item|</ul>|</ol>|\n\n)', r'<li>\1</li>', body, flags=re.DOTALL)

    # Format Paragraphs: wrap non-tagged lines into <p>
    paragraphs = body.split('\n\n')
    cleaned_p = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith(('<h', '<div', '<figure', '<ul', '<ol', '<table')):
            cleaned_p.append(p)
        elif p.startswith((r'\begin{equation', r'\begin{align', '$$')):
            cleaned_p.append(f'<div class="math-block">{p}</div>')
        else:
            cleaned_p.append(f'<p>{p}</p>')
    body_html = '\n'.join(cleaned_p)

    # Build Bibliography HTML
    bib_html = '<h2 id="sec-references" class="section-title">References</h2>\n<ol class="bibliography">\n'
    for num, key, entry in bib_list:
        bib_html += f'<li id="ref-{key}" class="bib-item"><span class="bib-key">[{num}]</span> {entry}</li>\n'
    bib_html += '</ol>\n'

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- MathJax 3 configuration -->
    <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true,
        tags: 'ams'
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    <style>
        :root {{
            --primary-color: #0b3d91;
            --accent-color: #1e88e5;
            --text-color: #212529;
            --bg-color: #fdfdfd;
            --card-bg: #ffffff;
            --border-color: #dee2e6;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --serif-font: "Times New Roman", Times, "Latin Modern Roman", Georgia, serif;
        }}
        body {{
            background-color: #e9ecef;
            color: var(--text-color);
            font-family: var(--serif-font);
            line-height: 1.62;
            margin: 0;
            padding: 20px;
        }}
        .paper-container {{
            max-width: 1080px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 50px 70px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-radius: 4px;
        }}
        .top-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 12px;
            margin-bottom: 30px;
            font-family: var(--font-family);
            font-size: 0.85rem;
            color: #6c757d;
        }}
        .btn {{
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn:hover {{
            background: #082b66;
        }}
        .paper-title {{
            font-size: 2.1rem;
            font-weight: 700;
            text-align: center;
            line-height: 1.25;
            margin-bottom: 20px;
            color: #111;
        }}
        .paper-authors {{
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.05rem;
            line-height: 1.5;
        }}
        .paper-abstract {{
            background: #f8f9fa;
            border-left: 4px solid var(--primary-color);
            padding: 20px 24px;
            margin: 30px 0;
            border-radius: 0 6px 6px 0;
        }}
        .abstract-title {{
            font-family: var(--font-family);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 1px;
            margin-bottom: 8px;
            color: var(--primary-color);
        }}
        .abstract-text {{
            font-size: 0.95rem;
            text-align: justify;
            margin: 0;
        }}
        .paper-keywords {{
            margin-top: 14px;
            font-size: 0.88rem;
            color: #495057;
        }}
        .section-title {{
            font-size: 1.45rem;
            font-weight: 700;
            border-bottom: 1px solid #ced4da;
            padding-bottom: 6px;
            margin-top: 40px;
            margin-bottom: 16px;
            color: #111;
        }}
        .subsection-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 28px;
            margin-bottom: 12px;
            color: #222;
        }}
        .subsubsection-title {{
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 8px;
        }}
        .sec-num {{
            color: var(--primary-color);
            margin-right: 6px;
        }}
        p {{
            text-align: justify;
            text-justify: inter-word;
            margin-bottom: 14px;
            font-size: 1.02rem;
        }}
        .paper-figure {{
            margin: 32px auto;
            text-align: center;
            background: #fafafa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 18px;
        }}
        .figure-img {{
            max-width: 95%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        figcaption {{
            margin-top: 12px;
            font-size: 0.9rem;
            color: #495057;
            text-align: justify;
            padding: 0 16px;
            line-height: 1.45;
        }}
        .table-container {{
            margin: 30px auto;
            overflow-x: auto;
        }}
        .table-caption {{
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }}
        .academic-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            font-family: var(--font-family);
        }}
        .academic-table th, .academic-table td {{
            padding: 7px 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .academic-table th {{
            background: #f1f3f5;
            font-weight: 600;
            border-top: 2px solid #212529;
            border-bottom: 2px solid #212529;
        }}
        .academic-table tr:last-child td {{
            border-bottom: 2px solid #212529;
        }}
        .cite-ref {{
            color: var(--accent-color);
            text-decoration: none;
            font-weight: 600;
        }}
        .cite-ref:hover {{
            text-decoration: underline;
        }}
        .in-ref {{
            color: var(--accent-color);
            text-decoration: none;
            font-weight: 600;
        }}
        .bibliography {{
            font-size: 0.88rem;
            line-height: 1.5;
            padding-left: 20px;
        }}
        .bib-item {{
            margin-bottom: 8px;
            text-align: justify;
        }}
        .bib-key {{
            font-weight: 600;
            color: var(--primary-color);
        }}
        .math-block {{
            margin: 18px 0;
            overflow-x: auto;
            text-align: center;
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .paper-container {{
                box-shadow: none;
                padding: 0;
                max-width: 100%;
            }}
            .top-bar {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="paper-container">
        <div class="top-bar">
            <div>
                <strong>Technical Report / arXiv Preprint</strong> (cs.CV, cs.CR)
            </div>
            <div>
                <button class="btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
            </div>
        </div>

        <h1 class="paper-title">{title}</h1>

        <div class="paper-authors">
            {author_clean}
        </div>

        <div class="paper-abstract">
            <div class="abstract-title">Abstract</div>
            <p class="abstract-text">{abstract_text}</p>
            <div class="paper-keywords">
                <strong>Keywords:</strong> {keywords_text}
            </div>
        </div>

        <div class="paper-body">
            {body_html}
        </div>

        <div class="paper-references">
            {bib_html}
        </div>
    </div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"[+] Successfully generated HTML preview: {html_path.resolve()} ({len(full_html):,} bytes)")

if __name__ == "__main__":
    build_html()
