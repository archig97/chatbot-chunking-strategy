import re
import json
from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine

PDF_PATH = "data/pdfs/saasbook.pdf"  # your uploaded PDF

# --- heuristics for recognizing headings like:
# "1 Introduction to Software as a Service..."
# "1.2 Software Development Processes: Plan-and-Document"
# "Preface to the Second Edition"
HEADING_RE = re.compile(
    r"""^(
        (\d+(\.\d+)*)      # numbered heading like "1" or "1.2" or "10.6"
        \s+
        .+                 # title text
      |
        (Preface|Contents|About the Authors|Afterword|Appendix)   # unnumbered top-levels
        .*
    )$""",
    re.IGNORECASE | re.VERBOSE
)

def iter_pdf_lines(pdf_path):
    """
    Yield (page_number, line_text) in reading order.
    """
    for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        line = text_line.get_text().strip()
                        if line:
                            yield page_num, line

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def is_heading(line: str) -> bool:
    # We *don't* want figure captions etc. We'll just check length and regex.
    if len(line.split()) > 30:  # too long to be a heading
        return False
    return bool(HEADING_RE.match(line))

def build_ast_nodes(pdf_path):
    """
    Returns a list of 'nodes'.
    Each node ~ one logical section with:
      {
        "id": "...",
        "title": "...",
        "pages": [int,...],
        "text_blocks": ["para1 ...", "para2 ...", ...]
      }
    """
    nodes = []
    current_node = None
    buffer_paragraph_lines = []

    def flush_paragraph():
        nonlocal buffer_paragraph_lines, current_node
        if buffer_paragraph_lines and current_node is not None:
            para = normalize_ws(" ".join(buffer_paragraph_lines))
            if para:
                current_node["text_blocks"].append(para)
        buffer_paragraph_lines = []

    def start_new_node(title, page_num):
        return {
            "id": f"sec_{len(nodes):04d}",
            "title": title,
            "pages": [page_num],
            "text_blocks": []
        }

    last_page_for_node = None

    for page_num, raw_line in iter_pdf_lines(pdf_path):
        line = normalize_ws(raw_line)

        # merge weird artifacts like "(cid:31)" etc.
        if "(cid:" in line:
            line = re.sub(r"\(cid:[^)]+\)", "", line).strip()

        if not line:
            # blank-ish => paragraph boundary
            flush_paragraph()
            continue

        if is_heading(line):
            # close out old node fully
            flush_paragraph()
            if current_node is not None:
                nodes.append(current_node)

            # start a new node
            current_node = start_new_node(title=line, page_num=page_num)
            last_page_for_node = page_num
        else:
            # regular body text line
            if current_node is None:
                # Found body text before any heading:
                # create a "preamble" node so we don't lose it.
                current_node = start_new_node(title="(preamble)", page_num=page_num)
                last_page_for_node = page_num

            buffer_paragraph_lines.append(line)

            # track pages covered by this node
            if last_page_for_node != page_num:
                current_node["pages"].append(page_num)
                last_page_for_node = page_num

    # end of loop: flush remaining
    flush_paragraph()
    if current_node is not None:
        nodes.append(current_node)

    return nodes


if __name__ == "__main__":
    ast_nodes = build_ast_nodes(PDF_PATH)
    print(f"Extracted {len(ast_nodes)} AST-style nodes/sections")
    # save raw nodes so we can embed later
    Path("chunk_db_raw.json").write_text(json.dumps(ast_nodes, indent=2), encoding="utf-8")
