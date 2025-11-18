from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import os

from utils.pdf import extract_pdf_text_with_pages
from utils.text import find_code_blocks, normalize_code_whitespace


def extract_pdf_code(pdf_path: str, out_dir: str = "data/processed") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pages = extract_pdf_text_with_pages(pdf_path)
    code_blocks = find_code_blocks(pages)

# Save raw blocks with page metadata for downstream parsing
    out_path = Path(out_dir) / (Path(pdf_path).stem + "_code_blocks.json")
    payload: List[Dict] = []
    for idx, (code, page_no) in enumerate(code_blocks):
        payload.append({
            "id": f"block_{idx:04d}",
            "page": page_no,
            "code": normalize_code_whitespace(code),
        })
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(payload)} candidate code blocks → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    '''
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/build_index.py <path/to.pdf> [config.yaml]")
        raise SystemExit(1)
    '''
    print(os.listdir())
    pdf = "data/pdfs/ECE506_textbook.pdf"
    out_dir = "data/processed"
    extract_pdf_code(pdf, out_dir)