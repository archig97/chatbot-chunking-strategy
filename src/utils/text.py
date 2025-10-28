from __future__ import annotations
import re
from typing import Iterable, List, Tuple

CODE_TOKEN_PCT_THRESH = 0.15
INDENTED_LINE_PCT_THRESH = 0.25

# Simple heuristic: lines that look like code
_CODEY = re.compile(r"[{}();,:=+\-\/*<>\[\]|&$#@]|\b(def|class|import|from|return|if|else|for|while|try|catch|finally|public|private|static|void|new)\b")


def split_candidate_blocks(text: str) -> List[str]:
# Split by double blank lines as coarse blocks
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip("\n") for b in blocks if b.strip()]


def block_is_codey(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    codey_hits = sum(1 for ln in lines if _CODEY.search(ln))
    codey_pct = codey_hits / max(1, len(lines))
    indent_hits = sum(1 for ln in lines if ln.startswith(" ") or ln.startswith("\t"))
    indent_pct = indent_hits / max(1, len(lines))
    return (codey_pct >= CODE_TOKEN_PCT_THRESH) or (indent_pct >= INDENTED_LINE_PCT_THRESH)


def find_code_blocks(paged_text: Iterable[Tuple[int, str]]) -> List[Tuple[str, int]]:
    """Return list of (code_text, page_number)."""
    results: List[Tuple[str, int]] = []
    for page_no, text in paged_text:
        for block in split_candidate_blocks(text):
            if block_is_codey(block):
                results.append((block, page_no))
    return results


def normalize_code_whitespace(code: str) -> str:
# Normalize common PDF artifacts like inconsistent multiple spaces
    code = code.replace('\r', '')
    # Collapse mixed indentation spaces
    code = re.sub(r"\t", " ", code)
    return code
