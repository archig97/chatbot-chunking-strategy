from __future__ import annotations
from typing import List, Tuple
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTAnno


def extract_pdf_text_with_pages(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number_1based, text) with best-effort line breaks.
    Uses pdfminer.six layout to keep lines reasonably intact.
    """
    pages_text: List[Tuple[int, str]] = []
    for i, page_layout in enumerate(extract_pages(path)):
        lines: List[str] = []
        current_line = []
        last_y = None
# Collect text blocks roughly in reading order
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    line_str = text_line.get_text()
                    if line_str is None:
                        continue
                lines.append(line_str.rstrip('\n'))
# Join lines with newlines
            page_text = "\n".join(lines)
            pages_text.append((i + 1, page_text))
    return pages_text