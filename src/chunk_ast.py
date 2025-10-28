from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class Chunk:
    id: str
    symbol: str
    code: str
    page: int


def _symbol_name(node: ast.AST, parents: List[str]) -> str:
    if isinstance(node, ast.FunctionDef):
        return ".".join(parents + [node.name])
    if isinstance(node, ast.AsyncFunctionDef):
        return ".".join(parents + [node.name])
    if isinstance(node, ast.ClassDef):
        return ".".join(parents + [node.name])
    return ".".join(parents or ["<module>"])


def _get_source_segment(src: str, node: ast.AST) -> str:
# ast.get_source_segment is robust when lineno/col_offset present
    seg = ast.get_source_segment(src, node)
    if seg is None:
        return src
    return seg


def chunk_python_ast(block_id: str, code: str, page: int) -> List[Chunk]:
    """Return AST-aware chunks: classes and top-level functions. Fallback to whole block on parse error."""
    chunks: List[Chunk] = []
    try:
        mod = ast.parse(code)
        parents: List[str] = []
            # include module docstring/top imports as a chunk
        module_body = code
            # Gather class/function nodes to create clean chunks
        for node in mod.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                sym = _symbol_name(node, parents)
                src = _get_source_segment(code, node)
                chunks.append(Chunk(id=f"{block_id}::{sym}", symbol=sym, code=src, page=page))
        # If no structured nodes found, fallback to whole code block
        if not chunks:
            chunks.append(Chunk(id=f"{block_id}::<module>", symbol="<module>", code=code, page=page))
    except SyntaxError:
    # PDF artifacts often break parsing; fallback to raw block
        chunks.append(Chunk(id=f"{block_id}::<unparsed>", symbol="<unparsed>", code=code, page=page))
    return chunks


def chunk_blocks_python(blocks: Iterable[Dict]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for b in blocks:
        code: str = b["code"]
        block_id: str = b["id"]
        page: int = b["page"]
        for ch in chunk_python_ast(block_id, code, page):
            all_chunks.append(ch)
    return all_chunks