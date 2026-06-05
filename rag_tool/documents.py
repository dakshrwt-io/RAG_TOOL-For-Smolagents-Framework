"""Document loading and file collection utilities."""

from pathlib import Path
from typing import Iterable, List, Union


def collect_files(input_paths: Iterable[Union[str, Path]], allowed_exts: set[str]) -> List[Path]:
    """Collect files from input paths matching allowed extensions."""
    files: List[Path] = []
    for path in input_paths:
        path = Path(path)
        if path.is_dir():
            for ext in allowed_exts:
                files.extend(path.rglob(f"*{ext}"))
        elif path.is_file() and path.suffix.lower() in allowed_exts:
            files.append(path)
    return files


def load_documents(input_paths: Iterable[Union[str, Path]], allowed_exts: set[str]) -> List[dict]:
    """Load documents from input paths into list of dicts with text, type, and source."""
    files = collect_files(input_paths, allowed_exts)
    if not files:
        raise FileNotFoundError("No input files found for upload_doc")

    documents: List[dict] = []
    for file in files:
        doc_type = file.parent.name or file.suffix.lstrip(".") or "document"
        text = file.read_text(encoding="utf-8", errors="ignore")
        documents.append({"type": doc_type, "source": file.as_posix(), "text": text})

    return documents
