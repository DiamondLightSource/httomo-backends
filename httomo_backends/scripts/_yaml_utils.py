"""Validation and output helpers shared by the YAML generators."""

import os
from pathlib import Path
import stat
import tempfile


def validate_name(name, dotted=False):
    """Require Python names so input names cannot escape template directories."""
    if not isinstance(name, str) or not all(
        part.isidentifier() for part in (name.split(".") if dotted else [name])
    ):
        raise ValueError(f"Invalid {'module' if dotted else 'method'} name: {name!r}")


def atomic_write(path: Path, text: str):
    """Replace an output only after its entire contents have been generated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
