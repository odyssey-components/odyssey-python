"""Shared ZIP entry metadata for OdyLab scene and algorithm packages."""

from __future__ import annotations

import zipfile
from importlib.resources import files
from zipfile import ZipFile


def write_zip_entry(archive: ZipFile, name: str, content: str | bytes, executable: bool = False) -> None:
    """Write a deflated ZIP entry with a stable Unix file mode."""
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (0o100755 if executable else 0o100644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, content)


def write_config_package(archive: ZipFile) -> None:
    """Include the shared configuration source in a standalone algorithm ZIP."""

    for source in sorted(files("odyssey_config").iterdir(), key=lambda item: item.name):
        if source.name.endswith(".py") and source.is_file():
            write_zip_entry(archive, f"odyssey_config/{source.name}", source.read_bytes())
