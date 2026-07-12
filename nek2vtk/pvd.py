"""Write ParaView ``.pvd`` collection files for time series.

A ``.pvd`` file is a tiny XML index that ties a set of per-timestep dataset
files (``.vtp``, ``.vtkhdf``, ...) to their simulation times, so ParaView loads
them as a single animatable object.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def write_pvd(path: Path, entries: List[Tuple[float, str]]) -> None:
    """Write a ``.pvd`` file.

    Parameters
    ----------
    path
        Output ``.pvd`` path.
    entries
        List of ``(time, relative_file_path)`` tuples.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time, rel in entries:
        lines.append(
            f'    <DataSet timestep="{time:.10g}" group="" part="0" '
            f'file="{rel}"/>'
        )
    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    Path(path).write_text("\n".join(lines) + "\n")
