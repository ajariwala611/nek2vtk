"""Parse a Nek5000/NekRS ``CASE.nek5000`` case file.

A ``.nek5000`` file is a small text file that describes a *series* of field
files. Example (``flat_plate.nek5000``)::

    filetemplate: flat_plate%01d.f%05d
    firsttimestep: 1
    numtimesteps:  10

``filetemplate`` is a C-style format string with two integer conversions.
The first is the (parallel) file index, the second is the time-step index.
For serial data (which is what these single-file dumps are) the first index is
always ``0`` and the second runs from ``firsttimestep`` to
``firsttimestep + numtimesteps - 1``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CaseFile:
    """Parsed contents of a ``CASE.nek5000`` file."""

    path: Path
    filetemplate: str
    firsttimestep: int
    numtimesteps: int

    @property
    def casename(self) -> str:
        """Base name of the case (``flat_plate`` for ``flat_plate.nek5000``)."""
        return self.path.stem

    def field_files(self, root: Path | None = None) -> List[Path]:
        """Return the list of field-file paths described by this case file.

        Parameters
        ----------
        root
            Directory the field files live in. Defaults to the directory that
            contains the ``.nek5000`` file.
        """
        if root is None:
            root = self.path.parent
        root = Path(root)
        files = []
        for step in range(self.firsttimestep, self.firsttimestep + self.numtimesteps):
            # The template has two %d style fields: (file-index, timestep).
            # Serial dumps always use file index 0.
            name = self.filetemplate % (0, step)
            files.append(root / name)
        return files


def read_casefile(path: str | Path) -> CaseFile:
    """Read and parse a ``CASE.nek5000`` file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a required key is missing or malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Case file not found: {path}")

    fields = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()

    try:
        filetemplate = fields["filetemplate"]
    except KeyError:
        raise ValueError(f"{path}: missing 'filetemplate' entry")

    def _int(key: str, default: int | None = None) -> int:
        if key not in fields:
            if default is not None:
                return default
            raise ValueError(f"{path}: missing '{key}' entry")
        try:
            return int(fields[key])
        except ValueError:
            raise ValueError(f"{path}: '{key}' is not an integer: {fields[key]!r}")

    first = _int("firsttimestep", 1)
    num = _int("numtimesteps", 1)

    # Sanity-check the template: it must contain at least two integer format
    # conversions so ``template % (0, step)`` works.
    n_int_conv = len(re.findall(r"%[0-9]*d", filetemplate))
    if n_int_conv < 2:
        raise ValueError(
            f"{path}: filetemplate {filetemplate!r} does not look like a Nek "
            "template (expected two integer fields, e.g. 'case%01d.f%05d')"
        )

    return CaseFile(
        path=path,
        filetemplate=filetemplate,
        firsttimestep=first,
        numtimesteps=num,
    )
