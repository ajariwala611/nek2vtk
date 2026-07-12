"""Read boundary-condition codes from a Nek5000 ``.re2`` mesh file.

The ``.re2`` stores a 3-character boundary code (``W``, ``v``, ``o``, ``P``,
...) for every element face, on the *input* (undeformed) mesh.  NekRS may both
reorder elements and deform the geometry at run time, so these codes cannot be
mapped to the field files by element index.  Instead we use them only as
*location-matched naming hints*: nek2vtk detects the boundaries from the field
geometry and asks the ``.re2`` "what code sits closest to here?".

This module therefore just returns, for every boundary face, its centre
coordinate and its code, plus the total element count for the sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import nekfaces

INTERIOR_CODES = {"", "E", "e"}


@dataclass
class Re2Boundaries:
    centers: np.ndarray   # (Nf, 3) boundary-face centres
    codes: np.ndarray     # (Nf,) <U3 boundary codes
    nelgt: int
    ndim: int

    @property
    def nfaces(self) -> int:
        return len(self.codes)


def read_re2_boundaries(path: str | Path) -> Re2Boundaries:
    """Read the ``.re2`` file and return its boundary-face centres and codes."""
    from pymech.neksuite import readre2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"re2 file not found: {path}")

    mesh = readre2(str(path))
    ndim = int(mesh.ndim)
    nel = int(mesh.nel)
    nf_per_elem = nekfaces.n_faces(ndim)

    centers = []
    codes = []
    for e in range(nel):
        el = mesh.elem[e]
        pos = np.asarray(el.pos)  # (ndim, lz, ly, lx), lx=2
        lz, ly, lx = pos.shape[1], pos.shape[2], pos.shape[3]
        bcs = el.bcs[0]
        for f0 in range(nf_per_elem):
            code = str(bcs[f0][0]).strip()
            if code in INTERIOR_CODES:
                continue
            face = f0 + 1
            idx = nekfaces.face_corner_indices(face, lx, ly, lz)
            c = np.zeros(3)
            for (kz, jy, ix) in idx:
                for d in range(ndim):
                    c[d] += pos[d, kz, jy, ix]
            c /= 4.0
            centers.append(c)
            codes.append(code)

    n = len(codes)
    return Re2Boundaries(
        centers=np.asarray(centers, dtype=np.float64) if n else np.zeros((0, 3)),
        codes=np.asarray(codes, dtype="<U3"),
        nelgt=nel,
        ndim=ndim,
    )
