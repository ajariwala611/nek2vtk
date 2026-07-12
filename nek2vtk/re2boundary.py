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

from . import geometry, nekfaces

INTERIOR_CODES = {"", "E", "e"}

# Codes that are generic placeholders written by mesh converters (e.g.
# gmsh2nek writes "MSH" for every physical boundary and stores the real
# sideset id in the 5th bc parameter).  For these we prefer the numeric id.
GENERIC_CODES = {"MSH", "msh"}


def _face_label(code: str, boundary_id: int) -> str:
    """Return the hint label for a boundary face.

    Prefers a meaningful Nek code (``W``, ``v``, ``o``, ...).  When the code is a
    generic converter placeholder (``MSH``) it falls back to the numeric
    boundary id (``bc3``) that gmsh2nek / exo2nek store in the bc parameters.
    """
    if code in GENERIC_CODES and boundary_id > 0:
        return f"bc{boundary_id}"
    return code


@dataclass
class Re2Boundaries:
    centers: np.ndarray   # (Nf, 3) boundary-face centres
    normals: np.ndarray   # (Nf, 3) outward unit normals
    codes: np.ndarray     # (Nf,) boundary labels (code or bc<id>)
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

    corners_list = []
    ecen_list = []
    codes = []
    for e in range(nel):
        el = mesh.elem[e]
        pos = np.asarray(el.pos)  # (ndim, lz, ly, lx), lx=2
        lz, ly, lx = pos.shape[1], pos.shape[2], pos.shape[3]
        ec = np.zeros(3)
        for d in range(ndim):
            ec[d] = pos[d].mean()
        bcs = el.bcs[0]
        for f0 in range(nf_per_elem):
            code = str(bcs[f0][0]).strip()
            if code in INTERIOR_CODES:
                continue
            # gmsh2nek/exo2nek store the sideset id in the 5th bc parameter (f7).
            boundary_id = int(round(float(bcs[f0][7])))
            face = f0 + 1
            idx = nekfaces.face_corner_indices(face, lx, ly, lz)
            corner = np.zeros((4, 3))
            for k, (kz, jy, ix) in enumerate(idx):
                for d in range(ndim):
                    corner[k, d] = pos[d, kz, jy, ix]
            corners_list.append(corner)
            ecen_list.append(ec)
            codes.append(_face_label(code, boundary_id))

    n = len(codes)
    if n:
        corners = np.asarray(corners_list, dtype=np.float64)
        ecen = np.asarray(ecen_list, dtype=np.float64)
        centers = corners.mean(axis=1)
        normals = geometry.outward_normals(corners, ecen)
    else:
        centers = np.zeros((0, 3))
        normals = np.zeros((0, 3))
    return Re2Boundaries(
        centers=centers,
        normals=normals,
        codes=np.asarray(codes, dtype="<U8"),
        nelgt=nel,
        ndim=ndim,
    )
