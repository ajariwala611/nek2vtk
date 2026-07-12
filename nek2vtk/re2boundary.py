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

# Generic placeholder codes written by the Nek5000 mesh converters for every
# (non-periodic) physical boundary, with the real sideset id stored in the 5th
# bc parameter:  gmsh2nek -> 'MSH', exo2nek -> 'EXO', cgns2nek -> 'CGN'.
# For these we prefer the numeric sideset id.  (Periodic faces are always
# coded 'P' by all three converters, never the generic code.)
GENERIC_CODES = {"MSH", "EXO", "CGN"}


def _face_label(code: str, boundary_id: int) -> str:
    """Return the hint label for a boundary face.

    Prefers a meaningful Nek code (``W``, ``v``, ``o``, ``P``, ...).  When the
    code is a generic converter placeholder (``MSH``/``EXO``/``CGN``) it falls
    back to the numeric sideset id (``bc3``) stored in the 5th bc parameter by
    gmsh2nek / exo2nek / cgns2nek.
    """
    if code.upper() in GENERIC_CODES and boundary_id > 0:
        return f"bc{boundary_id}"
    return code


@dataclass
class Re2Boundaries:
    centers: np.ndarray   # (Nf, 3) boundary-face centres
    normals: np.ndarray   # (Nf, 3) outward unit normals
    codes: np.ndarray     # (Nf,) boundary labels (code or bc<id>)
    nelgt: int            # total elements (fluid + solid)
    nelgv: int            # fluid (velocity) elements
    ndim: int

    @property
    def nfaces(self) -> int:
        return len(self.codes)

    @property
    def is_cht(self) -> bool:
        """True if the mesh has solid elements (conjugate heat transfer)."""
        return self.nelgv < self.nelgt


def read_re2_header(path: str | Path):
    """Parse the 80-byte ``.re2`` ASCII header.

    Returns ``(nelgt, ndim, nelgv)``.  The header looks like
    ``#v004    96576  3    96576   1 hdr ...`` — version token, then the total
    element count, the dimension, and the number of fluid (velocity) elements.
    For a conjugate-heat-transfer mesh ``nelgv < nelgt`` (the remainder are
    solid elements).
    """
    path = Path(path)
    with open(path, "rb") as fh:
        hdr = fh.read(80).decode("ascii", errors="replace")
    tokens = hdr.split()
    # tokens[0] is the version marker (e.g. '#v004'); the ints follow.
    ints = []
    for t in tokens[1:]:
        try:
            ints.append(int(t))
        except ValueError:
            break
    if len(ints) < 2:
        raise ValueError(f"{path}: could not parse .re2 header: {hdr!r}")
    nelgt = ints[0]
    ndim = ints[1]
    nelgv = ints[2] if len(ints) >= 3 else nelgt
    return nelgt, ndim, nelgv


def read_re2_boundaries(path: str | Path) -> Re2Boundaries:
    """Read the ``.re2`` file and return its boundary-face centres and codes."""
    from pymech.neksuite import readre2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"re2 file not found: {path}")

    nelgt_hdr, _, nelgv_hdr = read_re2_header(path)

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
        nelgv=int(nelgv_hdr),
        ndim=ndim,
    )
