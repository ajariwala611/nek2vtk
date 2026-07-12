"""Nek5000 hexahedral face conventions.

A spectral element stores its point data on an ``(lz, ly, lx)`` grid of GLL
points (this is the axis order used by both ``pymech`` and ``pysemtools``:
the fastest index is ``x``/``r``, then ``y``/``s``, then ``z``/``t``).

Nek numbers the six faces of a hexahedron 1..6 using the *symmetric* face
ordering.  The mapping below was verified against a genbox mesh: for the corner
element at the origin, face 1 lies on ``y=const`` (the wall ``W``), face 4 on
``x=const`` (the inlet ``v``) and face 5 on ``z=const`` (the periodic plane).

    face 1 : s = -1   -> jy = 0
    face 2 : r = +1   -> ix = lx-1
    face 3 : s = +1   -> jy = ly-1
    face 4 : r = -1   -> ix = 0
    face 5 : t = -1   -> kz = 0
    face 6 : t = +1   -> kz = lz-1

For 2D elements only faces 1..4 are used.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Nek face id (1-based) -> a function returning the numpy index expression that
# selects that face from an array shaped (lz, ly, lx).
#
# We return the slice as a tuple usable directly for indexing, given lx==ly==lz.
def face_slice(face: int, lx: int, ly: int, lz: int) -> Tuple:
    """Return a numpy index tuple selecting Nek ``face`` (1..6) from an
    ``(lz, ly, lx)`` array.  The result is a 2D ``(a, b)`` array."""
    if face == 1:
        return (slice(None), 0, slice(None))       # (lz, lx)
    if face == 2:
        return (slice(None), slice(None), lx - 1)   # (lz, ly)
    if face == 3:
        return (slice(None), ly - 1, slice(None))   # (lz, lx)
    if face == 4:
        return (slice(None), slice(None), 0)        # (lz, ly)
    if face == 5:
        return (0, slice(None), slice(None))        # (ly, lx)
    if face == 6:
        return (lz - 1, slice(None), slice(None))   # (ly, lx)
    raise ValueError(f"invalid Nek face id {face} (expected 1..6)")


# The four corner vertices of each face, expressed as (kz, jy, ix) with each
# index being either 0 or "max" (encoded as -1 meaning last).  Ordered so the
# four corners trace the face boundary (needed for a consistent quad winding).
_FACE_CORNERS = {
    1: [(0, 0, 0), (0, 0, -1), (-1, 0, -1), (-1, 0, 0)],
    2: [(0, 0, -1), (0, -1, -1), (-1, -1, -1), (-1, 0, -1)],
    3: [(0, -1, 0), (0, -1, -1), (-1, -1, -1), (-1, -1, 0)],
    4: [(0, 0, 0), (0, -1, 0), (-1, -1, 0), (-1, 0, 0)],
    5: [(0, 0, 0), (0, 0, -1), (0, -1, -1), (0, -1, 0)],
    6: [(-1, 0, 0), (-1, 0, -1), (-1, -1, -1), (-1, -1, 0)],
}


def face_corner_indices(face: int, lx: int, ly: int, lz: int):
    """Return the four ``(kz, jy, ix)`` corner index tuples of ``face``."""
    out = []
    for kz, jy, ix in _FACE_CORNERS[face]:
        out.append(
            (
                (lz - 1) if kz == -1 else 0,
                (ly - 1) if jy == -1 else 0,
                (lx - 1) if ix == -1 else 0,
            )
        )
    return out


def n_faces(ndim: int) -> int:
    return 6 if ndim == 3 else 4
