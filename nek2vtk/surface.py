"""Extract boundary regions as VTK PolyData (``.vtp``) surfaces.

Each Nek element face is an ``(na, nb)`` grid of GLL points.  We tessellate it
into ``(na-1) x (nb-1)`` flat quads (subdivided-linear representation), carry
all field values as point data, gather the per-rank pieces to the root rank,
merge duplicated GLL points shared between neighbouring faces, and write a
single ``.vtp`` per region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from . import nekfaces


def _quad_connectivity(na: int, nb: int) -> np.ndarray:
    """Quad connectivity for a single ``(na, nb)`` grid, C-ordered points.

    Returns an ``((na-1)*(nb-1), 4)`` int array of local point indices.
    """
    i = np.arange(na - 1)
    j = np.arange(nb - 1)
    ii, jj = np.meshgrid(i, j, indexing="ij")
    p00 = (ii * nb + jj).ravel()
    p01 = p00 + 1
    p10 = p00 + nb
    p11 = p10 + 1
    return np.stack([p00, p01, p11, p10], axis=1)


def extract_region_surface(
    region_id: int,
    elem_local: np.ndarray,   # (Nf,) local element index for this rank's region faces
    face_ids: np.ndarray,     # (Nf,) Nek face id 1..6
    coords: List[np.ndarray], # [X, Y, Z], each (nelv, lz, ly, lx)
    fields: Dict[str, np.ndarray],  # name -> (nelv, lz, ly, lx)
) -> Dict[str, np.ndarray]:
    """Build the (rank-local) surface geometry + point data for one region.

    Returns a dict with ``points`` (Np,3), ``quads`` (Nq,4) and one entry per
    field (Np,) plus ``velocity`` (Np,3) if u/v/w are present.
    """
    X, Y, Z = coords
    lz, ly, lx = X.shape[1], X.shape[2], X.shape[3]

    pts_chunks: List[np.ndarray] = []
    quad_chunks: List[np.ndarray] = []
    data_chunks: Dict[str, List[np.ndarray]] = {k: [] for k in fields}
    point_offset = 0

    # Group faces of this region by Nek face id so each group shares a slice.
    for f in np.unique(face_ids):
        sel = elem_local[face_ids == f]
        if sel.size == 0:
            continue
        sl = nekfaces.face_slice(int(f), lx, ly, lz)  # 2D slice of (lz,ly,lx)
        full = (slice(None),) + sl                    # add the element axis

        xf = X[sel][full]  # (ne, na, nb)
        yf = Y[sel][full]
        zf = Z[sel][full]
        ne, na, nb = xf.shape

        pts = np.stack([xf.reshape(ne, -1),
                        yf.reshape(ne, -1),
                        zf.reshape(ne, -1)], axis=-1)  # (ne, na*nb, 3)
        pts = pts.reshape(-1, 3)
        pts_chunks.append(pts)

        base = _quad_connectivity(na, nb)  # (nq, 4)
        nq = base.shape[0]
        npf = na * nb
        # replicate for each element with growing offsets
        elem_base = (point_offset
                     + np.arange(ne)[:, None, None] * npf
                     + base[None, :, :])  # (ne, nq, 4)
        quad_chunks.append(elem_base.reshape(-1, 4))
        point_offset += ne * npf

        for name, arr in fields.items():
            df = arr[sel][full]  # (ne, na, nb)
            data_chunks[name].append(df.reshape(-1))

    if pts_chunks:
        points = np.concatenate(pts_chunks, axis=0)
        quads = np.concatenate(quad_chunks, axis=0)
        data = {k: np.concatenate(v, axis=0) for k, v in data_chunks.items()}
    else:
        points = np.zeros((0, 3))
        quads = np.zeros((0, 4), dtype=np.int64)
        data = {k: np.zeros(0) for k in fields}

    out = {"points": points, "quads": quads.astype(np.int64)}
    out.update(data)
    return out


def merge_pieces(pieces: List[Dict[str, np.ndarray]],
                 field_names: List[str]) -> Dict[str, np.ndarray]:
    """Concatenate per-rank surface pieces into one, fixing quad offsets."""
    all_pts = []
    all_quads = []
    all_data: Dict[str, List[np.ndarray]] = {k: [] for k in field_names}
    offset = 0
    for p in pieces:
        if p is None or p["points"].shape[0] == 0:
            continue
        all_pts.append(p["points"])
        all_quads.append(p["quads"] + offset)
        offset += p["points"].shape[0]
        for k in field_names:
            all_data[k].append(p[k])
    if not all_pts:
        merged = {"points": np.zeros((0, 3)),
                  "quads": np.zeros((0, 4), dtype=np.int64)}
        merged.update({k: np.zeros(0) for k in field_names})
        return merged
    merged = {"points": np.concatenate(all_pts, axis=0),
              "quads": np.concatenate(all_quads, axis=0)}
    for k in field_names:
        merged[k] = np.concatenate(all_data[k], axis=0)
    return merged


def dedup_points(surf: Dict[str, np.ndarray], field_names: List[str],
                 decimals: int = 9) -> Dict[str, np.ndarray]:
    """Merge coincident GLL points shared between faces/elements."""
    pts = surf["points"]
    if pts.shape[0] == 0:
        return surf
    key = np.round(pts, decimals)
    _, first_idx, inverse = np.unique(
        key, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.ravel()
    new_pts = pts[first_idx]
    new_quads = inverse[surf["quads"]]
    out = {"points": new_pts, "quads": new_quads}
    for k in field_names:
        out[k] = surf[k][first_idx]
    return out


def assemble_polydata(surf: Dict[str, np.ndarray], field_names: List[str]):
    """Build a ``pyvista.PolyData`` from a merged surface dict."""
    import pyvista as pv

    pts = surf["points"]
    quads = surf["quads"]
    if quads.shape[0] == 0:
        return pv.PolyData(pts)
    faces = np.hstack(
        [np.full((quads.shape[0], 1), 4, dtype=np.int64), quads]
    ).ravel()
    poly = pv.PolyData(pts, faces)

    # Assemble velocity vector if components are present.
    have_vel = all(c in field_names for c in ("u", "v", "w"))
    for name in field_names:
        poly.point_data[name] = surf[name]
    if have_vel:
        vel = np.stack([surf["u"], surf["v"], surf["w"]], axis=-1)
        poly.point_data["velocity"] = vel
        poly.point_data["velocity_magnitude"] = np.linalg.norm(vel, axis=1)
    elif "u" in field_names and "v" in field_names:
        vel = np.stack([surf["u"], surf["v"], np.zeros_like(surf["u"])], axis=-1)
        poly.point_data["velocity"] = vel
        poly.point_data["velocity_magnitude"] = np.linalg.norm(vel, axis=1)
    return poly


def write_vtp(surf: Dict[str, np.ndarray], field_names: List[str],
              path: Path) -> int:
    """Write a merged surface to ``path`` (a ``.vtp`` file). Returns #points."""
    poly = assemble_polydata(surf, field_names)
    poly.save(str(path))
    return poly.n_points
