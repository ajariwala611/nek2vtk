"""Vectorised geometry helpers shared by the re2 and field boundary paths."""

from __future__ import annotations

import numpy as np

from . import nekfaces


def element_centroids(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Centroid of every element's GLL points. ``(ne, 3)``."""
    ne = X.shape[0]
    return np.stack(
        [X.reshape(ne, -1).mean(1),
         Y.reshape(ne, -1).mean(1),
         Z.reshape(ne, -1).mean(1)],
        axis=1,
    )


def face_corners_all(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     face: int) -> np.ndarray:
    """Corner coordinates of ``face`` for every element. ``(ne, 4, 3)``.

    Arrays are shaped ``(ne, lz, ly, lx)``.
    """
    ne, lz, ly, lx = X.shape
    idx = nekfaces.face_corner_indices(face, lx, ly, lz)
    out = np.empty((ne, 4, 3))
    for k, (kz, jy, ix) in enumerate(idx):
        out[:, k, 0] = X[:, kz, jy, ix]
        out[:, k, 1] = Y[:, kz, jy, ix]
        out[:, k, 2] = Z[:, kz, jy, ix]
    return out


def outward_normals(corners: np.ndarray, elem_centroids: np.ndarray) -> np.ndarray:
    """Unit normals of quad faces, oriented away from their element centroids.

    ``corners`` is ``(n, 4, 3)`` and ``elem_centroids`` is ``(n, 3)``.
    """
    v1 = corners[:, 2] - corners[:, 0]
    v2 = corners[:, 3] - corners[:, 1]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    fc = corners.mean(axis=1)
    # handle degenerate normals
    degenerate = (norm[:, 0] < 1e-30)
    n = np.where(norm > 1e-30, n / np.where(norm == 0, 1, norm), fc - elem_centroids)
    if degenerate.any():
        d = fc[degenerate] - elem_centroids[degenerate]
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(dn > 1e-30, d / np.where(dn == 0, 1, dn), np.array([0., 0., 1.]))
        n[degenerate] = d
    # renormalise (the where above may have used unnormalised fallback)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nn == 0, 1, nn)
    # orient outward
    outward = np.einsum("ij,ij->i", n, fc - elem_centroids) < 0
    n[outward] = -n[outward]
    return n
