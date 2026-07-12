"""Split a set of boundary faces into connected physical regions.

A "region" is a maximal set of faces that (a) share the same boundary code,
(b) are connected through shared edges, and (c) have outward normals within a
tolerance of each other.  Condition (c) separates two boundaries that share a
code and meet at a sharp edge (e.g. a genbox outlet and freestream top both
tagged ``o``) while keeping a smoothly curved wall as one region.

The same splitter is used for the ``.re2`` faces (to build naming hints) and
for the boundary faces detected topologically in the field files.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np


def split_faces(corners: np.ndarray, normal: np.ndarray,
                normal_angle_deg: float = 40.0,
                vertex_decimals: int = 7) -> np.ndarray:
    """Assign a region id to every face, purely from geometry.

    Two faces join into the same region when they share an edge (two corner
    vertices) and their outward normals differ by less than
    ``normal_angle_deg``.  Boundary *codes* are deliberately not used here:
    on a deformed NekRS mesh the nearest-``.re2``-code hint is unreliable
    face-by-face, so splitting on it fragments planar boundaries.  Codes are
    assigned per region afterwards by majority vote.

    Parameters
    ----------
    corners
        ``(Nf, 4, 3)`` corner coordinates of each face.
    normal
        ``(Nf, 3)`` outward unit normal per face.
    normal_angle_deg
        Maximum angle between adjacent normals to join them.
    vertex_decimals
        Rounding used when hashing corner coordinates into shared vertex ids.

    Returns
    -------
    np.ndarray
        ``(Nf,)`` int array of region ids, numbered deterministically by
        ``(-nfaces, centroid)`` so results are reproducible.
    """
    nf = corners.shape[0]
    region = np.full(nf, -1, dtype=np.int64)
    if nf == 0:
        return region

    centroid = corners.mean(axis=1)  # (Nf,3)

    # Shared vertex ids by hashing rounded corner coordinates.
    rounded = np.round(corners, vertex_decimals).reshape(-1, 3)
    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    vids = inv.reshape(nf, 4)

    v2f = defaultdict(list)
    for fi in range(nf):
        for v in vids[fi]:
            v2f[int(v)].append(fi)

    cos_thresh = np.cos(np.deg2rad(normal_angle_deg))
    vset = [set(vids[fi].tolist()) for fi in range(nf)]

    parent = np.arange(nf)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for faces in v2f.values():
        for i in range(len(faces)):
            fi = faces[i]
            for j in range(i + 1, len(faces)):
                fj = faces[j]
                if len(vset[fi] & vset[fj]) < 2:  # need a shared edge
                    continue
                if np.dot(normal[fi], normal[fj]) < cos_thresh:
                    continue
                union(fi, fj)

    roots = np.array([find(fi) for fi in range(nf)])

    summaries = []
    for r in np.unique(roots):
        mask = roots == r
        summaries.append(
            (r, int(mask.sum()), tuple(np.round(centroid[mask].mean(axis=0), 6)))
        )
    summaries.sort(key=lambda s: (-s[1], s[2]))
    for new_id, (r, *_rest) in enumerate(summaries):
        region[roots == r] = new_id
    return region


def sub_split_by_label(geo_region: np.ndarray, labels: np.ndarray,
                       centroid: np.ndarray, min_faces: int = 20,
                       min_frac: float = 0.02) -> np.ndarray:
    """Refine geometric regions by boundary label (sideset).

    Within each geometric region, faces are separated by their boundary label
    so that distinct sidesets that happen to be geometrically connected (e.g.
    several far-field patches meeting at shallow angles) become separate
    regions.  A label that covers fewer than ``max(min_faces, min_frac*N)`` of
    the region's faces is treated as a stray mis-match and folded into the
    region's majority label, so a handful of bad code hints cannot fragment a
    boundary.

    Returns a new region-id array, renumbered deterministically.
    """
    n = geo_region.size
    if n == 0:
        return geo_region.copy()

    adj = labels.copy()
    for g in np.unique(geo_region):
        mask = geo_region == g
        labs = labels[mask]
        vals, counts = np.unique(labs, return_counts=True)
        majority = vals[int(counts.argmax())]
        thr = max(min_faces, int(min_frac * mask.sum()))
        keep = set(vals[counts >= thr].tolist())
        new = np.where(np.isin(labs, list(keep)) if keep else False, labs, majority)
        adj[mask] = new

    # Final regions = unique (geo_region, adjusted-label) combinations.
    combo = {}
    for g, l in zip(geo_region.tolist(), adj.tolist()):
        combo.setdefault((g, l), 0)
    summaries = []
    for (g, l) in combo:
        m = (geo_region == g) & (adj == l)
        summaries.append(((g, l), int(m.sum()), tuple(np.round(centroid[m].mean(axis=0), 6))))
    summaries.sort(key=lambda s: (-s[1], s[2]))

    out = np.full(n, -1, dtype=np.int64)
    for new_id, ((g, l), *_rest) in enumerate(summaries):
        out[(geo_region == g) & (adj == l)] = new_id
    return out


@dataclass
class RegionInfo:
    """Human-facing summary of one boundary region."""

    region_id: int
    code: str
    nfaces: int
    centroid: np.ndarray
    normal: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    def signature(self) -> str:
        c = ",".join(f"{v:.5f}" for v in self.centroid)
        return f"{self.code}|{self.nfaces}|{c}"


def _mean_unit(normals: np.ndarray) -> np.ndarray:
    m = normals.mean(axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-30 else m


def _majority_code(codes: np.ndarray) -> str:
    vals, counts = np.unique(codes, return_counts=True)
    return str(vals[int(counts.argmax())])


def region_infos(corners: np.ndarray, normal: np.ndarray, region: np.ndarray,
                 codes: np.ndarray | None = None) -> List[RegionInfo]:
    """Build a sorted list of :class:`RegionInfo`.

    ``codes`` (per-face boundary code hints) are reduced to one code per region
    by majority vote, so a handful of mis-matched hints do not mislabel a region.
    """
    out = []
    nreg = int(region.max()) + 1 if region.size else 0
    centroid = corners.mean(axis=1) if corners.size else np.zeros((0, 3))
    for rid in range(nreg):
        mask = region == rid
        pts = corners[mask].reshape(-1, 3)
        code = _majority_code(codes[mask]) if codes is not None else "bc"
        out.append(
            RegionInfo(
                region_id=rid,
                code=code,
                nfaces=int(mask.sum()),
                centroid=centroid[mask].mean(axis=0),
                normal=_mean_unit(normal[mask]),
                bbox_min=pts.min(axis=0),
                bbox_max=pts.max(axis=0),
            )
        )
    return out


def format_region_table(infos: List[RegionInfo]) -> str:
    """Return a pretty table describing the detected regions."""
    lines = []
    header = (f"  {'id':>2}  {'code':<4} {'nfaces':>7}  "
              f"{'centroid (x,y,z)':<26} {'normal':<20} bbox")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in infos:
        cen = f"({r.centroid[0]:.3g}, {r.centroid[1]:.3g}, {r.centroid[2]:.3g})"
        nrm = f"({r.normal[0]:+.2f},{r.normal[1]:+.2f},{r.normal[2]:+.2f})"
        bbox = (f"x[{r.bbox_min[0]:.3g},{r.bbox_max[0]:.3g}] "
                f"y[{r.bbox_min[1]:.3g},{r.bbox_max[1]:.3g}] "
                f"z[{r.bbox_min[2]:.3g},{r.bbox_max[2]:.3g}]")
        lines.append(f"  {r.region_id:>2}  {r.code:<4} {r.nfaces:>7}  "
                     f"{cen:<26} {nrm:<20} {bbox}")
    return "\n".join(lines)
