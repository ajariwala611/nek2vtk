"""Detect physical boundaries directly from the field-file geometry (MPI aware).

Because NekRS reorders and deforms elements relative to the ``.re2``, the only
reliable source of the *as-simulated* boundary is the field geometry itself.  A
face is on a boundary if and only if it belongs to a single element (interior
faces are shared by two).  We detect those faces topologically by hashing face
centres, split them into connected regions, and attach a boundary code to each
by matching the nearest ``.re2`` boundary face (used purely as a naming hint).

Parallelism
-----------
An interior face shared across a rank partition is seen once on each of the two
ranks, so the "appears once" test must be global.  Each rank first drops faces
that are duplicated *within* its own partition (definitely interior), then the
surviving candidates are gathered to the root, deduplicated globally, split and
classified, and the per-rank face lists are scattered back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import geometry, nekfaces
from .regions import RegionInfo, region_infos, split_faces


@dataclass
class BoundaryPlan:
    """Per-rank boundary faces plus global region metadata (broadcast)."""

    loc_elem: np.ndarray     # (nf_local,) local element index
    loc_face: np.ndarray     # (nf_local,) Nek face id 1..6
    loc_region: np.ndarray   # (nf_local,) region id
    nreg: int
    infos: List[RegionInfo]
    total_faces: int
    match_dist_median: float
    match_dist_max: float


def detect_boundaries(
    comm,
    coords: List[np.ndarray],
    re2_centers: Optional[np.ndarray],
    re2_codes: Optional[np.ndarray],
    normal_angle_deg: float = 40.0,
    round_decimals: int = 7,
) -> BoundaryPlan:
    """Detect boundary faces from field geometry and assign them to regions.

    ``re2_centers`` / ``re2_codes`` are only needed on the root rank.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    X, Y, Z = coords
    ne, lz, ly, lx = X.shape
    ndim = 3 if lz > 1 else 2
    nfe = nekfaces.n_faces(ndim)

    ecen = geometry.element_centroids(X, Y, Z)

    centers_l, corners_l, normals_l, owner_l, faceid_l = [], [], [], [], []
    for f in range(1, nfe + 1):
        corners = geometry.face_corners_all(X, Y, Z, f)  # (ne,4,3)
        centers_l.append(corners.mean(axis=1))
        corners_l.append(corners)
        normals_l.append(geometry.outward_normals(corners, ecen))
        owner_l.append(np.arange(ne, dtype=np.int64))
        faceid_l.append(np.full(ne, f, dtype=np.int8))

    centers = np.concatenate(centers_l)
    corners = np.concatenate(corners_l)
    normals = np.concatenate(normals_l)
    owner = np.concatenate(owner_l)
    faceid = np.concatenate(faceid_l)

    # Drop faces duplicated within this rank (definitely interior).
    key = np.round(centers, round_decimals)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    cand = cnt[inv] == 1

    payload = {
        "centers": centers[cand],
        "corners": corners[cand],
        "normals": normals[cand],
        "owner": owner[cand],
        "faceid": faceid[cand],
        "rank": rank,
    }
    gathered = comm.gather(payload, root=0)

    scatter_list = None
    meta = None
    if rank == 0:
        C = np.concatenate([g["centers"] for g in gathered], axis=0)
        CO = np.concatenate([g["corners"] for g in gathered], axis=0)
        NR = np.concatenate([g["normals"] for g in gathered], axis=0)
        OE = np.concatenate([g["owner"] for g in gathered], axis=0)
        FID = np.concatenate([g["faceid"] for g in gathered], axis=0)
        RK = np.concatenate(
            [np.full(len(g["centers"]), g["rank"], dtype=np.int64) for g in gathered]
        )

        # Global dedup: true boundary faces appear exactly once.
        key = np.round(C, round_decimals)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        true = cnt[inv] == 1

        b_center = C[true]
        b_corners = CO[true]
        b_normal = NR[true]
        b_owner = OE[true]
        b_faceid = FID[true]
        b_rank = RK[true]

        # Assign codes from the nearest .re2 boundary face (naming hint).
        if re2_centers is not None and len(re2_centers) > 0:
            from scipy.spatial import cKDTree

            tree = cKDTree(re2_centers)
            dist, idx = tree.query(b_center)
            code = np.asarray(re2_codes)[idx].astype("<U3")
            md = float(np.median(dist)) if dist.size else 0.0
            mx = float(dist.max()) if dist.size else 0.0
        else:
            code = np.full(len(b_center), "bc", dtype="<U3")
            md = mx = 0.0

        region = split_faces(b_corners, b_normal, normal_angle_deg)
        infos = region_infos(b_corners, b_normal, region, codes=code)
        nreg = len(infos)

        buckets: List[List[Tuple[int, int, int]]] = [[] for _ in range(size)]
        for i in range(len(region)):
            buckets[int(b_rank[i])].append(
                (int(b_owner[i]), int(b_faceid[i]), int(region[i]))
            )
        scatter_list = []
        for lst in buckets:
            if lst:
                el = np.array([t[0] for t in lst], dtype=np.int64)
                fa = np.array([t[1] for t in lst], dtype=np.int8)
                rg = np.array([t[2] for t in lst], dtype=np.int64)
            else:
                el = np.zeros(0, dtype=np.int64)
                fa = np.zeros(0, dtype=np.int8)
                rg = np.zeros(0, dtype=np.int64)
            scatter_list.append((el, fa, rg))
        meta = {"nreg": nreg, "infos": infos, "total_faces": int(true.sum()),
                "md": md, "mx": mx}

    my_elem, my_face, my_region = comm.scatter(scatter_list, root=0)
    meta = comm.bcast(meta, root=0)

    return BoundaryPlan(
        loc_elem=my_elem,
        loc_face=my_face,
        loc_region=my_region,
        nreg=meta["nreg"],
        infos=meta["infos"],
        total_faces=meta["total_faces"],
        match_dist_median=meta["md"],
        match_dist_max=meta["mx"],
    )
