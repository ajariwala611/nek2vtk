"""End-to-end Nek5000/NekRS -> VTK conversion driver (MPI aware).

Boundaries are detected from the field-file geometry (robust to NekRS element
reordering and mesh deformation) and named with hints from the ``.re2`` codes.
The mesh is assumed static across the time series, so boundary topology is
detected once from the first field file and reused for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import boundary, naming, pvd, surface, volume
from .casefile import read_casefile
from .re2boundary import read_re2_boundaries
from .regions import format_region_table


@dataclass
class Config:
    casefile: Path
    re2file: Path
    outdir: Path
    write_volume: bool = False
    write_boundaries: bool = True
    normal_angle_deg: float = 40.0
    split_by_sideset: bool = True
    interactive: bool = True
    reconfigure: bool = False
    dtype: str = "single"          # 'single' or 'double'
    dedup_decimals: int = 9
    max_files: Optional[int] = None


def _log(comm, msg: str) -> None:
    if comm.Get_rank() == 0:
        print(msg, flush=True)


def _field_dict(fld) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in fld.registry.keys():
        out[name] = np.asarray(fld.registry[name])
    return out


def _field_time(fld, msh, step: int) -> float:
    for obj in (fld, msh):
        t = getattr(obj, "t", None)
        if t is not None:
            try:
                return float(t)
            except (TypeError, ValueError):
                pass
    return float(step)


def run(config: Config, comm) -> None:
    rank = comm.Get_rank()

    # ---- rank 0: read case + re2 boundary hints ---------------------------
    if rank == 0:
        case = read_casefile(config.casefile)
        field_files = case.field_files()
        present = [f for f in field_files if f.is_file()]
        missing = len(field_files) - len(present)
        if missing:
            print(f"WARNING: {missing} of {len(field_files)} field files listed "
                  f"in {config.casefile.name} are missing; converting the "
                  f"{len(present)} present ones.", flush=True)
        field_files = present
        if config.max_files:
            field_files = field_files[: config.max_files]
        if not field_files:
            raise FileNotFoundError("No field files found to convert.")

        print(f"Reading boundary codes from {config.re2file.name} ...", flush=True)
        re2b = read_re2_boundaries(config.re2file)
        print(f"  {re2b.nfaces} boundary faces, {re2b.nelgt} elements in mesh.",
              flush=True)
        payload = {
            "field_files": [str(f) for f in field_files],
            "casename": case.casename,
            "casedir": str(config.casefile.parent),
            "nelgt": re2b.nelgt,
        }
        re2_centers = re2b.centers
        re2_normals = re2b.normals
        re2_codes = re2b.codes
    else:
        payload = None
        re2_centers = None
        re2_normals = None
        re2_codes = None

    payload = comm.bcast(payload, root=0)
    field_files = [Path(f) for f in payload["field_files"]]
    casename = payload["casename"]
    casedir = Path(payload["casedir"])
    nelgt_re2 = payload["nelgt"]

    # ---- output directories ----------------------------------------------
    outdir = config.outdir
    vol_dir = outdir / "volume"
    surf_dir = outdir / "boundaries"
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        if config.write_volume:
            vol_dir.mkdir(exist_ok=True)
        if config.write_boundaries:
            surf_dir.mkdir(exist_ok=True)
    comm.Barrier()

    np_dtype = np.single if config.dtype == "single" else np.double

    # heavy readers
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.io.ppymech.neksuite import pynekread

    plan: Optional[boundary.BoundaryPlan] = None
    names: Dict[int, str] = {}
    vol_entries: List = []
    surf_entries: Dict[int, List] = {}
    coords: Optional[List[np.ndarray]] = None  # cached mesh (written once by NekRS)

    for fi, ffile in enumerate(field_files):
        step = fi + 1
        _log(comm, f"[{fi + 1}/{len(field_files)}] {ffile.name}")

        msh = Mesh(comm, create_connectivity=False)
        fld = FieldRegistry(comm)
        pynekread(str(ffile), comm, msh=msh, fld=fld)

        # NekRS writes the mesh only in the first field file; later files carry
        # fields only. Cache the geometry and reuse it.
        has_geom = hasattr(msh, "glb_nelv") and getattr(msh, "x", None) is not None
        if has_geom:
            if int(msh.glb_nelv) != int(nelgt_re2):
                raise ValueError(
                    f"Element count mismatch: {config.re2file.name} has "
                    f"{nelgt_re2} elements but {ffile.name} has "
                    f"{int(msh.glb_nelv)}."
                )
            coords = [np.asarray(msh.x), np.asarray(msh.y), np.asarray(msh.z)]
        elif coords is None:
            raise ValueError(
                f"{ffile.name} contains no mesh geometry and no earlier field "
                "file provided one. The first converted file must include the "
                "mesh."
            )

        fields = _field_dict(fld)
        field_names = list(fields.keys())
        time = _field_time(fld, msh, step)

        # ---- detect boundaries once (first file) --------------------------
        if config.write_boundaries and plan is None:
            plan = boundary.detect_boundaries(
                comm, coords,
                re2_centers if rank == 0 else None,
                re2_normals if rank == 0 else None,
                re2_codes if rank == 0 else None,
                normal_angle_deg=config.normal_angle_deg,
                split_by_sideset=config.split_by_sideset,
            )
            surf_entries = {r: [] for r in range(plan.nreg)}
            if rank == 0:
                print(f"Detected {plan.total_faces} boundary faces in "
                      f"{plan.nreg} regions (from field geometry).", flush=True)
                dom = _domain_scale(coords)
                if plan.match_dist_median > 1e-3 * dom:
                    print(f"  NOTE: nearest .re2 code match distance is large "
                          f"(median {plan.match_dist_median:.3g}); the mesh may be "
                          f"deformed away from the .re2. Check/edit the names.",
                          flush=True)
                cfg_path = naming.config_path(casedir, casename)
                names = naming.resolve_names(
                    plan.infos, cfg_path, casename,
                    interactive=config.interactive,
                    reconfigure=config.reconfigure,
                )
                print("\nBoundary region -> output name:")
                print(format_region_table(plan.infos))
                for info in plan.infos:
                    print(f"    region {info.region_id}  ->  {names[info.region_id]}")
                print(flush=True)
            names = comm.bcast(names, root=0)

        # ---- volume -------------------------------------------------------
        if config.write_volume:
            vpath = vol_dir / f"{casename}_volume_{step:05d}.vtkhdf"
            volume.write_volume_vtkhdf(comm, vpath, coords, fields, dtype=np_dtype)
            if rank == 0:
                vol_entries.append((time, f"volume/{vpath.name}"))

        # ---- boundaries ---------------------------------------------------
        if config.write_boundaries and plan is not None:
            for r in range(plan.nreg):
                rmask = plan.loc_region == r
                piece = surface.extract_region_surface(
                    r, plan.loc_elem[rmask], plan.loc_face[rmask], coords, fields
                )
                gathered = comm.gather(piece, root=0)
                if rank == 0:
                    merged = surface.merge_pieces(gathered, field_names)
                    merged = surface.dedup_points(
                        merged, field_names, decimals=config.dedup_decimals
                    )
                    name = naming.sanitize(names[r])
                    rdir = surf_dir / name
                    rdir.mkdir(exist_ok=True)
                    spath = rdir / f"{casename}_{name}_{step:05d}.vtp"
                    npts = surface.write_vtp(merged, field_names, spath)
                    surf_entries[r].append((time, f"boundaries/{name}/{spath.name}"))
                    _log(comm, f"    {name}: {merged['quads'].shape[0]} quads, "
                               f"{npts} points -> {spath.name}")

        del msh, fld, fields  # keep cached `coords` for later files

    # ---- pvd collections --------------------------------------------------
    if rank == 0:
        if config.write_volume and vol_entries:
            pvd.write_pvd(outdir / f"{casename}_volume.pvd", vol_entries)
        if config.write_boundaries and plan is not None:
            for r in range(plan.nreg):
                if surf_entries.get(r):
                    name = naming.sanitize(names[r])
                    pvd.write_pvd(outdir / f"{casename}_{name}.pvd", surf_entries[r])
        print(f"\nDone. Output written to {outdir}", flush=True)


def _domain_scale(coords) -> float:
    X, Y, Z = coords
    dx = float(X.max() - X.min())
    dy = float(Y.max() - Y.min())
    dz = float(Z.max() - Z.min())
    return max(dx, dy, dz, 1e-30)
