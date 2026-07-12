# nek2vtk

Convert **Nek5000 / NekRS** field data into ParaView-friendly VTK files —
**with the boundary information the field files don't carry.**

Nek field files (`*.f00001`, …) store the volume solution on the spectral-element
GLL grid but **no boundary connectivity**. To pull out a wall or an inlet in
ParaView you normally have to iso-contour by hand (e.g. "velocity magnitude = 0"
for a no-slip wall), which is fiddly and imprecise. `nek2vtk`:

1. reads the case file (`CASE.nek5000`) to find the time series,
2. reads the boundary-condition codes from the `.re2` mesh,
3. checks the element count in the mesh and field files match,
4. **detects each physical boundary directly from the field geometry** and lets
   you **name** them (inlet, outlet, wall, …),
5. writes **one `.vtp` surface per boundary** (carrying all the field data),
   with a `.pvd` time series for each — and, optionally (`--volume`), the
   **full volume as `.vtkhdf`**.

So instead of contouring, you just open `flat_plate_wall.pvd` and you have the
wall — with `u`, `v`, `w`, `p`, `velocity`, `velocity_magnitude` on it — for the
whole time series.

---

## Why boundaries are detected from the field, not mapped from the `.re2`

The obvious approach — read boundary faces from the `.re2` and copy the data
from the matching field element — **does not work for NekRS**. NekRS repartitions
the elements (so element *N* in the field file is not element *N* in the `.re2`)
and can deform the mesh at run time (`usrdat`), so the `.re2` geometry no longer
matches the field. The element-map stored in the field header is written as an
unreliable identity by NekRS.

`nek2vtk` therefore treats the **field file as ground truth**:

- A face is on a boundary **iff it belongs to a single element** (interior faces
  are shared by two). This is detected topologically by hashing face centres —
  robust to any reordering or deformation, and correct in parallel.
- Boundary faces are grouped into **regions** by shared-edge connectivity plus an
  outward-normal angle test. This separates, e.g., an outlet and a freestream
  top that share the Nek code `o` but meet at a sharp edge, while keeping a
  smoothly curved wall (an airfoil) as one surface.
- The `.re2` boundary **codes / sideset ids** are used as *labels*, matched to
  each field face by position **and** orientation (so a periodic face near a
  wall matches the periodic patch, not the closer wall). For meshes from
  `genbox` the label is the Nek code (`W`, `v`, `o`, `P`); for `gmsh2nek` /
  `exo2nek`, which write the generic code `MSH` and store the real sideset
  number in the bc parameters, the label is that number (`bc1`, `bc2`, …).
- By default each geometric region is then **sub-split by label**, so distinct
  sidesets that are geometrically connected (e.g. several far-field patches of a
  C-mesh meeting at shallow angles) are kept separate — while a handful of
  stray label mis-matches are folded into the region majority so they can't
  fragment a boundary. Pass `--no-sideset-split` for a purely geometric split.

---

## Installation

Requires Python ≥ 3.9 and these packages (all pip/conda-installable):

```
numpy  scipy  pymech  pyvista  mpi4py  h5py  pysemtools
```

Then, from the repository root:

```bash
pip install -e .
```

This installs the `nek2vtk` command. (You can also run it without installing via
`python -m nek2vtk.cli`.)

> **Parallel volume writes** need an MPI-enabled build of `h5py` (parallel
> HDF5). If you don't have one, `nek2vtk` still runs in parallel — it just
> gathers the volume to the root rank and writes it there (see *MPI* below).

---

## Quick start

From inside a case directory (the one holding `CASE.nek5000` and `CASE.re2`):

```bash
nek2vtk
```

With no arguments it finds the single `*.nek5000` file, uses `<case>.re2` for
the mesh, and writes output to `./vtk/`. On the first run it prints the detected
boundaries and asks you to name each one:

```
Detected boundary regions:
  id  code  nfaces  centroid (x,y,z)           normal               bbox
  ----------------------------------------------------------------------
   0  W       2000  (5, 0, 1.25)               (+0.00,-1.00,+0.00)  x[0,10] y[0,0] z[0,2.5]
   1  P       2000  (5, 1.14, 0)               (+0.00,+0.00,-1.00)  x[0,10] y[0,5] z[0,0]
   2  P       2000  (5, 1.14, 2.5)             (+0.00,+0.00,+1.00)  x[0,10] y[0,5] z[2.5,2.5]
   3  o       2000  (5, 5, 1.25)               (+0.00,+1.00,+0.00)  x[0,10] y[5,5] z[0,2.5]
   4  v        400  (0, 1.14, 1.25)            (-1.00,+0.00,+0.00)  x[0,0] y[0,5] z[0,2.5]
   5  o        400  (10, 1.14, 1.25)           (+1.00,+0.00,+0.00)  x[10,10] y[0,5] z[0,2.5]

Name each region (press Enter to accept the [default]).
  region 0 (code 'W', 2000 faces) name [wall]:
  region 1 (code 'P', 2000 faces) name [periodic]: periodic_z0
  ...
```

Use the `centroid`, `normal` and `bbox` columns to identify each surface (here
region 3 is the freestream top and region 5 is the real outlet — both coded `o`).

Names are saved to `<case>_boundaries.json` next to the case file, so **later
runs reuse them automatically** (no prompting). Edit that JSON any time to
rename, then re-run.

### Output layout

```
vtk/
├── flat_plate_wall.pvd                    # one .pvd per boundary
├── flat_plate_inlet.pvd
├── ...
├── flat_plate_volume.pvd                  # only with --volume
├── boundaries/
│   ├── wall/  flat_plate_wall_00001.vtp
│   ├── inlet/ flat_plate_inlet_00001.vtp
│   └── ...
└── volume/                                # only with --volume
    └── flat_plate_volume_00001.vtkhdf     # one per timestep
```

In ParaView, open any `.pvd` to load that surface (or the volume) as an
animatable time series.

---

## Command-line options

```
nek2vtk [CASE.nek5000] [options]

  CASE.nek5000        Case file (default: the single *.nek5000 in the cwd)
  --re2 PATH          Mesh file (default: <casename>.re2 beside the case file)
  -o, --outdir DIR    Output directory (default: <casedir>/vtk)

  --volume            Also export the full volume as VTKHDF (off by default;
                      the file is several times larger than the Nek .f data)
  --no-boundaries     Skip the per-boundary VTP export
  --normal-angle DEG  Max angle between adjacent face normals to merge them into
                      one region (default: 40). Lower = split more aggressively.
  --no-sideset-split  Purely geometric split; do not sub-split regions by the
                      .re2 sideset/code label (gives fewer, merged regions)
  --non-interactive   Don't prompt; use the saved config or auto defaults
  --reconfigure       Ignore an existing name config and rebuild it
  --dtype {single,double}   Output precision (default: single)
  --max-files N       Only convert the first N timesteps (handy for testing)
```

---

## MPI / large cases

Reading uses `pysemtools`, which distributes elements across ranks:

```bash
mpiexec -n 8 nek2vtk flat_plate.nek5000 --non-interactive
```

- Under MPI, **boundary naming is non-interactive** — run once serially first
  (or with `-n 1`) to create `<case>_boundaries.json`, then launch the parallel
  job, which reuses it. (Or edit the JSON by hand.)
- The **boundary VTPs are extracted in parallel** and gathered to the root rank
  to write one clean, de-duplicated `.vtp` per boundary.
- The **volume** (only with `--volume`) is written in parallel if your `h5py`
  has MPI support; otherwise it is gathered to the root rank and written there.
  For very large meshes, install an MPI-enabled `h5py`, or just leave the volume
  off (the default) and export only the boundaries in parallel.

> Use your MPI launcher that matches the `mpi4py` you installed (e.g. the one in
> your conda env), not an unrelated `mpiexec` from another application.

---

## How the high-order data is represented

Each Nek element face is an `N×N` grid of GLL points (order-7 → 8×8). `nek2vtk`
tessellates every face into `(N-1)×(N-1)` flat quads (**subdivided-linear**
surfaces), carrying every GLL value as point data. At order 7 this is visually
smooth and works with every ParaView filter. Coincident GLL points shared
between neighbouring faces are merged, so the surfaces are watertight.

The volume uses the same idea in 3D (via `pysemtools`): each spectral element of
order `N` becomes `N³` linear hexahedra in the `.vtkhdf` file.

---

## Assumptions & limitations

- **Static mesh**: boundary topology is detected once (from the first field
  file) and reused for the series. For moving/deforming-in-time meshes the
  boundary *connectivity* is assumed constant (geometry still updates per step).
- Boundary **codes** are naming hints only; if the mesh is deformed far from the
  `.re2`, the auto-suggested names may be off — the tool warns you and you can
  rename via the JSON or interactively.
- Two distinct physical boundaries that are **coplanar, edge-adjacent and share
  the same normal** will be detected as one region (they are geometrically
  indistinguishable). Split them afterwards in ParaView if needed.
- Periodic faces (`P`) are real mesh boundaries and are exported like any other
  boundary; ignore or delete the ones you don't want.

---

## Project layout

```
nek2vtk/
├── casefile.py     parse CASE.nek5000
├── re2boundary.py  read .re2 boundary codes (naming hints) via pymech
├── nekfaces.py     Nek hex face conventions (face id -> GLL slice)
├── geometry.py     vectorised face corners / centroids / normals
├── boundary.py     topological boundary detection + region split (MPI)
├── regions.py      connected-component splitter + region summaries
├── naming.py       interactive naming + JSON config load/save
├── surface.py      face tessellation -> merged .vtp per region
├── volume.py       full volume -> .vtkhdf (via pysemtools)
├── pvd.py          ParaView .pvd time-series writer
├── convert.py      end-to-end driver
└── cli.py          command-line entry point
```

## License

MIT — see [LICENSE](LICENSE).
