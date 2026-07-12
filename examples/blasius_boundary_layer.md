# Example: flat-plate Blasius boundary layer (NekRS)

This is the case used to develop and validate `nek2vtk`. It is a 3D flat-plate
boundary layer generated with `genbox` and run with NekRS:

- 40,000 spectral elements (100 × 20 × 20), polynomial order 7 (8×8×8 GLL).
- Boundary codes in the `.re2`: `v` (inlet), `o` (outlet **and** freestream
  top — same code), `W` (no-slip wall), `P` (spanwise periodic).
- The mesh is deformed at run time (`usrdat` clusters points toward the wall
  with a `tanh` map) and NekRS repartitions the elements — which is exactly why
  a `.re2`-index-based converter fails and `nek2vtk` detects boundaries from the
  field geometry instead.

## Run

```bash
cd /path/to/blasius_boundary_layer
nek2vtk flat_plate.nek5000            # prompts for boundary names on first run
```

Six regions are detected. Suggested names (identify them from the printed
`centroid` / `normal` / `bbox`):

| region | code | faces | location            | good name    |
|-------:|:----:|------:|---------------------|--------------|
| wall   |  W   |  2000 | y = 0               | `wall`       |
| top    |  o   |  2000 | y = 5, normal +y    | `top`        |
| outlet |  o   |   400 | x = 10, normal +x   | `outlet`     |
| inlet  |  v   |   400 | x = 0, normal −x    | `inlet`      |
| perio. |  P   |  2000 | z = 0               | `periodic_z0`|
| perio. |  P   |  2000 | z = 2.5             | `periodic_z1`|

Note the two `o` boundaries (top and outlet) are correctly separated even
though they share a code, because they meet at a right angle.

## Validation checks (what "correct" looks like)

- **Wall** `velocity_magnitude` ≈ 0 everywhere (no-slip), all points at y = 0.
- **Inlet** `u` follows the Blasius profile: 0 at the wall rising to ≈ 1 in the
  freestream.
- **Volume** `.vtkhdf` has 40000 × 8³ = 20,480,000 points and
  40000 × 7³ = 13,720,000 hexahedra.

## Open in ParaView

Open `vtk/flat_plate_wall.pvd` (or any boundary `.pvd`) for the wall surface
with data over the time series, or `vtk/flat_plate_volume.pvd` for the volume.
