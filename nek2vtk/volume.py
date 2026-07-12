"""Write the full spectral-element volume to VTKHDF.

This reuses pysemtools' VTKHDF writer, which builds the sub-element hexahedral
connectivity for the GLL grid internally.  Each spectral element of order
``N`` becomes ``N**3`` linear hexahedra, so no accuracy is lost beyond the
usual linear-between-GLL-points visualization.

Parallel writing of a single VTKHDF file needs an MPI-enabled build of
``h5py`` (parallel HDF5).  When that is unavailable, running on more than one
rank would deadlock (every rank tries to open the same file with the serial
driver), so we instead gather the volume to the root rank and write it there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def _h5py_has_mpi() -> bool:
    try:
        import h5py

        return bool(h5py.get_config().mpi)
    except Exception:  # noqa: BLE001
        return False


def _augment(fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    data = dict(fields)
    if all(c in data for c in ("u", "v", "w")):
        data["velocity_magnitude"] = np.sqrt(
            data["u"] ** 2 + data["v"] ** 2 + data["w"] ** 2
        )
    return data


def _write(comm, path: Path, coords, data_dict, dtype, parallel_io: bool) -> None:
    from pysemtools.io.wrappers import write_data

    write_data(
        comm,
        str(path),
        data_dict,
        parallel_io=parallel_io,
        dtype=dtype,
        msh=[coords[0], coords[1], coords[2]],
        write_mesh=True,
        distributed_axis=0,
    )


def write_volume_vtkhdf(comm, path: Path, coords: List[np.ndarray],
                        fields: Dict[str, np.ndarray], dtype=np.single) -> None:
    """Write a single-timestep volume ``.vtkhdf`` file (MPI aware)."""
    size = comm.Get_size()
    data_dict = _augment(fields)

    if size == 1:
        _write(comm, path, coords, data_dict, dtype, parallel_io=False)
        return

    if _h5py_has_mpi():
        _write(comm, path, coords, data_dict, dtype, parallel_io=True)
        return

    # Fallback: gather to root and write serially (no parallel HDF5 available).
    from mpi4py import MPI

    rank = comm.Get_rank()
    gx = comm.gather(coords[0], root=0)
    gy = comm.gather(coords[1], root=0)
    gz = comm.gather(coords[2], root=0)
    gfields = {k: comm.gather(v, root=0) for k, v in data_dict.items()}

    if rank == 0:
        full_coords = [np.concatenate(gx, axis=0),
                       np.concatenate(gy, axis=0),
                       np.concatenate(gz, axis=0)]
        full_fields = {k: np.concatenate(v, axis=0) for k, v in gfields.items()}
        _write(MPI.COMM_SELF, path, full_coords, full_fields, dtype,
               parallel_io=False)
    comm.Barrier()
