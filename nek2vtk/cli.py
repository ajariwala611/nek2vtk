"""Command-line interface for nek2vtk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .convert import Config, run


def _find_default_casefile(cwd: Path) -> Path | None:
    """Find a single ``.nek5000`` file in ``cwd``, else None."""
    candidates = sorted(cwd.glob("*.nek5000"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nek2vtk",
        description="Convert Nek5000/NekRS field data (.f) + mesh (.re2) into "
                    "per-boundary VTP surfaces and a VTKHDF volume, with "
                    "boundary connectivity that the .f files lack.",
    )
    p.add_argument(
        "case", nargs="?", default=None,
        help="Path to CASE.nek5000 (default: the single *.nek5000 in the "
             "current directory).",
    )
    p.add_argument(
        "--re2", default=None,
        help="Path to the .re2 mesh file (default: <casename>.re2 next to the "
             "case file).",
    )
    p.add_argument(
        "-o", "--outdir", default=None,
        help="Output directory (default: <casedir>/vtk).",
    )
    p.add_argument("--no-volume", action="store_true",
                   help="Skip the full-volume VTKHDF export.")
    p.add_argument("--no-boundaries", action="store_true",
                   help="Skip the per-boundary VTP export.")
    p.add_argument("--normal-angle", type=float, default=40.0,
                   help="Max angle (deg) between adjacent face normals for them "
                        "to be joined into one boundary region (default: 40).")
    p.add_argument("--non-interactive", action="store_true",
                   help="Do not prompt for boundary names; use saved config or "
                        "auto-generated defaults. (Forced under MPI with >1 rank.)")
    p.add_argument("--reconfigure", action="store_true",
                   help="Ignore any existing boundary-name config and re-create it.")
    p.add_argument("--dtype", choices=["single", "double"], default="single",
                   help="Output floating point precision (default: single).")
    p.add_argument("--max-files", type=int, default=None,
                   help="Only convert the first N field files (for testing).")
    return p


def main(argv=None) -> int:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = build_parser().parse_args(argv)

    # Resolve the case file (rank 0 decides, then everyone re-derives paths).
    if args.case is not None:
        casefile = Path(args.case).expanduser().resolve()
    else:
        found = _find_default_casefile(Path.cwd())
        if found is None:
            if rank == 0:
                print("error: no case file given and could not find a unique "
                      "*.nek5000 in the current directory.", file=sys.stderr)
            return 2
        casefile = found.resolve()

    casedir = casefile.parent
    casename = casefile.stem

    re2file = (Path(args.re2).expanduser().resolve()
               if args.re2 else casedir / f"{casename}.re2")
    outdir = (Path(args.outdir).expanduser().resolve()
              if args.outdir else casedir / "vtk")

    interactive = (not args.non_interactive) and size == 1 and sys.stdin.isatty()
    if size > 1 and not args.non_interactive and rank == 0:
        print("Running under MPI with >1 rank: boundary naming is "
              "non-interactive. Run once serially first to create the "
              "boundary-name config, or edit the JSON.", flush=True)

    config = Config(
        casefile=casefile,
        re2file=re2file,
        outdir=outdir,
        write_volume=not args.no_volume,
        write_boundaries=not args.no_boundaries,
        normal_angle_deg=args.normal_angle,
        interactive=interactive,
        reconfigure=args.reconfigure,
        dtype=args.dtype,
        max_files=args.max_files,
    )

    try:
        run(config, comm)
    except Exception as exc:  # noqa: BLE001
        # ensure a clean, single error message and non-zero exit under MPI
        if rank == 0:
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        comm.Barrier()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
