"""nek2vtk - convert Nek5000/NekRS data to VTK with boundary information.

The Nek field files (``.f#####``) contain no boundary connectivity, so
extracting a wall or inlet in ParaView normally means iso-contouring by hand.
nek2vtk reads the boundary codes stored in the ``.re2`` mesh, reconstructs each
physical boundary as a surface, and writes it (with all field data) to a
``.vtp`` file - plus the full volume as ``.vtkhdf``.
"""

__version__ = "0.1.0"

from .casefile import CaseFile, read_casefile  # noqa: F401
from .convert import Config, run  # noqa: F401
