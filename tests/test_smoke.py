"""Lightweight smoke tests that need no data files.

Run with:  python -m pytest tests/   (or)   python tests/test_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nek2vtk import nekfaces
from nek2vtk.casefile import CaseFile
from nek2vtk.regions import region_infos, split_faces
from nek2vtk.surface import _quad_connectivity


def test_face_slice_planes():
    """Each Nek face maps to a constant-index plane of an (lz,ly,lx) array."""
    lx = ly = lz = 4
    # face 1 -> jy=0, face 3 -> jy=max, face 4 -> ix=0, face 2 -> ix=max
    assert nekfaces.face_slice(1, lx, ly, lz) == (slice(None), 0, slice(None))
    assert nekfaces.face_slice(4, lx, ly, lz) == (slice(None), slice(None), 0)
    assert nekfaces.face_slice(5, lx, ly, lz) == (0, slice(None), slice(None))


def test_quad_connectivity_counts():
    q = _quad_connectivity(4, 3)
    assert q.shape == (3 * 2, 4)
    # first quad is the (0,0) cell
    assert list(q[0]) == [0, 1, 4, 3]


def test_casefile_field_files():
    cf = CaseFile(path=__import__("pathlib").Path("/tmp/foo.nek5000"),
                  filetemplate="foo%01d.f%05d", firsttimestep=1, numtimesteps=3)
    files = [p.name for p in cf.field_files(root=__import__("pathlib").Path("/tmp"))]
    assert files == ["foo0.f00001", "foo0.f00002", "foo0.f00003"]


def test_split_two_perpendicular_planes():
    """Two unit quads meeting at a right angle must become two regions."""
    # quad A on z=0 plane (normal -z), quad B on x=1 plane (normal +x),
    # sharing the edge x=1,z=0.
    A = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    B = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], float)
    corners = np.stack([A, B])
    normals = np.array([[0, 0, -1], [1, 0, 0]], float)
    region = split_faces(corners, normals, normal_angle_deg=40.0)
    assert len(np.unique(region)) == 2


def test_split_two_coplanar_quads_merge():
    """Two coplanar quads sharing an edge become one region."""
    A = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    B = np.array([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], float)
    corners = np.stack([A, B])
    normals = np.array([[0, 0, 1], [0, 0, 1]], float)
    region = split_faces(corners, normals, normal_angle_deg=40.0)
    assert len(np.unique(region)) == 1
    infos = region_infos(corners, normals, region,
                         codes=np.array(["W", "W"], dtype="<U3"))
    assert len(infos) == 1 and infos[0].code == "W" and infos[0].nfaces == 2


def test_re2_header_parse_and_cht(tmp_path=None):
    """Header parser reads nelgt/ndim/nelgv; CHT is when nelgv < nelgt."""
    import tempfile
    from nek2vtk.re2boundary import read_re2_header

    def _write(hdr_text):
        raw = hdr_text.encode("ascii")
        raw = raw.ljust(80)[:80]
        f = tempfile.NamedTemporaryFile(suffix=".re2", delete=False)
        f.write(raw)
        f.close()
        return f.name

    # pure fluid
    p = _write("#v004    40000  3    40000   1 hdr")
    assert read_re2_header(p) == (40000, 3, 40000)
    # conjugate heat transfer: fewer fluid than total
    p = _write("#v004     1000  3      600   1 hdr")
    nelgt, ndim, nelgv = read_re2_header(p)
    assert (nelgt, ndim, nelgv) == (1000, 3, 600)
    assert nelgv < nelgt  # -> is_cht


def test_face_labels_all_converters():
    from nek2vtk.re2boundary import _face_label
    assert _face_label("MSH", 3) == "bc3"   # gmsh2nek
    assert _face_label("EXO", 2) == "bc2"   # exo2nek
    assert _face_label("CGN", 5) == "bc5"   # cgns2nek
    assert _face_label("P", 6) == "P"       # periodic stays P
    assert _face_label("W", 0) == "W"       # genbox code kept


def test_mesh_export_helpers():
    """Point de-dup merges coincident corners; VTK hex order has 8 nodes."""
    from nek2vtk.mesh_export import _VTK_HEX, _dedup

    assert len(_VTK_HEX) == 8
    # two quads sharing an edge -> 6 unique points, connectivity remapped
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],   # quad A
        [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0],   # quad B (shares 2 pts)
    ], float)
    upts, inv = _dedup(pts)
    assert upts.shape[0] == 6
    assert inv.shape[0] == 8
    # the shared corners map to the same unique index
    assert inv[1] == inv[4]  # (1,0,0)
    assert inv[2] == inv[7]  # (1,1,0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all smoke tests passed")
