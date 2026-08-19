"""Checks the cutter on a real model. Needs only numpy.

    python3 tests/test_plane_cut.py [model.3mf]

Defaults to bottom_shelf.3mf next to the repository root. Any watertight 3mf
will do - the checks are all relative to whatever model is handed in.
"""

import os
import re
import sys
import zipfile
from collections import Counter

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "MeshCutter"))
from PlaneCut import cutMesh

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), os.pardir, "bottom_shelf.3mf")
UNITS = {"meter": 1000.0, "millimeter": 1.0, "micron": 0.001,
         "centimeter": 10.0, "inch": 25.4, "foot": 304.8}


def loadThreeMF(path):
    xml = zipfile.ZipFile(path).read("3D/3dmodel.model").decode()
    scale = UNITS[re.search(r'unit="(\w+)"', xml).group(1)]
    vertices = numpy.array(re.findall(
        r'<vertex x="([-\d.eE+]+)" y="([-\d.eE+]+)" z="([-\d.eE+]+)"', xml), dtype=numpy.float64) * scale
    faces = numpy.array(re.findall(
        r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', xml), dtype=numpy.int64)
    return vertices, faces


def volume(triangle_soup):
    triangles = triangle_soup.reshape(-1, 3, 3)
    return float(numpy.einsum("ij,ij->i", triangles[:, 0],
                              numpy.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def openEdges(triangle_soup, tolerance = 1e-7):
    """Count edges not shared by exactly two triangles: 0 means watertight."""
    triangles = triangle_soup.reshape(-1, 3, 3)
    quantised = numpy.round(triangles.reshape(-1, 3) / tolerance).astype(numpy.int64)
    _, corners = numpy.unique(quantised, axis=0, return_inverse=True)
    corners = corners.reshape(-1, 3)
    tally = Counter()
    for a, b, c in corners:
        for first, second in ((a, b), (b, c), (c, a)):
            tally[(min(first, second), max(first, second))] += 1
    return sum(1 for count in tally.values() if count != 2)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if not os.path.exists(model):
        print("No model to test against: {0} is missing.\n"
              "Pass a watertight 3mf: python3 tests/test_plane_cut.py my_model.3mf".format(model))
        return 1

    vertices, faces = loadThreeMF(model)
    whole = volume(vertices[faces].reshape(-1, 3))
    assert openEdges(vertices[faces].reshape(-1, 3)) == 0, "test model is not watertight to begin with"

    failures = 0
    generator = numpy.random.default_rng(11)
    for step in range(60):
        if step % 4 == 3:
            normal = generator.normal(size = 3)
            normal /= numpy.linalg.norm(normal)
        else:
            normal = numpy.zeros(3)
            normal[step % 3] = 1.0
        along = vertices.dot(normal)
        offset = float(generator.uniform(along.min(), along.max()))

        lower, upper = cutMesh(vertices, faces, normal, offset)
        problems = []
        if not len(lower) or not len(upper):
            problems.append("one half came out empty")
        if abs((volume(lower) + volume(upper)) / whole - 1) > 1e-9:
            problems.append("volume not preserved")
        for name, half in (("lower", lower), ("upper", upper)):
            open_count = openEdges(half)
            if open_count:
                problems.append("{0} half has {1} open edges".format(name, open_count))
        if problems:
            failures += 1
            print("FAIL normal={0} offset={1:.2f}: {2}".format(normal.round(3), offset, "; ".join(problems)))

    print("{0} of 60 cuts produced two watertight halves".format(60 - failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
