"""Checks the cutter against watertight test shapes. Needs only numpy.

    python3 tests/test_plane_cut.py [model.3mf ...]

With no arguments it builds its own shapes: a plain cube, and a square frame
whose cross-section has a hole in it and walls thin enough to be awkward to
triangulate. Any watertight 3mf can be passed instead, or as well - the checks
are all relative to whatever model is handed in.
"""

import os
import re
import sys
import zipfile
from collections import Counter

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "MeshCutter"))
from PlaneCut import cutMesh

CUTS_PER_SHAPE = 60
UNITS = {"meter": 1000.0, "millimeter": 1.0, "micron": 0.001,
         "centimeter": 10.0, "inch": 25.4, "foot": 304.8}


def loadThreeMF(path):
    xml = zipfile.ZipFile(path).read("3D/3dmodel.model").decode()
    scale = UNITS[re.search(r'unit="(\w+)"', xml).group(1)]
    vertices = numpy.array(re.findall(
        r'<vertex x="([-\d.eE+]+)" y="([-\d.eE+]+)" z="([-\d.eE+]+)"', xml), dtype = numpy.float64) * scale
    faces = numpy.array(re.findall(
        r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', xml), dtype = numpy.int64)
    return vertices, faces


def makeCube(size = 40.0):
    """A cube, off-centre and off-axis-aligned in size, as the simplest case."""
    corners = numpy.array([[x, y, z] for x in (0.0, size) for y in (0.0, size * 0.75)
                           for z in (0.0, size * 1.5)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = [triangle for quad in quads
             for triangle in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3]))]
    return corners, numpy.array(faces, dtype = numpy.int64)


def makeFrame(outer = 40.0, wall = 2.5, height = 30.0):
    """A square tube: its cross-section is a thin ring, so cutting across it
    produces an outline with a hole in it and walls only a couple of triangles
    apart - the case that plain ear clipping gets wrong."""
    corners = 4
    outer_loop = _square(outer / 2.0)
    inner_loop = _square(outer / 2.0 - wall)
    vertices = ([[x, y, 0.0] for x, y in outer_loop] + [[x, y, 0.0] for x, y in inner_loop] +
                [[x, y, height] for x, y in outer_loop] + [[x, y, height] for x, y in inner_loop])
    outer_bottom, inner_bottom, outer_top, inner_top = 0, corners, 2 * corners, 3 * corners

    faces = []
    for corner in range(corners):
        following = (corner + 1) % corners

        def edge(base):
            return base + corner, base + following

        # Outer wall faces away from the axis, the wall around the hole faces
        # back towards it, and the two end rings face out along Z.
        (here, ahead), (above, above_ahead) = edge(outer_bottom), edge(outer_top)
        faces += [(here, ahead, above_ahead), (here, above_ahead, above)]

        (here, ahead), (above, above_ahead) = edge(inner_bottom), edge(inner_top)
        faces += [(here, above, above_ahead), (here, above_ahead, ahead)]

        (outside, outside_ahead), (inside, inside_ahead) = edge(outer_bottom), edge(inner_bottom)
        faces += [(outside, inside, inside_ahead), (outside, inside_ahead, outside_ahead)]

        (outside, outside_ahead), (inside, inside_ahead) = edge(outer_top), edge(inner_top)
        faces += [(outside, outside_ahead, inside_ahead), (outside, inside_ahead, inside)]

    return numpy.array(vertices, dtype = numpy.float64), numpy.array(faces, dtype = numpy.int64)


def _square(radius):
    return [(-radius, -radius), (radius, -radius), (radius, radius), (-radius, radius)]


def volume(triangle_soup):
    triangles = triangle_soup.reshape(-1, 3, 3)
    return float(numpy.einsum("ij,ij->i", triangles[:, 0],
                              numpy.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def openEdges(triangle_soup, tolerance = 1e-7):
    """Count unsound edges: 0 means watertight and consistently wound.

    Every edge of a closed surface must be walked exactly once in each
    direction. Counting directions rather than just neighbours also catches a
    face that sits on the right edge but faces the wrong way.
    """
    triangles = triangle_soup.reshape(-1, 3, 3)
    quantised = numpy.round(triangles.reshape(-1, 3) / tolerance).astype(numpy.int64)
    _, corners = numpy.unique(quantised, axis = 0, return_inverse = True)
    corners = corners.reshape(-1, 3)
    tally = Counter()
    for a, b, c in corners:
        for first, second in ((a, b), (b, c), (c, a)):
            tally[(first, second)] += 1
    return sum(1 for (first, second), count in tally.items()
               if count != 1 or tally[(second, first)] != 1)


def checkShape(name, vertices, faces):
    """Cut one shape every which way and report how many halves came out sound."""
    whole = volume(vertices[faces].reshape(-1, 3))
    if openEdges(vertices[faces].reshape(-1, 3)):
        print("{0}: SKIPPED, the shape isn't watertight to begin with".format(name))
        return 1

    failures = 0
    generator = numpy.random.default_rng(11)
    for step in range(CUTS_PER_SHAPE):
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
        for half_name, half in (("lower", lower), ("upper", upper)):
            open_count = openEdges(half)
            if open_count:
                problems.append("{0} half has {1} open edges".format(half_name, open_count))
        if problems:
            failures += 1
            print("{0}: FAIL normal={1} offset={2:.2f}: {3}".format(
                name, normal.round(3), offset, "; ".join(problems)))

    print("{0}: {1} of {2} cuts produced two watertight halves".format(
        name, CUTS_PER_SHAPE - failures, CUTS_PER_SHAPE))
    return failures


def main():
    shapes = [("cube", makeCube()), ("frame", makeFrame())]
    for path in sys.argv[1:]:
        shapes.append((os.path.basename(path), loadThreeMF(path)))

    failures = sum(checkShape(name, vertices, faces) for name, (vertices, faces) in shapes)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
