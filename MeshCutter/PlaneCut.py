# Copyright (c) 2026 shelojara
# Released under the terms of the LGPLv3 or higher.
#
# Pure-geometry plane cutter. Deliberately free of any Uranium/Cura imports so
# it can be unit tested with a plain numpy install.

from typing import Dict, List, Optional, Tuple

import numpy

#  Outlines enclosing less than this many square millimetres are treated as
#  collapsed: filled in, but not used to work out what is a hole in what.
_COLLAPSED_AREA = 1e-6


class Loop:
    """A closed polyline of 2D points, plus the 3D points they came from."""

    def __init__(self, points_2d: numpy.ndarray, points_3d: numpy.ndarray) -> None:
        self.points_2d = points_2d
        self.points_3d = points_3d
        self.area = _signedArea(points_2d)
        self.holes: List["Loop"] = []


def cutMesh(vertices: numpy.ndarray, indices: Optional[numpy.ndarray],
            plane_normal: numpy.ndarray, plane_offset: float,
            weld_tolerance: float = 1e-7) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Cut a triangle soup in two with an infinite plane, capping both halves.

    :param vertices: (n, 3) float array of vertex positions.
    :param indices: (m, 3) int array of triangle indices, or None when the
        vertex array is a flat triangle soup (every 3 vertices is a triangle).
    :param plane_normal: (3,) unit normal of the cut plane.
    :param plane_offset: signed distance of the plane from the origin along the
        normal, so the plane is {p : dot(p, normal) == offset}.
    :param weld_tolerance: distance under which two cut points are considered
        the same point when stitching the cap outline together.
    :return: two (k, 3) float arrays of flat, non-indexed triangles. The first
        is the half on the negative side of the normal, the second the half on
        the positive side.
    """
    vertices = numpy.asarray(vertices, dtype=numpy.float64)
    if indices is None:
        faces = numpy.arange(len(vertices) - len(vertices) % 3).reshape(-1, 3)
    else:
        faces = numpy.asarray(indices, dtype=numpy.int64).reshape(-1, 3)

    plane_normal = numpy.asarray(plane_normal, dtype=numpy.float64)
    plane_normal = plane_normal / numpy.linalg.norm(plane_normal)

    distances = vertices.dot(plane_normal) - plane_offset
    # An epsilon relative to the model size, so that vertices sitting a rounding
    # error off the plane are treated as being on it rather than straddling it.
    epsilon = max(1e-9, 1e-7 * float(numpy.abs(vertices).max() if len(vertices) else 1.0))

    below: List[numpy.ndarray] = []
    above: List[numpy.ndarray] = []

    face_distances = distances[faces]
    face_side = numpy.zeros(face_distances.shape, dtype=numpy.int8)
    face_side[face_distances > epsilon] = 1
    face_side[face_distances < -epsilon] = -1
    side_sum = face_side.sum(axis=1)
    has_positive = (face_side > 0).any(axis=1)
    has_negative = (face_side < 0).any(axis=1)

    # Triangles fully on one side (on-plane vertices don't count) pass straight
    # through, which keeps the vast majority of the mesh bit-identical.
    keep_above = has_positive & ~has_negative
    keep_below = has_negative & ~has_positive
    above.append(vertices[faces[keep_above]].reshape(-1, 3))
    below.append(vertices[faces[keep_below]].reshape(-1, 3))

    # Triangles lying in the plane belong to whichever half they close off.
    coplanar = (side_sum == 0) & ~has_positive & ~has_negative
    for face in faces[coplanar]:
        triangle = vertices[face]
        normal = numpy.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        (above if normal.dot(plane_normal) < 0 else below).append(triangle)

    straddling = has_positive & has_negative
    for face in faces[straddling]:
        triangle = vertices[face]
        sides = numpy.array([_side(distances[i], epsilon) for i in face], dtype=numpy.int8)
        signed = distances[face]
        above.extend(_fanTriangulate(_clipToHalfSpace(triangle, signed, sides, 1)))
        below.extend(_fanTriangulate(_clipToHalfSpace(triangle, signed, sides, -1)))

    above_vertices = _stack(above)
    below_vertices = _stack(below)

    # The outline to cap is the open boundary the cut left behind: every edge of
    # the half that has no neighbour on the other side of it. Deriving it from
    # the sliced geometry rather than from the cut itself is what keeps the
    # result watertight even where the original mesh already had edges lying
    # exactly in the plane.
    cap_below, cap_above = _buildCaps(above_vertices, plane_normal, plane_offset,
                                      weld_tolerance, epsilon)
    above_vertices = _stack([above_vertices, cap_above])
    below_vertices = _stack([below_vertices, cap_below])
    return below_vertices, above_vertices


def _stack(parts: List[numpy.ndarray]) -> numpy.ndarray:
    parts = [numpy.asarray(part).reshape(-1, 3) for part in parts]
    parts = [part for part in parts if len(part)]
    if not parts:
        return numpy.zeros((0, 3))
    return numpy.concatenate(parts)


def _boundaryEdges(triangle_soup: numpy.ndarray, plane_normal: numpy.ndarray,
                   plane_offset: float, weld_tolerance: float,
                   epsilon: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Directed edges of a triangle soup that no opposite-facing edge matches.

    Only edges with both ends in the cut plane are returned, so pre-existing
    holes elsewhere in the model are left exactly as they were.
    :return: an (e, 2) array of point indices and the (p, 3) welded points.
    """
    if not len(triangle_soup):
        return numpy.zeros((0, 2), dtype=numpy.int64), numpy.zeros((0, 3))
    quantised = numpy.round(triangle_soup / weld_tolerance).astype(numpy.int64)
    _, first, inverse = numpy.unique(quantised, axis=0, return_index=True, return_inverse=True)
    points = triangle_soup[first]
    corners = inverse.reshape(-1, 3)

    edges = numpy.concatenate((corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]))
    on_plane = numpy.abs(points.dot(plane_normal) - plane_offset) <= max(epsilon * 100, weld_tolerance)
    edges = edges[on_plane[edges].all(axis=1) & (edges[:, 0] != edges[:, 1])]
    if not len(edges):
        return numpy.zeros((0, 2), dtype=numpy.int64), points

    # An edge is on the boundary when it outnumbers its own reverse.
    stride = len(points)
    forward = edges[:, 0] * stride + edges[:, 1]
    backward = edges[:, 1] * stride + edges[:, 0]
    unique_keys, counts = numpy.unique(forward, return_counts=True)
    tally = dict(zip(unique_keys.tolist(), counts.tolist()))
    keep = numpy.array([tally.get(int(f), 0) > tally.get(int(b), 0)
                        for f, b in zip(forward, backward)], dtype=bool)
    return edges[keep], points


def _side(distance: float, epsilon: float) -> int:
    if distance > epsilon:
        return 1
    if distance < -epsilon:
        return -1
    return 0


def _clipToHalfSpace(triangle: numpy.ndarray, distances: numpy.ndarray,
                     sides: numpy.ndarray, keep: int) -> numpy.ndarray:
    """Sutherland-Hodgman clip of one triangle against a half space.

    Winding is preserved, and vertices that sit on the plane are emitted as-is
    instead of being re-derived, so both halves share the exact same points.
    """
    output: List[numpy.ndarray] = []
    for i in range(3):
        j = (i + 1) % 3
        side_i, side_j = int(sides[i]), int(sides[j])
        if side_i == keep or side_i == 0:
            output.append(triangle[i])
        if side_i != 0 and side_j != 0 and side_i != side_j:
            t = distances[i] / (distances[i] - distances[j])
            output.append(triangle[i] + t * (triangle[j] - triangle[i]))
    return numpy.array(output) if output else numpy.zeros((0, 3))


def _fanTriangulate(polygon: numpy.ndarray) -> List[numpy.ndarray]:
    """Fan a small convex polygon (3 or 4 points here) into triangles."""
    triangles = []
    for i in range(1, len(polygon) - 1):
        triangles.append(numpy.array([polygon[0], polygon[i], polygon[i + 1]]))
    return triangles


def _buildCaps(above_soup: numpy.ndarray, plane_normal: numpy.ndarray,
               plane_offset: float, weld_tolerance: float,
               epsilon: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Triangulate the cut outline and return the cap for each half.

    Both halves get the same triangles, one of them wound the other way, so the
    two parts mate back together exactly.
    """
    edges, points = _boundaryEdges(above_soup, plane_normal, plane_offset,
                                   weld_tolerance, epsilon)
    if not len(edges):
        return numpy.zeros((0, 3)), numpy.zeros((0, 3))

    basis_u, basis_v = _planeBasis(plane_normal)
    points_2d = numpy.column_stack((points.dot(basis_u), points.dot(basis_v)))

    loops: List[Loop] = []
    collapsed: List[Loop] = []
    for chain in _chainLoops(edges, points_2d):
        if len(chain) < 3:
            continue
        loop = Loop(points_2d[chain], points[chain])
        # Where the plane grazes a curved face the outline encloses next to no
        # area. It still has to be filled, or the seam is left open, but it has
        # no meaningful orientation so it is triangulated on its own.
        (loops if abs(loop.area) > _COLLAPSED_AREA else collapsed).append(loop)
    if not loops and not collapsed:
        return numpy.zeros((0, 3)), numpy.zeros((0, 3))

    # Outlines come out of the slice already oriented: outer boundaries one way,
    # holes the other. Assign every hole to its innermost containing boundary.
    outers = [loop for loop in loops if loop.area > 0]
    holes = [loop for loop in loops if loop.area < 0]
    if loops and not outers:  # Orientation flipped from what we expected; swap the roles.
        outers, holes = holes, outers
        for loop in outers + holes:
            loop.points_2d = loop.points_2d[::-1]
            loop.points_3d = loop.points_3d[::-1]
            loop.area = -loop.area
    for hole in holes:
        parent = None
        for outer in outers:
            if _pointInPolygon(hole.points_2d[0], outer.points_2d):
                if parent is None or abs(outer.area) < abs(parent.area):
                    parent = outer
        if parent is not None:
            parent.holes.append(hole)

    cap: List[numpy.ndarray] = []
    for outer in outers:
        cap.extend(_triangulateWithHoles(outer))
    for loop in collapsed:
        cap.extend(_earClip(loop.points_2d, loop.points_3d))
    if not cap:
        return numpy.zeros((0, 3)), numpy.zeros((0, 3))

    cap_array = numpy.concatenate(cap).reshape(-1, 3)
    # As collected, the cap winds the same way as the boundary of the positive
    # half, which makes it the outward-facing cap of the negative half.
    below = cap_array
    above = cap_array.reshape(-1, 3, 3)[:, ::-1, :].reshape(-1, 3)
    return below, above


def _planeBasis(plane_normal: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    helper = numpy.array([1.0, 0.0, 0.0])
    if abs(plane_normal.dot(helper)) > 0.9:
        helper = numpy.array([0.0, 1.0, 0.0])
    basis_u = numpy.cross(helper, plane_normal)
    basis_u /= numpy.linalg.norm(basis_u)
    basis_v = numpy.cross(plane_normal, basis_u)
    return basis_u, basis_v


def _chainLoops(edges: numpy.ndarray, points_2d: numpy.ndarray) -> List[List[int]]:
    """Walk directed boundary edges into closed loops.

    Where several boundary edges meet in one point - two parts of the model
    touching at a corner, for instance - the walk takes the tightest turn
    available, which keeps the loops simple instead of tying them into a knot.
    """
    outgoing: Dict[int, List[int]] = {}
    for index, (start, _) in enumerate(edges):
        outgoing.setdefault(int(start), []).append(index)
    used = numpy.zeros(len(edges), dtype=bool)

    loops: List[List[int]] = []
    for seed in range(len(edges)):
        if used[seed]:
            continue
        used[seed] = True
        first = int(edges[seed][0])
        chain = [first]
        current_edge = seed
        while True:
            current = int(edges[current_edge][1])
            if current == first:
                break
            chain.append(current)
            candidates = [i for i in outgoing.get(current, []) if not used[i]]
            if not candidates:
                chain = []  # Dead end: shouldn't happen, but never emit a partial loop.
                break
            if len(candidates) == 1:
                current_edge = candidates[0]
            else:
                current_edge = _tightestTurn(edges, points_2d, current_edge, candidates)
            used[current_edge] = True
        if len(chain) >= 3:
            loops.append(chain)
    return loops


def _tightestTurn(edges: numpy.ndarray, points_2d: numpy.ndarray,
                  incoming: int, candidates: List[int]) -> int:
    """Pick the candidate edge that turns most sharply clockwise."""
    pivot = points_2d[int(edges[incoming][1])]
    back = points_2d[int(edges[incoming][0])] - pivot
    reference = numpy.arctan2(back[1], back[0])
    best, best_turn = candidates[0], -1.0
    for candidate in candidates:
        forward = points_2d[int(edges[candidate][1])] - pivot
        turn = (reference - numpy.arctan2(forward[1], forward[0])) % (2 * numpy.pi)
        if turn > best_turn:
            best_turn, best = turn, candidate
    return best


def _signedArea(points_2d: numpy.ndarray) -> float:
    x, y = points_2d[:, 0], points_2d[:, 1]
    return 0.5 * float(numpy.dot(x, numpy.roll(y, -1)) - numpy.dot(y, numpy.roll(x, -1)))


def _pointInPolygon(point: numpy.ndarray, polygon: numpy.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    count = len(polygon)
    for i in range(count):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % count]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) / (y1 - y0) * (x1 - x0):
            inside = not inside
    return inside


def _triangulateWithHoles(outer: Loop) -> List[numpy.ndarray]:
    """Bridge the holes into the outer ring, then ear clip the result."""
    ring_2d = list(outer.points_2d)
    ring_3d = list(outer.points_3d)
    for hole in sorted(outer.holes, key=lambda h: -float(h.points_2d[:, 0].max())):
        ring_2d, ring_3d = _bridgeHole(ring_2d, ring_3d, list(hole.points_2d), list(hole.points_3d))
    return _earClip(numpy.array(ring_2d), numpy.array(ring_3d))


def _bridgeHole(ring_2d, ring_3d, hole_2d, hole_3d):
    """Splice a hole into the outer ring with a zero-width bridge.

    The hole's rightmost vertex is joined to a ring vertex it can actually see,
    which turns ring-plus-hole into a single simple polygon that ear clipping
    handles without any special casing.
    """
    hole_start = int(numpy.argmax([point[0] for point in hole_2d]))
    rotated_2d = hole_2d[hole_start:] + hole_2d[:hole_start]
    rotated_3d = hole_3d[hole_start:] + hole_3d[:hole_start]
    bridge = _findBridgeVertex(numpy.array(ring_2d), numpy.array(rotated_2d))
    merged_2d = ring_2d[:bridge + 1] + rotated_2d + [rotated_2d[0]] + ring_2d[bridge:]
    merged_3d = ring_3d[:bridge + 1] + rotated_3d + [rotated_3d[0]] + ring_3d[bridge:]
    return merged_2d, merged_3d


def _findBridgeVertex(ring: numpy.ndarray, hole: numpy.ndarray) -> int:
    """Index of the nearest ring vertex the hole's first vertex can see.

    A bridge that crosses no edge of either outline is guaranteed to stay inside
    the region being filled, which is a cheaper thing to get right than the
    usual angle-and-sector heuristics.
    """
    start = hole[0]
    for candidate in numpy.argsort(numpy.linalg.norm(ring - start, axis = 1)):
        if not _crossesOutline(start, ring[candidate], ring, int(candidate)) \
                and not _crossesOutline(start, ring[candidate], hole, 0):
            return int(candidate)
    return int(numpy.argmin(numpy.linalg.norm(ring - start, axis = 1)))


def _crossesOutline(start: numpy.ndarray, end: numpy.ndarray,
                    outline: numpy.ndarray, touching: int) -> bool:
    """Does the segment properly cross any edge of a closed outline?

    Edges meeting the segment at the vertex it is allowed to touch don't count,
    and neither does a mere graze: only a genuine crossing disqualifies it.
    """
    count = len(outline)
    first = outline
    second = numpy.roll(outline, -1, axis = 0)
    keep = numpy.ones(count, dtype = bool)
    keep[[touching, (touching - 1) % count]] = False
    first, second = first[keep], second[keep]
    if not len(first):
        return False

    def cross(origin, to, points):
        return ((to[0] - origin[0]) * (points[:, 1] - origin[1]) -
                (to[1] - origin[1]) * (points[:, 0] - origin[0]))

    side_first = cross(start, end, first)
    side_second = cross(start, end, second)
    straddles = (side_first > 0) != (side_second > 0)
    both = numpy.column_stack((start, end)).T
    side_start = ((second[:, 0] - first[:, 0]) * (both[0, 1] - first[:, 1]) -
                  (second[:, 1] - first[:, 1]) * (both[0, 0] - first[:, 0]))
    side_end = ((second[:, 0] - first[:, 0]) * (both[1, 1] - first[:, 1]) -
                (second[:, 1] - first[:, 1]) * (both[1, 0] - first[:, 0]))
    return bool((straddles & ((side_start > 0) != (side_end > 0))).any())


def _earClip(ring_2d: numpy.ndarray, ring_3d: numpy.ndarray) -> List[numpy.ndarray]:
    """Ear clipping for a simple, counter-clockwise polygon."""
    triangles: List[numpy.ndarray] = []
    pending = [list(range(len(ring_2d)))]
    while pending:
        ring = pending.pop()
        triangles.extend(_clipRing(ring_2d, ring_3d, ring, pending))
    return triangles


def _clipRing(ring_2d: numpy.ndarray, ring_3d: numpy.ndarray,
              remaining: List[int], pending: List[List[int]]) -> List[numpy.ndarray]:
    """Clip one ring, handing back any piece it had to split off.

    CAD meshes hand us long straight edges chopped into many collinear
    vertices, and after hole bridging the ring also contains doubled points.
    Those corners are clipped as zero-area ears rather than skipped: the cap has
    to keep every vertex the walls reference, or the seam stops being watertight.
    """
    triangles: List[numpy.ndarray] = []

    def emit(position: int) -> None:
        count = len(remaining)
        previous = remaining[(position - 1) % count]
        current = remaining[position]
        following = remaining[(position + 1) % count]
        triangles.append(numpy.array([ring_3d[previous], ring_3d[current], ring_3d[following]]))
        remaining.pop(position)

    while len(remaining) > 3:
        points = ring_2d[remaining]
        before = numpy.roll(points, 1, axis = 0)
        after = numpy.roll(points, -1, axis = 0)
        cross = ((points[:, 0] - before[:, 0]) * (after[:, 1] - before[:, 1]) -
                 (points[:, 1] - before[:, 1]) * (after[:, 0] - before[:, 0]))

        # A corner counts as straight only when the vertex is within a
        # nanometre of the line through its neighbours. Judging that against the
        # size of the whole outline instead would let a run of clips flatten a
        # fillet into a chord that cuts clean across the shape.
        chord = numpy.linalg.norm(after - before, axis = 1)
        flat = numpy.abs(cross) <= 1e-9 * numpy.maximum(chord, 1e-12)

        # Straight-through and doubled vertices: always safe to clip, and doing
        # it first stops them from blocking every real ear around them.
        clipped = False
        for position in numpy.flatnonzero(flat):
            arm_in = before[position] - points[position]
            arm_out = after[position] - points[position]
            if arm_in.dot(arm_out) < 0 or numpy.allclose(arm_in, 0.0) or numpy.allclose(arm_out, 0.0):
                emit(int(position))
                clipped = True
                break
        if clipped:
            continue

        for position in numpy.argsort(-cross):
            if cross[position] <= 0.0:
                break
            if _blocked(points, int(position)):
                continue
            emit(int(position))
            clipped = True
            break
        if clipped:
            continue

        # Thin-walled outlines can reach a state where every ear reaches across
        # the wall and swallows a vertex from the far side. Cutting the ring in
        # two along a diagonal that stays inside it gets things moving again.
        split = _findSplit(points)
        if split is None:
            # Give up gracefully: take the most convex corner so most of the
            # area still gets filled rather than losing the outline entirely.
            emit(int(numpy.argmax(cross)))
            continue
        first, second = split
        pending.append(remaining[first:second + 1])
        pending.append(remaining[second:] + remaining[:first + 1])
        return triangles

    if len(remaining) == 3:
        triangles.append(numpy.array([ring_3d[remaining[0]], ring_3d[remaining[1]], ring_3d[remaining[2]]]))
    return triangles


def _findSplit(points: numpy.ndarray) -> Optional[Tuple[int, int]]:
    """Find two vertices that can be joined by a diagonal inside the ring.

    Short diagonals are tried first: they slice off a small piece, and they are
    far less likely to run along a wall where the geometry is ambiguous.
    """
    count = len(points)
    for gap in range(2, count - 1):
        for first in range(count - gap):
            second = first + gap
            if first == 0 and second == count - 1:
                continue
            if _isDiagonal(points, first, second):
                return first, second
    return None


def _isDiagonal(points: numpy.ndarray, first: int, second: int) -> bool:
    """Does the segment between two ring vertices stay inside the ring?"""
    start, end = points[first], points[second]
    if numpy.allclose(start, end):
        return False
    if not _pointsInwards(points, first, end) or not _pointsInwards(points, second, start):
        return False

    count = len(points)
    edge_start = points
    edge_end = numpy.roll(points, -1, axis = 0)
    keep = numpy.ones(count, dtype = bool)
    keep[[first, (first - 1) % count, second, (second - 1) % count]] = False
    edge_start, edge_end = edge_start[keep], edge_end[keep]

    if len(edge_start):
        side_start = ((end[0] - start[0]) * (edge_start[:, 1] - start[1]) -
                      (end[1] - start[1]) * (edge_start[:, 0] - start[0]))
        side_end = ((end[0] - start[0]) * (edge_end[:, 1] - start[1]) -
                    (end[1] - start[1]) * (edge_end[:, 0] - start[0]))
        along_start = ((edge_end[:, 0] - edge_start[:, 0]) * (start[1] - edge_start[:, 1]) -
                       (edge_end[:, 1] - edge_start[:, 1]) * (start[0] - edge_start[:, 0]))
        along_end = ((edge_end[:, 0] - edge_start[:, 0]) * (end[1] - edge_start[:, 1]) -
                     (edge_end[:, 1] - edge_start[:, 1]) * (end[0] - edge_start[:, 0]))
        if (((side_start > 0) != (side_end > 0)) & ((along_start > 0) != (along_end > 0))).any():
            return False

    return _pointInPolygon((start + end) / 2.0, points)


def _pointsInwards(points: numpy.ndarray, index: int, target: numpy.ndarray) -> bool:
    """Does a segment leaving one ring vertex head into the ring, not out of it?

    Without this a long diagonal that slips out through a reflex corner and back
    in somewhere else looks perfectly valid.
    """
    count = len(points)
    here = points[index]
    previous = points[(index - 1) % count]
    following = points[(index + 1) % count]

    def turn(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1]) -
                (first[1] - origin[1]) * (second[0] - origin[0]))

    if turn(previous, here, following) > 0:  # Convex corner: inside is the wedge.
        return turn(here, following, target) > 0 and turn(here, target, previous) > 0
    return turn(here, following, target) > 0 or turn(here, target, previous) > 0


def _blocked(points: numpy.ndarray, position: int) -> bool:
    """Does any other ring vertex sit in the candidate ear?"""
    count = len(points)
    a, b, c = points[position - 1], points[position], points[(position + 1) % count]
    mask = numpy.ones(count, dtype=bool)
    mask[[(position - 1) % count, position, (position + 1) % count]] = False
    others = points[mask]
    if not len(others):
        return False
    side_ab = (b[0] - a[0]) * (others[:, 1] - a[1]) - (b[1] - a[1]) * (others[:, 0] - a[0])
    side_bc = (c[0] - b[0]) * (others[:, 1] - b[1]) - (c[1] - b[1]) * (others[:, 0] - b[0])
    side_ca = (a[0] - c[0]) * (others[:, 1] - c[1]) - (a[1] - c[1]) * (others[:, 0] - c[0])
    return bool(((side_ab >= 0) & (side_bc >= 0) & (side_ca >= 0)).any())
