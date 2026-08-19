# Mesh Cutter — a Cura extension

Cuts the selected model in half through the middle of its bounding box and
leaves **two independent, closed solids** behind. Each half has a real cap over
the cut face, so laying a half flat on that face keeps the split instead of
undoing it — which is what makes it usable for a model that is too long to print
in one piece and can't be printed on its side.

## Install

Copy the `MeshCutter` folder into Cura's plugin directory and restart Cura:

```
~/Library/Application Support/cura/<version>/plugins/MeshCutter/MeshCutter/
```

(The doubled folder name is the layout Cura's own package installer uses, and is
where the already-installed plugins on this machine live.)

## Use

`Extensions → Cut in Half`, with a model selected:

| Menu item | Cut plane |
| --- | --- |
| Cut along the longest side | across whichever bounding-box side is longest |
| Cut left / right (X) | vertical plane across the width |
| Cut top / bottom (Y) | horizontal plane across the height |
| Cut front / back (Z) | vertical plane across the depth |

Both halves land on the plate 8 mm apart, sideways from the cut so they don't
recreate the length that didn't fit. They inherit the original's per-object
settings, extruder and build plate, and they arrive selected. One Ctrl+Z undoes
the whole thing.

## How the cut works

`MeshCutter/PlaneCut.py` holds the geometry and imports nothing but numpy, so it
runs outside Cura too.

1. Every triangle is clipped against the plane (Sutherland–Hodgman), preserving
   winding and reusing vertices that already sit on the plane, so the two halves
   share the cut points exactly.
2. The outline to cap is taken from the *boundary edges of the sliced half* —
   edges with no opposite-facing partner — not from the cut itself. That is what
   handles a model whose triangles already have edges lying in the cut plane.
3. Boundary edges are chained into loops, holes are matched to the outline that
   contains them and bridged in, and the result is ear clipped. Collinear and
   doubled vertices are clipped as zero-area ears rather than dropped, because
   the cap has to keep every vertex the walls reference.
4. Both halves get the same cap triangles, one set wound the other way, so the
   parts mate back together exactly.

Known limitation: where the plane is very nearly tangent to a curved face, the
cross-section collapses to a sliver. Those outlines are still filled, but the
seam there is only as good as the model's own tessellation.

## Tests

```
python3 tests/test_plane_cut.py                     # built-in shapes
python3 tests/test_plane_cut.py path/to/model.3mf   # ...and your own model
```

No model file is needed: the suite builds its own. A **cube** covers the plain
case, and a **square tube** covers the hard one — cutting across it leaves an
outline with a hole in it and walls a couple of triangles apart, which is what
breaks naive ear clipping. Both are checked against their analytic volume first,
so a broken test shape can't be mistaken for a broken cutter.

Each shape is then cut 60 ways — the three axes at random heights plus oblique
planes — and every half is checked to be watertight (each edge walked exactly
once in each direction, so a face that is turned the wrong way counts as a
failure too) and for the two volumes to add back up to the original. Any
watertight 3mf passed on the command line gets the same treatment.
