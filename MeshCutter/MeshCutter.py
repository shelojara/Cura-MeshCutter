# Copyright (c) 2026 shelojara
# Released under the terms of the LGPLv3 or higher.

import copy
from typing import List, Optional, Tuple

import numpy

from UM.Application import Application
from UM.Logger import Logger
from UM.Extension import Extension
from UM.Math.Matrix import Matrix
from UM.Math.Vector import Vector
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Message import Message
from UM.Operations.AddSceneNodeOperation import AddSceneNodeOperation
from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.RemoveSceneNodeOperation import RemoveSceneNodeOperation
from UM.Operations.TranslateOperation import TranslateOperation
from UM.Scene.SceneNode import SceneNode
from UM.Scene.Selection import Selection
from UM.i18n import i18nCatalog

from .PlaneCut import cutMesh

catalog = i18nCatalog("cura")

#  Gap left between the two halves on the build plate, in millimetres. Matches
#  the offset Cura itself uses when it multiplies an object.
SEPARATION = 8.0


class MeshCutter(Extension):
    """Cuts the selected model in half and leaves two independent objects.

    Both halves are closed solids in their own right, so rotating one onto its
    cut face - the whole point of splitting an over-long model - keeps the split
    instead of undoing it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMenuName(catalog.i18nc("@item:inmenu", "Cut in Half"))
        self.addMenuItem(catalog.i18nc("@item:inmenu", "Cut along the longest side"),
                         lambda: self.cutSelection(None))
        self.addMenuItem(catalog.i18nc("@item:inmenu", "Cut left / right (X)"),
                         lambda: self.cutSelection(0))
        self.addMenuItem(catalog.i18nc("@item:inmenu", "Cut top / bottom (Y)"),
                         lambda: self.cutSelection(1))
        self.addMenuItem(catalog.i18nc("@item:inmenu", "Cut front / back (Z)"),
                         lambda: self.cutSelection(2))

    def cutSelection(self, axis: Optional[int]) -> None:
        """Cut every selected model through the middle of its bounding box.

        :param axis: 0, 1 or 2 for a cut across X, Y or Z, or None to pick
            whichever side of the model is longest.
        """
        nodes = [node for node in Selection.getAllSelectedObjects() if node.getMeshData() is not None]
        if not nodes:
            self._warn(catalog.i18nc("@info:status", "Select a model to cut in half first."))
            return
        if any(node.callDecoration("isGroup") for node in Selection.getAllSelectedObjects()):
            self._warn(catalog.i18nc("@info:status",
                                     "Groups can't be cut. Ungroup the models and cut them one by one."))

        operation = GroupedOperation()
        parts: List[SceneNode] = []
        for node in nodes:
            result = self._cutNode(node, axis)
            if result is None:
                continue
            halves, cut_axis = result
            self._addCutOperations(operation, node, halves, cut_axis)
            parts.extend(halves)

        if not parts:
            return

        Selection.clear()
        operation.push()
        for part in parts:
            Selection.add(part)
        scene = Application.getInstance().getController().getScene()
        scene.sceneChanged.emit(scene.getRoot())

    def _cutNode(self, node: SceneNode,
                 axis: Optional[int]) -> Optional[Tuple[Tuple[SceneNode, SceneNode], int]]:
        """Build the two halves of one node, and report which axis was cut."""
        mesh = node.getMeshData()
        bounding_box = node.getBoundingBox()
        if mesh is None or mesh.getVertexCount() == 0 or bounding_box is None:
            return None

        cut_axis = axis if axis is not None else self._longestAxis(bounding_box)
        normal = numpy.zeros(3)
        normal[cut_axis] = 1.0
        offset = float(bounding_box.center.getData()[cut_axis])

        # Cut in world space, where the bounding box the user sees lives, then
        # bring the result back into the node's own frame so both halves keep
        # the original object's transformation.
        world = node.getWorldTransformation(copy = False)
        world_mesh = mesh.getTransformed(world)
        try:
            indices = world_mesh.getIndices() if world_mesh.hasIndices() else None
            lower, upper = cutMesh(world_mesh.getVertices(), indices, normal, offset)
        except Exception:
            Logger.logException("e", "Cutting %s in half failed", node.getName())
            self._warn(catalog.i18nc("@info:status", "Could not cut {0}.").format(node.getName()))
            return None

        if not len(lower) or not len(upper):
            self._warn(catalog.i18nc("@info:status", "{0} has nothing to cut along that axis.")
                       .format(node.getName()))
            return None

        inverse = world.getInverse()
        return (self._buildHalf(node, lower, inverse, 1),
                self._buildHalf(node, upper, inverse, 2)), cut_axis

    def _buildHalf(self, node: SceneNode, triangles: numpy.ndarray,
                   inverse: Matrix, part_number: int) -> SceneNode:
        """Copy the original node and give the copy one half of the geometry.

        Copying means the halves inherit the per-object settings, extruder and
        build plate of the model they came from.
        """
        half = copy.deepcopy(node)
        half.setName("{0} - part {1}".format(node.getName(), part_number))

        builder = MeshBuilder()
        builder.setVertices(_transformPoints(inverse, triangles).astype(numpy.float32))
        builder.calculateNormals(fast = True)
        builder.setFileName(node.getMeshData().getFileName())
        half.setMeshData(builder.build())
        half.calculateBoundingBoxMesh()
        half.setSelectable(True)

        build_plate = node.callDecoration("getBuildPlateNumber")
        half.callDecoration("setBuildPlateNumber", build_plate)
        for child in half.getChildren():
            child.callDecoration("setBuildPlateNumber", build_plate)
        return half

    def _addCutOperations(self, operation: GroupedOperation, node: SceneNode,
                          halves: Tuple[SceneNode, SceneNode], cut_axis: int) -> None:
        """Swap the original for its two halves, moved apart on the plate."""
        root = Application.getInstance().getController().getScene().getRoot()
        operation.addOperation(RemoveSceneNodeOperation(node))
        for half in halves:
            operation.addOperation(AddSceneNodeOperation(half, root))

        shift = self._separation(halves[0], cut_axis)
        operation.addOperation(TranslateOperation(halves[0], -shift))
        operation.addOperation(TranslateOperation(halves[1], shift))

    def _separation(self, half: SceneNode, cut_axis: int) -> Vector:
        """Half the distance to move each part, sideways across the plate.

        Moving them apart along the cut itself would just re-create the length
        that didn't fit in the first place, so the parts step sideways instead.
        """
        sideways = 2 if cut_axis == 0 else 0  # Across the plate, never upwards.
        box = half.getBoundingBox()
        extent = (box.width, box.height, box.depth)[sideways] if box is not None else 0.0
        distance = (abs(float(extent)) + SEPARATION) / 2.0
        return Vector(*(distance if index == sideways else 0.0 for index in range(3)))

    @staticmethod
    def _longestAxis(bounding_box) -> int:
        sizes = [bounding_box.width, bounding_box.height, bounding_box.depth]
        return int(numpy.argmax(sizes))

    @staticmethod
    def _warn(text: str) -> None:
        Message(text, title = catalog.i18nc("@info:title", "Cut in Half"), lifetime = 15).show()



def _transformPoints(matrix: Matrix, points: numpy.ndarray) -> numpy.ndarray:
    data = matrix.getData()
    return points.dot(data[:3, :3].T) + data[:3, 3]
