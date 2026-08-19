# Copyright (c) 2026 Marcelo
# Released under the terms of the LGPLv3 or higher.

from . import MeshCutter

from UM.i18n import i18nCatalog
catalog = i18nCatalog("cura")


def getMetaData():
    return {}


def register(app):
    return {"extension": MeshCutter.MeshCutter()}
