"""Stage 2 furniture reconstruction: model-library matching, B15 structural recovery, and zero-penetration placement."""

from ipcr.furniture.chairs import build_chair_parts
from ipcr.furniture.library import CanonicalLibraryAsset, FurnitureLibrary
from ipcr.furniture.tables_cabinets_sofas import (
    TableSpec,
    build_cabinet_parts,
    build_sofa_parts,
    build_table_parts,
)

__all__ = [
    "CanonicalLibraryAsset",
    "FurnitureLibrary",
    "TableSpec",
    "build_table_parts",
    "build_cabinet_parts",
    "build_sofa_parts",
    "build_chair_parts",
]
