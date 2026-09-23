"""Indexado y control de calidad del dataset NEU Surface Defect Database."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

# Prefijos de archivo usados en la variante NEU-CLS (p. ej. "Cr_12.bmp")
_PREFIXES = {
    "cr": "crazing",
    "in": "inclusion",
    "pa": "patches",
    "ps": "pitted_surface",
    "rs": "rolled_in_scale",
    "sc": "scratches",
}

_SPLITS = {"train": "train", "training": "train", "validation": "validation",
           "val": "validation", "valid": "validation", "test": "test"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png"}


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def infer_label(path: Path) -> str | None:
    """Clase según la carpeta contenedora o, si no, según el nombre del archivo."""
    parent = _normalize(path.parent.name)
    if parent in CLASSES:
        return parent
    stem = _normalize(path.stem)
    for cls in CLASSES:
        if stem.startswith(cls):
            return cls
    return _PREFIXES.get(stem.split("_")[0])


def infer_split(path: Path) -> str:
    """Partición (train/validation/test) según las carpetas de la ruta."""
    for part in reversed(path.parts):
        split = _SPLITS.get(part.lower())
        if split:
            return split
    return "sin_particion"


def build_index(root: str | Path) -> pd.DataFrame:
    """Recorre `root` y devuelve una fila por imagen con su ruta, clase y partición."""
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label(path)
        if label is None:
            continue
        rows.append({
            "path": str(path),
            "filename": path.name,
            "label": label,
            "split": infer_split(path.relative_to(root)),
        })
    if not rows:
        raise FileNotFoundError(f"No se encontraron imágenes del NEU en {root}")
    return pd.DataFrame(rows)


def image_metadata(path: str | Path) -> dict:
    """Tamaño, modo de color, hash MD5 y si la imagen se puede decodificar."""
    data = Path(path).read_bytes()
    info = {"md5": hashlib.md5(data).hexdigest(), "file_kb": len(data) / 1024}
    try:
        with Image.open(path) as img:
            img.load()  # fuerza la decodificación completa para detectar archivos corruptos
            info.update(width=img.width, height=img.height, mode=img.mode, corrupt=False)
    except OSError:
        info.update(width=None, height=None, mode=None, corrupt=True)
    return info


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega a `df` las columnas de `image_metadata` para cada imagen."""
    meta = pd.DataFrame([image_metadata(p) for p in df["path"]], index=df.index)
    return pd.concat([df, meta], axis=1)
