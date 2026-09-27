"""Particiones del dataset y carga de imágenes con tf.data."""
from __future__ import annotations

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.dataset import CLASSES

SEED = 42


def make_splits(df: pd.DataFrame, val_size: float = 0.15, seed: int = SEED):
    """Divide el índice (con columna `md5`) en train / val / test.

    La partición oficial `validation` del NEU se reserva como test y solo se usa en la
    evaluación final. De `train` se separa un subconjunto estratificado para validar
    durante el entrenamiento. Se eliminan duplicados de train y cualquier imagen de train
    que también esté en test.
    """
    test = df[df["split"] == "validation"]
    train_full = df[df["split"] == "train"].drop_duplicates("md5")
    train_full = train_full[~train_full["md5"].isin(test["md5"])]
    train, val = train_test_split(
        train_full, test_size=val_size, stratify=train_full["label"], random_state=seed
    )
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def make_dataset(frame: pd.DataFrame, image_size: int = 200, channels: int = 1,
                 batch_size: int = 32, shuffle: bool = False, seed: int = SEED) -> tf.data.Dataset:
    """tf.data.Dataset de (imagen float32 en [0, 255], índice de clase) a partir del índice."""
    labels = frame["label"].map(CLASSES.index).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((frame["path"].to_numpy(), labels))
    if shuffle:
        ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=channels,
                                 expand_animations=False)
        img = tf.image.resize(img, (image_size, image_size))
        return img, label

    return (ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))
