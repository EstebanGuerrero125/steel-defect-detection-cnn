"""Arquitecturas de los modelos de clasificación de defectos."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.dataset import CLASSES


def augmentation() -> keras.Sequential:
    """Aumento de datos para imágenes en [0, 255].

    Los defectos no tienen una orientación fija, así que se voltean en ambos ejes. Los
    cambios de brillo y contraste evitan que el modelo aprenda a distinguir clases por la
    iluminación, que en el NEU varía mucho entre clases.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.15),
    ], name="augmentacion")


def build_baseline_cnn(input_shape=(200, 200, 1), num_classes: int = len(CLASSES),
                       dropout: float = 0.3) -> keras.Model:
    """CNN pequeña entrenada desde cero: 4 bloques Conv-BN-ReLU-MaxPool y una capa densa.

    BatchNormalization usa momentum 0.9 (Keras usa 0.99 por defecto): con ~38 lotes por
    época, las estadísticas móviles que se usan en inferencia no alcanzan a converger con
    0.99 y el modelo predice una sola clase al evaluar.
    """
    inputs = keras.Input(shape=input_shape)
    x = augmentation()(inputs)
    x = layers.Rescaling(1 / 255)(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_baseline")


def build_mobilenetv2(image_size: int = 224, num_classes: int = len(CLASSES),
                      dropout: float = 0.2) -> keras.Model:
    """MobileNetV2 preentrenada en ImageNet, congelada, con una capa de clasificación nueva."""
    base = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = augmentation()(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 espera píxeles en [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def unfreeze_top_layers(model: keras.Model, n_layers: int = 30) -> int:
    """Descongela las últimas `n_layers` capas de la base para el ajuste fino.

    Las capas BatchNormalization se dejan congeladas para que sigan usando las estadísticas
    de ImageNet: en Keras 3 el `training=False` con que se llama a la base no se respeta
    durante `fit()`, y una BatchNormalization entrenable pasa a normalizar con las
    estadísticas de cada lote, lo que desestabiliza el ajuste fino.

    Devuelve el número total de capas de la base. Hay que volver a compilar el modelo.
    """
    base = next(layer for layer in model.layers
                if isinstance(layer, keras.Model) and layer.name != "augmentacion")
    base.trainable = True
    for i, layer in enumerate(base.layers):
        frozen = i < len(base.layers) - n_layers
        if frozen or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return len(base.layers)
