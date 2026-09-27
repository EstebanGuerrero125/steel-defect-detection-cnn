# Detección de defectos superficiales en acero con CNN

Clasificación de defectos superficiales en láminas de acero laminado en caliente usando redes neuronales convolucionales (CNN), con un modelo servido mediante una API REST.

El control de calidad visual es un proceso clave en normas como ISO 9001: automatizar la inspección reduce errores humanos y permite trazabilidad de cada decisión.

## Dataset

[NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) — 1.800 imágenes en escala de grises (200×200 px), 300 por clase:

| Clase | Descripción |
|---|---|
| Crazing | Grietas finas en red |
| Inclusion | Inclusiones de material no metálico |
| Patches | Manchas |
| Pitted surface | Superficie picada |
| Rolled-in scale | Cascarilla incrustada por laminación |
| Scratches | Rayones |

### Hallazgos de la exploración

- Dataset balanceado: 240 imágenes por clase en `train` y 60 en `validation`. No trae partición de test, así que `validation` se reserva como **test** y la validación durante el entrenamiento sale de un 15 % estratificado de `train`.
- Todas las imágenes son de 200×200 y ninguna está corrupta. Vienen en RGB, pero son en escala de grises, así que el baseline usa 1 canal.
- Hay 1 par de imágenes duplicadas dentro de `train` (`patches_101` y `patches_105`) y ninguna fuga entre particiones. El duplicado se elimina antes de entrenar.
- El brillo medio varía mucho entre clases (de 95 en `scratches` a 177 en `pitted_surface`). Para evitar que el modelo clasifique por iluminación, se usa aumento de datos con cambios de brillo y contraste.

## Hoja de ruta

- [x] 1. Descargar y explorar el dataset NEU — [`01_exploracion_dataset.ipynb`](notebooks/01_exploracion_dataset.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EstebanGuerrero125/steel-defect-detection-cnn/blob/main/notebooks/01_exploracion_dataset.ipynb)
- [ ] 2. Entrenar una CNN pequeña desde cero en Keras (baseline) — [`02_cnn_baseline.ipynb`](notebooks/02_cnn_baseline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EstebanGuerrero125/steel-defect-detection-cnn/blob/main/notebooks/02_cnn_baseline.ipynb)
- [ ] 3. Transfer learning con MobileNetV2
- [ ] 4. Evaluación: matriz de confusión y `classification_report` (scikit-learn)
- [ ] 5. Endpoint `/predict` en FastAPI con el modelo guardado

## Estructura

```
├── api/              # Servicio FastAPI (/predict)
├── data/             # Dataset (no se versiona)
├── models/           # Modelos entrenados (.keras)
├── notebooks/        # Exploración y entrenamiento (Colab)
├── reports/figures/  # Matrices de confusión, curvas de entrenamiento
├── src/              # Código reutilizable (carga de datos, preprocesamiento)
└── requirements.txt
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Stack

Python · TensorFlow/Keras · scikit-learn · FastAPI · Google Colab (GPU)
