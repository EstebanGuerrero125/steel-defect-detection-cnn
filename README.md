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

## Hoja de ruta

- [ ] 1. Descargar y explorar el dataset NEU
- [ ] 2. Entrenar una CNN pequeña desde cero en Keras (baseline)
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
