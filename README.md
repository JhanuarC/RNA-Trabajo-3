# RNA-Trabajo-3

Tercer trabajo de **Redes Neuronales y Algoritmos Bio-inspirados**.

Este repositorio contiene tres módulos independientes que abordan distintos problemas con redes neuronales: **regresión temporal (LSTM)**, **clasificación de imágenes (CNN con Transfer Learning)** y **sistema de recomendación híbrido (MLP)**.

---

## Estructura del repositorio

```
RNA-Trabajo-3/
├── modelo_lstm_modulo_1/                               # Módulo 1 — notebook, dataset, modelo y gráficas
│   ├── creacion_datos_globales.ipynb                   # Notebook único del módulo (dataset + regresión + LSTM)
│   ├── demanda_transporte_global.csv                   # Dataset sintético global (22 países, 144.738 filas)
│   ├── modelo_lstm_transporte.keras                    # Modelo LSTM entrenado
│   ├── scaler_transporte.pkl                           # MinMaxScaler ajustado a las features
│   ├── curva_aprendizaje_lstm.png
│   ├── prediccion_vs_real_lstm.png
│   └── prediccion_vs_real_lstm_japon.png               # Caso de estudio: Japón
│
├── modulo2_clasificacion_imagenes_conductor.ipynb     # Módulo 2
├── saved_models/                                       # Pesos PyTorch (.pth) del módulo 2
│   ├── best_mobilenet.pth
│   ├── best_resnet50.pth
│   └── best_efficientnet.pth
├── results/                                            # Reportes, matrices y curvas del módulo 2
│
├── modulo3_sistema_recomendacion_module3_prueba_new_dataset.ipynb  # Módulo 3 (versión final)
├── sistema_recomendacion_module3.ipynb                 # Módulo 3 (versión previa)
├── best_model.pth                                      # Pesos del recomendador (versión previa)
├── best_model_recom.pth                                # Pesos del recomendador (versión final)
│
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python 3.11+
- GPU CUDA recomendada (los notebooks detectan automáticamente CPU/GPU)

Instalación:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Los módulos 2 y 3 descargan sus datasets automáticamente desde Kaggle vía `kagglehub`. Asegúrate de tener configuradas tus credenciales de Kaggle (`~/.kaggle/kaggle.json`).

---

## Módulo 1 — Regresión: predicción de demanda de transporte (LSTM)

**Notebook:** [modelo_lstm_modulo_1/creacion_datos_globales.ipynb](modelo_lstm_modulo_1/creacion_datos_globales.ipynb)

### Objetivo
Predecir la demanda diaria de pasajeros en rutas turísticas a nivel **global**, considerando país, hemisferio, tipo de vehículo, método de pago, eventos especiales y estacionalidad climática.

### Pipeline
1. **Generación sintética del dataset global** (144.738 registros, 2024–2025) sobre 22 países agrupados por hemisferio (Norte / Sur / Tropical), con:
   - 3 tipos de transporte: `luxury_bus`, `tourist_van`, `train_express`.
   - 3 métodos de pago: `cash`, `card`, `digital_wallet` (incluye UPI, PIX, M-Pesa, Alipay).
   - Estacionalidad climática por hemisferio + estacionalidad semanal.
   - Eventos especiales como anomalías para la RNN: Año Nuevo Chino, Navidad global y Golden Week en Japón.
2. **Baseline:** Regresión Lineal con One-Hot Encoding (32 features).
3. **Modelo final:** LSTM multivariada con ventana temporal de **14 días** y **33 features** de entrada, agrupando secuencias por `(país, vehículo, método de pago)` y haciendo split cronológico real (80/20) para evitar leakage temporal. Incluye `BatchNormalization`, `Dropout(0.3)` y callbacks `EarlyStopping` + `ReduceLROnPlateau`.

### Arquitectura LSTM
```
LSTM(128, return_sequences=True) → BatchNorm → Dropout(0.3)
LSTM(64)                         → BatchNorm → Dropout(0.3)
Dense(32, relu) → Dense(1, linear)
```

### Resultados

| Modelo            | MAE     | RMSE    | MAPE | Precisión |
|-------------------|---------|---------|------|-----------|
| Regresión Lineal  | 143.37  | 183.04  | —    | —         |
| **LSTM**          | pendiente | pendiente | pendiente | pendiente |

> Las métricas finales del LSTM están pendientes de la próxima ejecución completa del notebook; los artefactos (`modelo_lstm_transporte.keras`, `scaler_transporte.pkl`) y las gráficas ya están actualizados con el nuevo dataset global.

Artefactos generados: [modelo_lstm_transporte.keras](modelo_lstm_modulo_1/modelo_lstm_transporte.keras), [scaler_transporte.pkl](modelo_lstm_modulo_1/scaler_transporte.pkl), [curva_aprendizaje_lstm.png](modelo_lstm_modulo_1/curva_aprendizaje_lstm.png), [prediccion_vs_real_lstm_japon.png](modelo_lstm_modulo_1/prediccion_vs_real_lstm_japon.png).

---

## Módulo 2 — Clasificación: comportamiento del conductor (CNN)

**Notebook:** [modulo2_clasificacion_imagenes_conductor.ipynb](modulo2_clasificacion_imagenes_conductor.ipynb)

### Objetivo
Clasificar imágenes de conductores en 5 comportamientos: `safe_driving`, `talking_phone`, `texting_phone`, `other_activities`, `turning`.

### Dataset
[Multi-Class Driver Behavior Image Dataset](https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset) (Kaggle). Split 80/10/10 con `class_weight` balanceado.

### Modelos comparados (Transfer Learning + Fine-Tuning)
- **MobileNetV2** — feature extractor congelado.
- **ResNet50** — `layer4` descongelada.
- **EfficientNet-B0** — solo clasificador entrenable.

Configuración común: `AdamW`, `lr=1e-3`, `ReduceLROnPlateau`, `EarlyStopping (patience=6)`, hasta 30 épocas, `224x224`, data augmentation (flip, rotation, color jitter).

### Resultados

| Modelo         | Accuracy | AUC-ROC | Recall `texting_phone` | Recall `talking_phone` |
|----------------|----------|---------|------------------------|------------------------|
| MobileNetV2    | 0.782    | 0.9547  | —                      | —                      |
| **ResNet50**   | **0.947** | **0.9972** | **0.970**         | **0.994**              |
| EfficientNet-B0 | 0.723   | 0.9070  | —                      | —                      |

**Mejor modelo: ResNet50.** Reportes y matrices de confusión en [results/](results/).

---

## Módulo 3 — Sistema de Recomendación de Destinos Turísticos

**Notebook:** [modulo3_sistema_recomendacion_module3_prueba_new_dataset.ipynb](modulo3_sistema_recomendacion_module3_prueba_new_dataset.ipynb)

### Objetivo
Recomendar a un usuario las categorías turísticas que más le interesarán y, en cascada, los destinos del mundo que mejor encajan con su perfil.

### Datasets (Kaggle)
- [Travel Review Ratings](https://www.kaggle.com/datasets/ishbhms/travel-review-ratings) — 5.456 usuarios, 24 categorías (Google Reviews Europa).
- [Popular Tourist Destinations](https://www.kaggle.com/datasets/cosmox23/popular-tourist-destinations-and-their-features) — 2.000 destinos del mundo con `Type`, `Avg Rating`, `Avg Cost`, etc.

### Arquitectura — `HybridRecommender` (MLP)
```
Input(user_features [24] ⊕ cat_one_hot [24])  →  48
Linear(48 → 64) → BatchNorm → ReLU → Dropout(0.3)
Linear(64 → 32) → BatchNorm → ReLU → Dropout(0.2)
Linear(32 → 1)  → Sigmoid (BCEWithLogitsLoss con pos_weight=4.6)
```

Entrenamiento: `Adam(lr=1e-3, weight_decay=1e-3)`, batch 256, hasta 100 épocas con `EarlyStopping (patience=10)`.

### Resultados (umbral 0.6)

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.9953 |
| Precision | 0.9788 |
| Recall    | 0.9955 |
| **F1**    | **0.9871** |

### Sistema en cascada
1. La red predice la probabilidad de cada categoría para el usuario.
2. Se toman las **Top-K categorías** y se mapean a tipos de destino (`Beach`, `Nature`, `Adventure`, `Religious`, `Historical`, `City`).
3. Se devuelven los **Top-K destinos** con mayor `Avg Rating` del tipo correspondiente, junto con país y costo medio diario.

Ejemplo para un perfil "Playa": *Grand Ruins (Vietnam)*, *Crystal Park (Canada)*, *Serene Falls (Australia)*…

---

## Cómo ejecutar

Cada módulo es un notebook independiente y autocontenido. Abre con Jupyter / VS Code y ejecuta las celdas en orden:

```powershell
jupyter notebook modelo_lstm_modulo_1/creacion_datos_globales.ipynb
jupyter notebook modulo2_clasificacion_imagenes_conductor.ipynb
jupyter notebook modulo3_sistema_recomendacion_module3_prueba_new_dataset.ipynb
```

Los pesos pre-entrenados ya están incluidos en el repositorio, por lo que es posible saltar la fase de entrenamiento y ejecutar solo evaluación/inferencia en los módulos 2 y 3.

---

## Como usar la herramienta web

- Ingresar a la [página web](https://huggingface.co/spaces/Jhanuar/RNA-TRABAJO-3)

- Ingresar como invitado o como administrador (la clave es usuario:admin,contraseña:admin)

## Autores
Trabajo desarrollado para la asignatura **Redes Neuronales y Algoritmos Bio-inspirados**.


