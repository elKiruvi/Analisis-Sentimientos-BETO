# Análisis de Sentimientos en Tweets usando BETO

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-F9AB00?logo=huggingface&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)

Este repositorio contiene un pipeline de Procesamiento de Lenguaje Natural (PLN) enfocado en la clasificación de emociones y sentimientos en textos cortos informales (tweets) redactados en español. El proyecto documenta la implementación y evaluación de técnicas de *Transfer Learning* mediante el *Fine-Tuning* del modelo fundacional BETO (`dccuchile/bert-base-spanish-wwm-cased`).

## Resumen de Resultados

El proceso de fine-tuning demostró una mejora significativa en el rendimiento predictivo en comparación con el estado inicial del modelo (cuyos pesos en la capa de clasificación se inicializan de forma aleatoria):

* **Accuracy previo al Fine-Tuning:** `13.11%` (Métrica inferior a la probabilidad aleatoria para 6 clases).
* **Accuracy posterior al Fine-Tuning:** `70.44%`
* **F1-Score (Macro) Final:** `70.25%`
* **Incremento Absoluto de Precisión:** `+57.33%` tras 8 épocas de entrenamiento.

---

## Tecnologías y Dependencias

* **Modelo Base:** BETO (Modelo BERT preentrenado en español, desarrollado por la Universidad de Chile).
* **Deep Learning Framework:** PyTorch.
* **Ecosistema Hugging Face:** `transformers` (Tokenización, Trainer API, Pipelines) y `datasets` (Optimización y manejo eficiente de memoria).
* **Análisis y Visualización de Datos:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.

---

## Arquitectura del Pipeline

El flujo de trabajo está estructurado en las siguientes fases secuenciales:

1. **Análisis Exploratorio de Datos (EDA):** Evaluación de la distribución de clases, longitud de secuencias y detección de desbalances.
2. **Preprocesamiento Textual:** Normalización mediante Expresiones Regulares (RegEx) para la eliminación de URLs, menciones y hashtags, preservando la carga semántica de caracteres especiales como los emojis.
3. **Evaluación *Zero-Shot*:** Comparativa preliminar de modelos fundacionales preentrenados en español (BETO, RoBERTuito).
4. **Tokenización Optimizada:** Implementación de procesamiento por lotes (*batched mapping*) mediante la librería `datasets`.
5. **Fine-Tuning:** Configuración del bucle de entrenamiento incorporando optimizaciones como Precisión Mixta (FP16), *Gradient Accumulation*, planificador de tasa de aprendizaje Cosine y *Early Stopping*.
6. **Evaluación Continua e Inferencia:** Generación de métricas globales, matrices de confusión y pruebas heurísticas de inferencia en tiempo real.

---

## Visualizaciones

### 1. Distribución del Dataset

<img width="1388" height="490" alt="image" src="https://github.com/user-attachments/assets/cda64fd4-2f3e-4177-8452-2939df8704ce" />

### 2. Curvas de Entrenamiento (Loss y F1-Score)

<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/672a3be0-8036-4e09-83f5-77d3689b8115" />

### 3. Impacto del Transfer Learning

<img width="690" height="489" alt="image" src="https://github.com/user-attachments/assets/c08d7d22-9898-47fb-868b-5a1d1e5fdde9" />

---

## Conclusiones del Análisis

El análisis de los resultados obtenidos permite establecer las siguientes conclusiones:

### 1. Viabilidad del Transfer Learning
El experimento valida empíricamente la efectividad del fine-tuning en tareas de clasificación de texto. La adaptación de los pesos lingüísticos generales de BETO hacia un dominio semántico específico permitió escalar el Accuracy del 13.11% al 70.44% en un ciclo de entrenamiento corto (8 épocas).

### 2. Resiliencia frente al Desbalance de Clases
A pesar del desbalance intrínseco en el conjunto de datos original, la convergencia obtenida entre el Accuracy (70.44%) y el F1-Score Macro (70.25%) demuestra que el modelo no desarrolló un sesgo perjudicial hacia la clase mayoritaria. El análisis de la matriz de confusión revela que los falsos positivos ocurren mayoritariamente entre clases semánticamente adyacentes, indicando la captura de representaciones latentes coherentes.

### 3. Estabilidad en la Convergencia
Las curvas de optimización reflejan un proceso estable. La ausencia de divergencia en la función de pérdida de validación (*Validation Loss*) descarta la presencia de un sobreajuste (*overfitting*) severo, justificando la selección de hiperparámetros de regularización como *Weight Decay* (0.01) y *Dropout* (0.2).

### 4. Generalización y Ambigüedad Lingüística
Las pruebas empíricas de inferencia con datos no vistos evidencian la capacidad del modelo para gestionar la polaridad extrema y resolver la ambigüedad generada por modismos locales, priorizando el núcleo semántico de la oración por encima de la gramática informal.

---

## Trabajo Futuro

Para iteraciones posteriores de este proyecto, se proponen las siguientes líneas de investigación:

* **Adopción de Modelos de Dominio Específico:** Replicar el experimento utilizando arquitecturas como RoBERTuito, cuyos *embeddings* fueron entrenados nativamente sobre un corpus de la red social X (Twitter), optimizando la lectura de sintaxis informal.
* **Aumento de Datos (Data Augmentation):** Incorporar técnicas como *back-translation* o generación sintética mediante Modelos de Lenguaje Grande (LLMs) para mitigar el desbalance en las clases minoritarias.
