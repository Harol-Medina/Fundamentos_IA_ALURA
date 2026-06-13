# Fundamentos de IA - Alura

Proyecto educativo en formato Jupyter Notebook para practicar conceptos básicos de inteligencia artificial, aprendizaje automático, procesamiento de imágenes y procesamiento de lenguaje natural.

El archivo principal es `Fundamentos_IA.ipynb`. El notebook contiene ejemplos comentados paso a paso para entender cómo se cargan datos, se entrenan modelos, se evalúan resultados y se visualizan predicciones.

## Archivos del proyecto

- `Fundamentos_IA.ipynb`: notebook principal con todos los ejercicios.
- `poseidon.png`: imagen usada en la sección de procesamiento de imágenes.
- `.gitignore`: evita subir archivos innecesarios como el entorno virtual `.venv`.
- `README.md`: documentación general del proyecto.

## Contenido del notebook

### Clase 1: Aprendizaje supervisado

En esta sección se trabajan modelos que aprenden a partir de datos con respuestas conocidas.

Temas principales:

- Clasificación con el dataset Iris.
- Entrenamiento de un árbol de decisión.
- Comparación entre árbol de decisión y KNN.
- Normalización de datos con `StandardScaler`.
- Separación entre datos de entrenamiento y prueba.
- Evaluación de modelos con exactitud.
- Regresión lineal con datos simulados de inmuebles.
- Evaluación de regresión con R2, MAE y RMSE.
- Visualización de la recta de regresión.

### Clase 2: Clasificación, regresión y optimización

En esta clase se usan datasets de `scikit-learn` para practicar flujos completos de machine learning.

Temas principales:

- Dataset California Housing.
- Regresión lineal para estimar precios de viviendas.
- Comparación entre valores reales y predichos.
- Clasificación binaria con el dataset Diabetes.
- Matriz de confusión.
- Optimización de hiperparámetros con `GridSearchCV`.
- Importancia de atributos con `RandomForestRegressor`.
- Comparación entre Random Forest y Gradient Boosting.

### Clase 3: Aprendizaje no supervisado

En esta sección se trabajan modelos que buscan patrones sin recibir etiquetas como respuesta.

Temas principales:

- Agrupamiento con K-Means.
- Clustering jerárquico.
- Dendrogramas.
- Normalización antes del clustering.
- Reducción de dimensionalidad con PCA.
- Visualización del dataset Iris usando dos componentes principales.
- Interpretación de la varianza explicada por PCA.

### Clase 4: Procesamiento de imágenes

Esta sección usa la imagen `poseidon.png` para mostrar cómo una imagen puede convertirse en datos numéricos y cómo puede ser analizada por modelos de visión por computadora.

Temas principales:

- Lectura de imágenes con OpenCV.
- Conversión a escala de grises.
- Visualización de intensidades de píxeles.
- Creación de una arquitectura CNN básica con Keras.
- Uso de MobileNetV2 preentrenado con ImageNet.
- Predicción de las 3 clases más probables para `poseidon.png`.

### Procesamiento de lenguaje natural

El notebook también incluye ejemplos introductorios de NLP.

Temas principales:

- Bag of Words con `CountVectorizer`.
- Conversión de frases a matrices numéricas.
- Generación de texto con `transformers.pipeline`.
- Uso del modelo `sshleifer/tiny-gpt2` para una demostración liviana.

## Requisitos

Se recomienda usar Python 3.10 o superior.

Instala las dependencias principales con:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy opencv-python tensorflow transformers torch
```

Notas:

- Algunas celdas pueden descargar datasets de `scikit-learn`.
- La celda de MobileNetV2 puede descargar pesos preentrenados la primera vez que se ejecuta.
- La celda de generación de texto puede descargar `sshleifer/tiny-gpt2` la primera vez que se ejecuta.
- Si ejecutas el notebook en Google Colab, varias dependencias ya pueden estar instaladas.

## Cómo ejecutar el proyecto

### Opción 1: Jupyter local

1. Abre una terminal en la carpeta del proyecto.
2. Instala las dependencias.
3. Inicia Jupyter:

```bash
jupyter notebook
```

4. Abre `Fundamentos_IA.ipynb`.
5. Ejecuta las celdas en orden.

### Opción 2: Visual Studio Code

1. Abre la carpeta del proyecto en VS Code.
2. Abre `Fundamentos_IA.ipynb`.
3. Selecciona un kernel de Python con las dependencias instaladas.
4. Ejecuta las celdas en orden.

Si instalaste dependencias mientras el notebook estaba abierto, reinicia el kernel antes de ejecutar nuevamente:

```text
Ctrl+Shift+P -> Jupyter: Restart Kernel
```

### Opción 3: Google Colab

1. Sube `Fundamentos_IA.ipynb` a Google Colab.
2. Sube también `poseidon.png` cuando trabajes con las celdas de imagen.
3. Ejecuta las celdas en orden.

## Imagen usada

La imagen principal del proyecto es:

```text
poseidon.png
```

Se usa para:

- Mostrar una imagen en escala de grises.
- Revisar valores de píxeles.
- Probar predicciones con MobileNetV2.

## Buenas prácticas del repositorio

El proyecto incluye `.gitignore` para evitar subir archivos generados automáticamente o dependientes de la máquina local.

## Objetivo del proyecto

El objetivo es tener una guía práctica de fundamentos de IA con ejemplos simples y ejecutables. El notebook sirve como material de estudio para comprender el flujo básico de un proyecto de machine learning: preparar datos, entrenar modelos, evaluar resultados y visualizar conclusiones.

