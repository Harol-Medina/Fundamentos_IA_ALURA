# Fundamentos de IA - Alura

Proyecto educativo en formato Jupyter Notebook para practicar conceptos base de inteligencia artificial, aprendizaje automático, procesamiento de imágenes y generación de texto.

El archivo principal es `Fundamentos_IA.ipynb`. El notebook fue revisado y documentado para que cada bloque de código tenga comentarios más claros, explicaciones del objetivo de cada ejercicio y validaciones sencillas cuando se trabaja con la imagen `poseidon.png`.

## Archivos del proyecto

- `Fundamentos_IA.ipynb`: notebook principal con todos los ejercicios.
- `poseidon.png`: imagen usada en la sección de procesamiento de imágenes y predicción con MobileNetV2.
- `README.md`: documentación general del proyecto.

## Contenido trabajado

### Clase 1: Aprendizaje supervisado

Se trabajan problemas donde el modelo aprende usando ejemplos con respuestas conocidas.

Temas incluidos:

- Clasificación con el dataset Iris.
- Entrenamiento de un `DecisionTreeClassifier`.
- Comparación entre árbol de decisión y KNN.
- Normalización de variables con `StandardScaler`.
- Separación de datos en entrenamiento y prueba.
- Evaluación con exactitud.

También se incluye un ejemplo de regresión lineal:

- Creación de datos simulados de inmuebles.
- Predicción de precios a partir del tamaño.
- Interpretación de coeficiente e intercepto.
- Evaluación con R2, MAE y RMSE.
- Validación cruzada.
- Visualización de la recta de regresión.

### Clase 2: Modelos de clasificación y regresión

Se practican flujos completos de machine learning con datasets reales de `scikit-learn`.

Temas incluidos:

- Carga del dataset California Housing.
- Regresión lineal para predecir precios de viviendas.
- Visualización de valores reales contra valores predichos.
- Clasificación binaria usando el dataset Diabetes.
- Matriz de confusión para analizar aciertos y errores.
- Optimización de hiperparámetros con `GridSearchCV`.
- Uso de `RandomForestRegressor` para medir importancia de atributos.
- Comparación entre `RandomForestRegressor` y `GradientBoostingRegressor`.

### Clase 3: Aprendizaje no supervisado

Se revisan modelos que buscan patrones sin usar etiquetas de respuesta.

Temas incluidos:

- Agrupamiento con K-Means.
- Clustering jerárquico.
- Dendrogramas para visualizar uniones entre grupos.
- Normalización de datos antes del clustering.
- Reducción de dimensionalidad con PCA.
- Visualización de datos Iris en dos componentes principales.
- Interpretación de la varianza explicada por PCA.

### Clase 4: Procesamiento de imágenes y modelos preentrenados

Se trabaja con la imagen `poseidon.png` para entender cómo una imagen puede representarse como datos numéricos.

Temas incluidos:

- Lectura de imágenes con OpenCV.
- Conversión a escala de grises.
- Visualización de intensidades de píxeles.
- Ejemplo de arquitectura CNN con Keras.
- Uso de MobileNetV2 preentrenado con ImageNet.
- Preprocesamiento de imágenes para modelos de visión.
- Predicción de las 3 clases más probables para `poseidon.png`.

### Procesamiento de lenguaje natural

El notebook también incluye ejemplos básicos de texto:

- Bag of Words con `CountVectorizer`.
- Conversión de frases en matriz numérica.
- Generación de texto con `transformers.pipeline`.
- Parámetros básicos de generación como `temperature`, `top_p`, `top_k` y `repetition_penalty`.

## Mejoras realizadas

Se mejoró el notebook para que sea más fácil de estudiar y reutilizar:

- Se agregaron explicaciones por sección.
- Se ampliaron comentarios dentro del código.
- Se corrigieron textos mezclados entre español y portugués.
- Se aclaró la función de cada modelo y cada métrica.
- Se agregó `random_state` en modelos para resultados reproducibles.
- Se usó `stratify` en divisiones de clasificación para conservar proporciones de clases.
- Se corrigió `plt.show()` donde faltaba ejecutar la función.
- Se ordenó la visualización de gráficos con `plt.tight_layout()`.
- Se adaptó el procesamiento de imágenes para usar directamente `poseidon.png`.
- Se agregó respaldo para subir la imagen manualmente si el notebook se ejecuta en Google Colab.
- Se añadieron mensajes de error claros cuando falta la imagen o no puede leerse.
- Se agregó configuración para TensorFlow en Windows con `TF_ENABLE_ONEDNN_OPTS=0`.
- Se ajustó la celda de `transformers` para detectar correctamente PyTorch en notebooks.
- Se cambió el modelo de texto a `sshleifer/tiny-gpt2` para que la demostración sea más liviana.
- Se revisó el notebook para evitar problemas de codificación en tildes y caracteres especiales.

## Estado actual del notebook

El notebook quedó preparado para ejecutarse en Python 3.12 con las siguientes mejoras prácticas:

- Las celdas de TensorFlow/Keras configuran `TF_ENABLE_ONEDNN_OPTS=0` antes de importar TensorFlow. Esto ayuda a evitar inicializaciones lentas o conflictos de `oneDNN` en Windows.
- La celda de generación de texto importa `torch` explícitamente y limpia la caché interna de `transformers` cuando el notebook ya había cargado la librería antes de instalar PyTorch.
- El modelo de lenguaje usado por defecto es `sshleifer/tiny-gpt2`, que es pequeño y útil para pruebas educativas.
- `poseidon.png` se carga directamente desde la carpeta del proyecto y, si no existe en Colab, se permite subirlo manualmente.
- Los textos del notebook y del README fueron revisados para conservar correctamente tildes como `á`, `é`, `í`, `ó`, `ú`, `ñ` y signos especiales como `²`.

## Requisitos

Para ejecutar todo el notebook se recomienda usar Python 3.10 o superior.

Bibliotecas principales:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy opencv-python tensorflow transformers torch
```

Notas:

- Algunas celdas descargan datasets desde `scikit-learn`.
- La celda de MobileNetV2 puede descargar pesos preentrenados de ImageNet la primera vez que se ejecuta.
- La celda de `transformers` usa PyTorch (`torch`) y puede descargar el modelo `sshleifer/tiny-gpt2` si no está guardado en caché.
- Puedes cambiar `model_id = "sshleifer/tiny-gpt2"` por `model_id = "gpt2"` si quieres probar el modelo GPT-2 completo.
- Si se ejecuta en Google Colab, normalmente varias dependencias ya están instaladas.

## Solución de errores comunes

### TensorFlow no importa correctamente

Si aparece un error relacionado con TensorFlow, reinicia el kernel y vuelve a ejecutar la celda. El notebook ya incluye esta configuración antes de importar TensorFlow:

```python
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
```

### `transformers` no detecta PyTorch

Si aparece un error como `PyTorch should be installed` o `name 'torch' is not defined`, asegúrate de tener instalado `torch` y reinicia el kernel de Jupyter. En VS Code puedes hacerlo desde:

```text
Ctrl+Shift+P -> Jupyter: Restart Kernel
```

Después ejecuta nuevamente la celda completa de generación de texto.

### La imagen `poseidon.png` no se encuentra

Verifica que `poseidon.png` esté en la misma carpeta que `Fundamentos_IA.ipynb`. Si ejecutas el notebook en Colab, sube la imagen cuando la celda lo solicite.

## Cómo ejecutar

### Opción 1: Jupyter local

1. Abre una terminal en la carpeta del proyecto.
2. Instala las dependencias necesarias.
3. Ejecuta Jupyter:

```bash
jupyter notebook
```

4. Abre `Fundamentos_IA.ipynb`.
5. Ejecuta las celdas en orden.

### Opción 2: Google Colab

1. Sube `Fundamentos_IA.ipynb` a Google Colab.
2. Sube también `poseidon.png` cuando llegues a las celdas de imagen, o déjalo disponible en la misma carpeta del notebook.
3. Ejecuta las celdas en orden.

## Imagen usada

La imagen principal del proyecto es:

```text
poseidon.png
```

Esta imagen se utiliza para:

- Convertir una imagen a escala de grises.
- Mostrar una muestra de valores de píxeles.
- Probar una predicción con MobileNetV2.

## Objetivo del proyecto

El objetivo es tener una guía práctica y comentada de fundamentos de IA. El notebook funciona como material de estudio porque muestra el ciclo completo de varios ejercicios: cargar datos, preparar entradas, entrenar modelos, evaluar resultados y visualizar lo aprendido.
