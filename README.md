# 🥊 Frame-Data-Bayes

> **"En un frame se gana o se pierde la partida. En un dato se asegura o se compromete el sistema."**

**Frame-Data-Bayes** es un motor de análisis probabilístico y clasificación estadística desarrollado bajo estándares de **Clean Code** y mentalidad **Zero Trust**. El sistema trata cada dataset como un entorno hostil: valida superficies de entrada, detecta tipos de datos de forma determinística y aplica inferencia bayesiana para predecir eventos anómalos.

---

## 🛠️ Tech Stack & Arquitectura

El proyecto está orquestado mediante una separación estricta de responsabilidades para minimizar la deuda técnica y maximizar la escalabilidad:

- **Core Lógico:** `Python 3.12+` con `Pandas` y `NumPy` para procesamiento vectorial.
- **Motor de Inferencia:** `Scikit-Learn` (Gaussian Naïve Bayes) para clasificación predictiva.
- **Capa de Presentación:** `Streamlit` para una interfaz reactiva y desacoplada.
- **Motor Gráfico:** `Matplotlib` & `Seaborn` para la visualización de matrices de confusión y desplazamientos de probabilidad.

## 🚀 Funcionalidades (Key Bindings)

1.  **Ingesta Segura:** Carga de CSV con validación de integridad y manejo de excepciones.
2.  **Escaneo de Superficie:** Detección automática de vectores numéricos, categóricos, temporales y binarios.
3.  **Cálculo Bayesiano Puro:** Implementación estricta del Teorema de Bayes:
    $$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$
4.  **Entrenamiento Naïve Bayes:** Clasificador para detección de anomalías con reportes de Accuracy, Sensibilidad y Especificidad.
5.  **Visualización de Hitboxes:** Generación de histogramas, series temporales y matrices de confusión optimizadas para evitar el overflow de texto.

## 📂 Estructura del Proyecto

```text
.
├── app/
│   ├── main.py              # Orquestador de la UI (Streamlit)
│   └── modules/
│       ├── data_loader.py   # Pipeline de ingesta y validación
│       ├── bayes_engine.py  # Lógica matemática y entrenamiento
│       └── visualizer.py    # Renderizado de assets gráficos
├── data/
│   ├── raw/                 # Datasets originales (Ignorados en Git)
│   └── processed/           # Datos sanitizados
├── notebooks/               # Sandbox de experimentación
└── requirements.txt         # Manifiesto de dependencias
```
