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

## ⚙️ Instalación Rápida

1. **Clonar el repositorio**

```bash
git clone https://github.com/RyuheiRG/frame-data-bayes.git
cd frame-data-bayes

Crear y activar entorno virtual

Windows:

python -m venv venv
venv\Scripts\activate

Linux / macOS:

python3 -m venv venv
source venv/bin/activate

Instalar dependencias

pip install -r requirements.txt

Ejecutar la aplicación

cd app
streamlit run main.py

La aplicación se iniciará en:

http://localhost:8501
```

## 📂 Estructura del Proyecto

```text
.
├── app/
│   ├── analytics/
│   │   └── insights.py          # Generación de insights analíticos
│   ├── assets/                  # Recursos estáticos (imágenes, estilos, etc.)
│   ├── data/
│   │   └── data_loader.py       # Pipeline de ingesta y validación de datos
│   ├── models/
│   │   └── bayes_engine.py      # Lógica matemática y entrenamiento del modelo
│   ├── ui/
│   │   ├── bayes_panel.py       # Interfaz del motor bayesiano
│   │   ├── eda_panel.py         # Interfaz de Análisis Exploratorio (EDA)
│   │   └── insights_panel.py    # Interfaz de presentación de insights
│   ├── visualization/
│   │   └── visualizer.py        # Renderizado de matrices y gráficos
│   └── main.py                  # Orquestador de la UI (Streamlit)
├── venv/                        # Entorno virtual aislado
├── .gitignore                   # Exclusiones de control de versiones
├── README.md                    # Documentación principal
└── requirements.txt             # Manifiesto de dependencias
```
