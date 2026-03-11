🥊 Frame-Data-Bayes

"En un frame se gana o se pierde la partida. En un dato se asegura o se compromete el sistema."

Frame-Data-Bayes es un motor de análisis probabilístico y clasificación estadística desarrollado bajo estándares de Clean Code y mentalidad Zero Trust. El sistema trata cada dataset como un entorno hostil: valida superficies de entrada, aísla la memoria visual, detecta tipos de datos de forma determinística y aplica inferencia bayesiana para predecir eventos anómalos sin bloquear el hilo principal (GIL).

🛠️ Tech Stack & Arquitectura

El proyecto está orquestado mediante una separación estricta de responsabilidades (SRP) para minimizar la deuda técnica y maximizar la escalabilidad:

Core Lógico: Python 3.12+ con Pandas y NumPy para procesamiento vectorial (SIMD).

Motor de Inferencia: Scikit-Learn (Gaussian Naïve Bayes) con imputación segura de dimensionalidad.

Capa de Presentación: Streamlit para una interfaz reactiva, manejando estados aislados (Sandboxing).

Motor Gráfico: Matplotlib (API Orientada a Objetos) & Seaborn aislando la memoria para prevenir fugas (Memory Leaks).

🚀 Funcionalidades (Key Bindings)

Ingesta Segura: Carga de CSV con validación de integridad (mitigación de DoS por archivos masivos).

Escaneo de Superficie: Detección automática de vectores numéricos, categóricos, temporales y binarios.

Cálculo Bayesiano Vectorizado: Implementación estricta del Teorema de Bayes optimizada en C (vía NumPy):

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

Clasificador Naïve Bayes: Entrenamiento de modelo con reportes de Accuracy, Sensibilidad, Especificidad y Validación Cruzada (Cross-Validation).

Visualización Thread-Safe: Generación de hitboxes estadísticos (histogramas, series temporales, matrices térmicas) con destrucción garantizada en el recolector de basura.

⚙️ Guía de Instalación y Ejecución (Setup)

Para garantizar la integridad del entorno y evitar conflictos de dependencias (Worldline Divergence), siga estos pasos estrictamente:

1. Requisitos Previos

Python 3.10 o superior (Recomendado 3.12+).

Git instalado en su sistema.

2. Extracción del Repositorio (Clonar)

Para replicar el entorno en su máquina local, primero clone el repositorio usando Git y navegue a la carpeta raíz del proyecto:

# Clone el repositorio (Reemplace la URL con su repositorio real)

git clone https://github.com/RyuheiRG/frame-data-bayes.git

# Ingrese al directorio raíz del proyecto

cd frame-data-bayes

(Nota: Si descargó el código como un archivo .zip, simplemente descomprímalo, abra su terminal y navegue hasta la carpeta extraída).

3. Aislamiento del Entorno (Virtual Environment)

Una vez dentro de la carpeta raíz, cree y active un entorno virtual para aislar las dependencias del sistema global:

En Windows:

python -m venv venv
venv\Scripts\activate

En macOS / Linux:

python3 -m venv venv
source venv/bin/activate

(Confirmación de Hit: Sabrá que el entorno está activo si ve (venv) al inicio de su línea de comandos).

4. Inyección de Dependencias

Con la capa de aislamiento activa, instale los paquetes y dependencias estrictas requeridas por el sistema:

pip install -r requirements.txt

5. Inicialización del Motor (Deploy)

⚠️ Paso Crítico (Routing): Debido a la arquitectura modular de Zero Trust, el servidor de Streamlit debe ejecutarse estrictamente dentro de la carpeta app/ para que el orquestador pueda resolver correctamente las rutas de los submódulos.

# 1. Navegar a la capa de la aplicación

cd app

# 2. Ejecutar el orquestador

streamlit run main.py

El sistema levantará un servidor local y abrirá la interfaz automáticamente en su navegador web seguro (http://localhost:8501).

📂 Estructura del Proyecto

.
├── app/
│ ├── analytics/
│ │ └── insights.py # Diagnóstico estadístico e Información Mutua
│ ├── assets/ # Recursos estáticos
│ ├── ingestion/
│ │ └── data_loader.py # Pipeline de ingesta Zero Trust y tipado estricto
│ ├── models/
│ │ └── bayes_engine.py # Lógica matemática (vectorizada) y GaussianNB
│ ├── ui/
│ │ ├── bayes_panel.py # Interfaz del motor bayesiano y control de estado
│ │ ├── eda_panel.py # Interfaz de Análisis Exploratorio de Datos
│ │ └── insights_panel.py # Interfaz de presentación de métricas y validación
│ ├── visualization/
│ │ └── visualizer.py # Renderizado de gráficas (Matplotlib OO API)
│ └── main.py # Entrypoint y Orquestador de Streamlit
├── venv/ # Entorno virtual (Aislado)
├── .gitignore # Reglas de exclusión de control de versiones
├── requirements.txt # Manifiesto estricto de dependencias
└── README.md # Documentación arquitectónica

Desarrollado bajo protocolos de Clean Code y Seguridad Arquitectónica.
