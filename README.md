# Análisis de Sentimiento en Tweets — Fine-Tuning con Transformers

Fine-tuning de modelos transformer sobre el dataset **TweetEval** (subtarea `sentiment`) para clasificación de tweets en tres categorías: **negativo**, **neutro** y **positivo**.

Incluye interfaz interactiva con explicaciones generativas mediante **Flan-T5**.

---

## Contenido del repositorio

```
trabajo_final_mineria_textos/
├── SA_TweetEval_Transformers_FineTuning.ipynb   # notebook de entrenamiento
├── app.py                                        # interfaz Gradio
├── modelo_guardado/                              # modelo fine-tuned (tras descomprimir)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── requirements.txt
```

---

## Requisitos previos

- Python **3.10, 3.11 o 3.12** (Python 3.13 puede tener incompatibilidades con algunas dependencias)
- pip actualizado

---

## Instalación

### 1. Clonar o descargar el repositorio

Este repositorio usa **Git LFS** para los archivos del modelo. Asegúrate de tener Git LFS instalado:

```bash
# Instalar Git LFS (solo la primera vez)
git lfs install
```

```bash
git clone <url-del-repo>
cd trabajo_final_mineria_textos
git lfs pull   # descarga el modelo (~463 MB)
```

### 2. Crear entorno virtual (recomendado)

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota macOS con Apple Silicon (M1/M2/M3):** PyTorch se instala con soporte MPS automáticamente desde la versión 2.x. No se necesita configuración adicional.

> **Nota Windows con GPU NVIDIA:** instala primero PyTorch con CUDA desde [pytorch.org](https://pytorch.org/get-started/locally/) antes de ejecutar `pip install -r requirements.txt`.

### 4. Descomprimir el modelo (si se entrega como zip)

Si `modelo_guardado/` no existe y tienes `modelo_guardado.zip`:

**macOS / Linux:**
```bash
unzip modelo_guardado.zip -d modelo_guardado
```

**Windows:**
```bash
# con PowerShell
Expand-Archive -Path modelo_guardado.zip -DestinationPath modelo_guardado
```

---

## Lanzar la interfaz

Con el modelo en `./modelo_guardado/`, ejecuta:

```bash
python app.py
```

Se abrirá automáticamente en el navegador en `http://127.0.0.1:7860`.

La primera ejecución descargará **Flan-T5-base** (~250 MB) desde HuggingFace. Las siguientes arrancarán desde caché.

> Si ejecutas desde un servidor remoto (SSH, Colab), añade `share=True` en `demo.launch()` para obtener un enlace público temporal.

---

## Entrenamiento (notebook)

El notebook `SA_TweetEval_Transformers_FineTuning.ipynb` está diseñado para ejecutarse en **Google Colab con GPU T4**.

Compara tres estrategias:

| Estrategia | Modelo base | Preprocesamiento |
|---|---|---|
| E1 | `distilbert-base-uncased` | estándar |
| E2 | `cardiffnlp/twitter-roberta-base` | estándar |
| E3 | `distilbert-base-uncased` | limpieza de URLs/menciones |

El mejor modelo (por F1-macro en test) se guarda automáticamente en `/content/modelo_guardado` y se puede descargar como zip desde el panel de archivos de Colab.

---

## Tecnologías

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [TweetEval dataset](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [Gradio](https://gradio.app)
- [Flan-T5](https://huggingface.co/google/flan-t5-base)
