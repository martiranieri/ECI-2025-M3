# 📚 Entregable

¡Bienvenido a tu tarea de programación de VAEs! 🎯

Se debe entregar un documento escrito con las respuestas de las diferentes partes y las imágenes solicitadas, junto con el archivo `entregable.zip` conteniendo los archivos modificados por ustedes.

## 💻 Tarea de programación

Puedes explorar todos los archivos sin restricciones. Los únicos que necesitarás modificar son:

1. 🔧 `codebase/utils.py`
1. 🤖 `codebase/models/vae.py`
1. 🎨 `codebase/models/gmvae.py`
1. 📊 `codebase/models/ssvae.py`

⚠️ **¡Importante!** No modifiques los otros archivos. Todos los hiperparámetros por defecto ya han sido cuidadosamente preparados para ti, así que por favor no los cambies. Si decides hacer la tarea de programación desde cero, copia fielmente los hiperparámetros para que tus resultados sean comparables con lo que esperamos.

⏰ **Tip:** Los modelos pueden tardar un tiempo en ejecutarse en CPU, así que ¡prepárate con tiempo y paciencia!

## 🎨 Recursos adicionales

También puedes crear nuevos archivos o notebooks de Jupyter para ayudarte a responder cualquiera de las preguntas escritas del trabajo. Para la generación de imágenes, te serán súper útiles las siguientes funciones:

1. 📦 `codebase.utils.load_model_by_name` (para cargar un modelo. Ver ejemplo de uso en `run_vae.py`)
1. 🎲 Las funcionalidades de muestreo en `vae.py`/`gmvae.py`/`ssvae.py`/`fsvae.py`
1. 🔄 `numpy.swapaxes` y/o `torch.permute` (para organizar imágenes en mosaico cuando se representan como arrays de numpy)
1. 🖼️ `matplotlib.pyplot.imshow` (para generar una imagen a partir de un array de numpy)

## ✅ Lista de verificación

A continuación tienes una lista de verificación de las distintas funciones que necesitas implementar en la base de código, ¡en orden cronológico!:

1. ⭐ `sample_gaussian` en `utils.py`
1. 🔍 `negative_elbo_bound` en `vae.py`
1. 📈 `log_normal` en `utils.py`
1. 🎯 `log_normal_mixture` en `utils.py`
1. 🤖 `negative_elbo_bound` en `gmvae.py`
1. 📊 `negative_elbo_bound` en `ssvae.py`

🎉 **¡Casi listo!** Una vez que hayas completado la tarea, ejecuta el script `make_submission.sh` y sube el archivo `entregable.zip`.

---

## 🚀 Configuración del Entorno Virtual

### 🐍 Opción 1: Usando Conda (Recomendado)

Si tienes Anaconda o Miniconda instalado:

```bash
# Crear entorno virtual con Python 3.8.20
conda create -n vae-env python=3.8.20

# Activar el entorno
conda activate vae-env

# Instalar las dependencias
pip install -r requirements.txt
```

### 🔧 Opción 2: Usando venv

Si prefieres usar el módulo venv estándar de Python:

```bash
# Crear entorno virtual
python3.8 -m venv vae-env

# Activar el entorno (Linux/Mac)
source vae-env/bin/activate

# Activar el entorno (Windows)
# vae-env\Scripts\activate

# Instalar las dependencias
pip install -r requirements.txt
```

💡 **Nota:** Asegúrate de tener Python 3.8.20 instalado en tu sistema antes de usar la opción venv.

### ✅ Verificar la instalación

Para verificar que todo está correctamente instalado:

```bash
python -c "import torch, torchvision, numpy, tqdm; print('¡Todas las dependencias instaladas correctamente!')"
```

---

### 📚 Dependencias

Este código fue construido y probado usando las siguientes librerías:

```
numpy==1.24.4
torchvision==0.9.1
torch==1.8.1
tqdm==4.59.0
```

---

### 📦 Entrega

Utiliza `make_submission.sh` para comprimir los archivos apropiados. Este script simplemente comprime los siguientes archivos en `entregable.zip`:
```
codebase/utils.py 
codebase/models/vae.py 
codebase/models/gmvae.py
codebase/models/ssvae.py 
```

🗂️ **Ten en cuenta** que `entregable.zip` preserva los directorios. Si estás creando `entregable.zip` manualmente, puede que desees verificar tu archivo zip ejecutando:
```
python verify_submission.py
```

🔍 Para hacer la inspección manualmente, simplemente ejecuta:
```
unzip entregable.zip -d tmp_dir
```
Y luego verifica que `tmp_dir` tenga la siguiente estructura de archivos:
```
tmp_dir/
└── codebase
    ├── models
    │   ├── gmvae.py
    │   ├── ssvae.py
    │   └── vae.py
    └── utils.py
```
