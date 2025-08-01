# Copyright (c) 2021 Rui Shu
import glob
import os
import subprocess

if not os.path.exists("entregable.zip"):
    raise FileNotFoundError("""No se pudo encontrar entregable.zip

    ¿Por qué ocurrió este error?
    Errores comunes incluyen:
    1. Ejecutar este código antes de crear tu archivo zip
    2. Crear tu archivo zip con un nombre diferente
    """.rstrip())

print("Descomprimiendo entregable.zip al directorio temporal test_zip")
subprocess.call("""
unzip entregable.zip -d test_zip
""", shell=True)

filepaths = ["test_zip/codebase/utils.py",
             "test_zip/codebase/models/vae.py",
             "test_zip/codebase/models/gmvae.py",
             "test_zip/codebase/models/ssvae.py"]

filedirpaths = ["test_zip/codebase",
                "test_zip/codebase/models",
                "test_zip/codebase/utils.py",
                "test_zip/codebase/models/gmvae.py",
                "test_zip/codebase/models/ssvae.py",
                "test_zip/codebase/models/vae.py"]

try:
    # Verificar que los archivos importantes existan
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print("Verificando {} en {}".format(filename, filepath))
        if os.path.exists(filepath):
            print("...¡Encontrado!")
        else:
            raise FileNotFoundError(
                """
    No se pudo encontrar {0} en la ruta {1}

    ¿Por qué ocurrió este error?
    Errores comunes incluyen:
    1. Comprimir los archivos en una jerarquía plana en lugar de preservar los directorios
    3. Comprimir los archivos desde el directorio raíz incorrecto
    2. Olvidar incluir {0} en tu archivo zip (incluye todos los archivos,
       incluso si no completaste la tarea para este archivo)
                """.format(filename, filepath).rstrip())

    # Verificar que no existan otros archivos y directorios
    paths = glob.glob("test_zip/**/*", recursive=True)
    for path in paths:
        if path not in filedirpaths:
            raise FileExistsError("""
     El archivo zip contiene un archivo o directorio no permitido: {0}

     ¿Por qué ocurrió este error?
     Errores comunes incluyen:
     1. Comprimir toda tu base de código
     2. Comprimir todo tu repositorio git... (¿por qué harías esto? ಠ_ಠ)
            """.format(path).rstrip())

finally:
    subprocess.call("""
    rm -r test_zip
    """, shell=True)
