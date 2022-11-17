import pickle
import os
from datetime import datetime
import pandas as pd


def load_data_temp():
    with open("temp/datos_temp.pkl", "rb") as tf:
        dicc = pickle.load(tf)

    minimo = dicc["minimo"]
    maximo = dicc["maximo"]
    nombre_archivo = dicc["archivo"]

    df = pd.read_csv(f"temp/{nombre_archivo}")

    return df, minimo, maximo


def temporal_storage(minimo, maximo, data):
    try:
        os.mkdir('temp')
    except OSError:
        print("Correpto !!\tCarpeta temp exite")

    now = datetime.now()
    nombre_archivo = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.second}.csv"
    data.to_csv(f"temp/{nombre_archivo}")

    dicc = {"minimo": minimo, "maximo": maximo, "archivo": nombre_archivo}

    with open("temp/datos_temp.pkl", "wb") as tf:
        pickle.dump(dicc, tf)


def guardar_metricas(metricas):
    file = 'resultados_metricas.xlsx'
    folder = 'resultados'
    path = f'{folder}/{file}'

    if not os.path.isdir(folder):
        os.mkdir(folder)

    if os.path.isfile(path):
        pandas_metrics = pd.read_excel(path, index_col=[0])
        pandas_metrics = pd.concat([pandas_metrics, metricas], ignore_index=True)
    else:
        pandas_metrics = metricas

    pandas_metrics.to_excel(path)


def guardar_prompt(prompt, version_prompt=None):
    nombre_archivo = "versiones_prompt"
    ruta = f"prompt_examples/{nombre_archivo}.xlsx"

    try:
        if version_prompt is not None:
            int(version_prompt)
    except ValueError:
        return 'Error: se ha ingresado un valor no numero como version'

    if not os.path.isdir("prompt_examples"):
        os.mkdir("prompt_examples")

    if os.path.isfile(ruta):
        lista = cargar_lista_prompt()
        total_prompts = lista.shape[0]
        if version_prompt is None:
            version_prompt = total_prompts + 1
            datos = {
                'version': [f'version {version_prompt}'],
                'prompt': [prompt]
            }
            datos = pd.DataFrame(datos)
            lista = pd.concat([lista, datos], ignore_index=True)
        elif 0 < version_prompt <= total_prompts:
            lista.at[version_prompt - 1, 'prompt'] = prompt
        else:
            return 'Error: numero de version no valido'

    else:
        datos = {
            'version': ['version 1'],
            'prompt': [prompt]
        }
        lista = pd.DataFrame(datos)

    lista.to_excel(ruta)


def cargar_lista_prompt():
    nombre_archivo = "versiones_prompt"
    ruta = f"prompt_examples/{nombre_archivo}.xlsx"
    try:
        lista = pd.read_excel(ruta, index_col=[0])
    except FileNotFoundError:
        print("No existe lista")
        return False
    return lista
