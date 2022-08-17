import os
import pandas as pd


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
