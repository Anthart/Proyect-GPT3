import pandas as pd
from tokens import calcular_total_pagar
from func import palabras_complejas
from func import load_data
from func import promedio_valor_escala
from func import guardar_metricas
from prompt_file import cargar_lista_prompt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="Carga datos si existio algun error en el codigo", action="store_true")
    args = parser.parse_args()

    minimo = 0
    maximo = 1
    file_corpus_train = "corpus/lcp_single_train_arreglo_escala.xlsx"
    file_corpus_test = "corpus/test_depurado.xlsx"

    if args.load:
        load, minimo, maximo = load_data()
    else:
        load = None

    lista_prompt = cargar_lista_prompt()

    prompt = lista_prompt["prompt"][9]

    # print(prompt)

    df = pd.read_excel(file_corpus_test)
    df.loc[df["token"].isnull(), "token"] = "null"
    datos = df.loc[minimo:, ["id", "source", "sentence", "token", "complexity", "escala"]]

    # calcular_total_pagar(datos, prompt, 10, True)

    if load is None:
        palabras_complejas(datos, prompt, promedio_valor_escala(file_corpus_train), save_result=True)
    else:
        palabras_complejas(datos, prompt, promedio_valor_escala(file_corpus_train), save_result=True, load=load)


