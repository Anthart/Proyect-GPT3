import pandas as pd
from tokens import calcular_total_pagar
from func import palabras_complejas
from func import load_data
from func import promedio_valor_escala
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="Carga datos si existio algun error en el codigo", action="store_true")
    args = parser.parse_args()

    minimo = 0
    maximo = 60
    file_corpus_train = "corpus/lcp_single_train_arreglo_escala.xlsx"

    if args.load:
        load, minimo, maximo = load_data()
    else:
        load = None

    prompt = ("I'm reading fragments from some source such as: bible, biomed and europarl, and some words are not easy "
              "to understand. I'm classifying these words into \"very easy\", \"easy\", \"neutral\", "
              "\"difficult\" and \"very difficult\". The sentence is \"neutral\" when it is neither "
              "\"very easy\", nor \"easy\", nor \"difficult\", nor \"very difficult\".\n"
              "Several examples are:\n"
              "The following fragment comes from the \"bible\" and after reading the fragment "
              "\" At the evening offering I arose up from my humiliation, even with my garment and my robe "
              "torn; and I fell on my knees, and spread out my hands to Yahweh my God; \". I find that word "
              "\"garment\" is easy\n###\n"
              "The following fragment comes from the \"biomed\" and after reading the fragment "
              "\" Expression of pendrinm RNA in the inner ear has been found in several places including the "
              "cochlea, the vestibular labyrinth and the endolymphatic sac [8]. \" I find that word "
              "\"Expression\" is neutral\n###\n"
              "The following fragment comes from the \"europarl\" and after reading the fragment "
              "\" The struggle against global Islamic terrorism is an asymmetric war with unforeseen results "
              "and unprecedented consequences, and new jurisprudence internationally is now needed to address "
              "this. \" I find that word \"jurisprudence\" is difficult\n###\n"
              "The following fragment comes from the @recurso and after reading the fragment @oracion I find that word "
              "@aEvaluar is")

    # print(prompt)

    df = pd.read_excel(file_corpus_train)
    df.loc[df["token"].isnull(), "token"] = "null"
    datos = df.loc[minimo:maximo, ["source", "sentence", "token", "complexity", "escala"]]

    calcular_total_pagar(datos, prompt, 10, True)

    if load is None:
        resultado = palabras_complejas(datos, prompt, promedio_valor_escala(file_corpus_train))
    else:
        resultado = palabras_complejas(datos, prompt, promedio_valor_escala(file_corpus_train), load)

    resultado.to_excel('resultados/resultadoPrueba.xlsx')

