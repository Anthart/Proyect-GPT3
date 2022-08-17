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
    maximo = 30
    file_corpus_train = "corpus/lcp_single_train_arreglo_escala.xlsx"

    if args.load:
        load, minimo, maximo = load_data()
    else:
        load = None

    prompt = ("I'm reading fragments from some source, and some words are not easy to understand. "
              "I'm classifying these words into \"very easy\", \"easy\", "
              "\"neutral\", \"difficult\" and \"very difficult\".\nAfter reading the fragment \" However, "
              "no defects in axon pathfinding along the monosynaptic reflex arc or in muscle "
              "spindle differentiation have been noted in PV KO mice, which develop normally and "
              "show no apparent changes in their behavior or physical activity (Schwaller et al. 1999)."
              " \". I find that word \"spindle\" is neutral\n\n###\n\n"
              "After reading the fragment \" I will sprinkle clean water on you, "
              "and you shall be clean: from all your filthiness, and from all your idols, will I cleanse you. \"."
              "I find that word \"filthiness\" is easy\n\n###\n\n"
              "After reading the fragment \" Moreover, acute dosing does not recapitulate the "
              "marked learning deficits produced in rodents [15,16] by chronic exposure to dopamine D2R "
              "antagonists [6,7] \" . I find that word \"antagonists\" is difficult\n\n###\n\n"
              "After reading the fragment \" Thrombus formation on fissured atherosclerotic "
              "plaques is the precipitating event in the transition from a stable or subclinical atherosclerotic"
              "disease and leads to acute myocardial infarction, ischemic stroke or peripheral arterial occlusion. "
              "\". I find that word \"Thrombus\" is very difficult\n\n###\n\n"
              "After reading the fragment @oracion I find that word @aEvaluar is")

    # print(prompt)

    df = pd.read_excel(file_corpus_train)
    df.loc[df["token"].isnull(), "token"] = "null"
    datos = df.loc[minimo:maximo, ["source", "sentence", "token", "complexity", "escala"]]

    # calcular_total_pagar(datos, prompt, 10, True)

    if load is None:
        resultado = palabras_complejas(datos, prompt, promedio_valor_escala(file_corpus_train))
    else:
        resultado = palabras_complejas(datos, prompt, promedio_valor_escala(file_corpus_train), load)

    resultado.to_excel('resultados/resultadoPrueba.xlsx')

