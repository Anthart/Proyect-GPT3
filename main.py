import os

import pandas as pd
from proyect_modules.gpt3 import Gpt3
import argparse

from proyect_modules.total_pagar import calcular_total_pagar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--percent", help="Muestra porcentaje de las respuestas", action="store_true")
    parser.add_argument("-l", "--load", help="Carga datos si existio algun error en el codigo", action="store_true")
    args = parser.parse_args()

    minimo = 0
    maximo = 20
    file_corpus_train = "corpus/lcp_single_train_arreglo_escala.xlsx"
    file_corpus_test = "corpus/test_depurado.xlsx"

    prompt = ("I'm reading fragments from some source, and some words are not easy to understand. "
              "I'm classifying these words into \"very easy\", \"easy\", "
              "\"neutral\", \"difficult\" and \"very difficult\".\n Several examples are: \nAfter reading the fragment \" However, "
              "no defects in axon pathfinding along the monosynaptic reflex arc or in muscle "
              "spindle differentiation have been noted in PV KO mice, which develop normally and "
              "show no apparent changes in their behavior or physical activity (Schwaller et al. 1999)."
              " \". I classified the word \"spindle\" as \"neutral\"\n\n###\n\n"
              "After reading the fragment \" I will sprinkle clean water on you, "
              "and you shall be clean: from all your filthiness, and from all your idols, will I cleanse you. \"."
              "I classified the word \"filthiness\" as \"easy\"\n\n###\n\n"
              "The following fragment comes from the \"bible\" and its fragment is "
              "\" David said to Saul, \"Your servant was keeping his father's sheep; and when a "
              "lion or a bear came, and took a lamb out of the flock, \". I classified the word \"lion\" as "
              "\"very easy\".\n###\n"
              "The following fragment comes from the \"biomed\" and its fragment is \" "
              "In this report, we generated targeted mutations of Pygo1 and Pygo2 to determine their "
              "functions, with a particular interest in the contributions of these genes to canonical Wnt "
              "signaling during kidney development. \". I classified the word \"kidney\" as "
              "\"easy\".\n###\n"
              "The following fragment comes from the \"biomed\" and its fragment is \" The model "
              "of intermediate MSUD was created by partial transgenic rescue of the E2 gene knockout. \". I classified the word \"rescue\" as \"neutral\".\n###\n"
              "The following fragment comes from the \"europarl\" and its fragment is \" VIS, for its part, "
              "has entered an important phase. \". I classified the word \"VIS\" as "
              "\"difficult\".\n###\n"
              "The following fragment comes from the \"bible\" and this fragment is \" "
              "He overlaid the cherubim with gold. \" I classified the word \"cherubim\" as \"very difficult\".\n###\n"
              "After reading the fragment \" Moreover, acute dosing does not recapitulate the "
              "marked learning deficits produced in rodents [15,16] by chronic exposure to dopamine D2R "
              "antagonists [6,7] \" . I classified the word \"antagonists\" as \"difficult\"\n\n###\n\n"
              "After reading the fragment \" Thrombus formation on fissured atherosclerotic "
              "plaques is the precipitating event in the transition from a stable or subclinical atherosclerotic"
              "disease and leads to acute myocardial infarction, ischemic stroke or peripheral arterial occlusion. "
              "\". I classified the word \"Thrombus\" as \"very difficult\"\n\n###\n\n"
              "After reading the fragment @oracion. I classified @aEvaluar as")

    df = pd.read_excel(file_corpus_test)
    df.loc[df["token"].isnull(), "token"] = "null"
    datos = df.loc[minimo:, ["id", "source", "sentence", "token", "complexity", "escala"]]
    key = "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"
    gpt = Gpt3(datos, prompt, key, load=args.load)
    # calcular_total_pagar(prompt, datos, cost=0.03)
    gpt.process_all(file_path=file_corpus_train, save_result=True, percent=args.percent)



