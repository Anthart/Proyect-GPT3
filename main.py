import pandas as pd
from proyect_modules.gpt3 import Gpt3
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--percent", help="Muestra porcentaje de las respuestas", action="store_true")
    parser.add_argument("-l", "--load", help="Carga datos si existio algun error en el codigo", action="store_true")
    args = parser.parse_args()

    minimo = 0
    maximo = 5
    file_corpus_train = "corpus/lcp_single_train_arreglo_escala.xlsx"
    file_corpus_test = "corpus/test_depurado.xlsx"

    prompt = (
        "I'm reading fragments from some source such as: bible, biomed and europarl, "
        "and some words are not easy to understand. "
        "I'm classifying these words into \"very easy\", \"easy\", "
        "\"neutral\", \"difficult\" and \"very difficult\". The sentence is \"neutral\" "
        "when it is neither \"very easy\", nor \"easy\", nor \"difficult\", nor \"very difficult\"."
        "\nSeveral examples are: \" However, "
        "no defects in axon pathfinding along the monosynaptic reflex arc or in muscle "
        "spindle differentiation have been noted in PV KO mice, which develop normally and "
        "show no apparent changes in their behavior or physical activity (Schwaller et al. 1999)."
        " \". I find that word \"spindle\" is neutral\n\n###\n\n"
        "The following fragment comes from the \"bible\" and after reading the "
        "fragment \" I will sprinkle clean water on you, "
        "and you shall be clean: from all your filthiness, and from all your idols, will I cleanse you. \"."
        "I find that word \"filthiness\" is easy\n\n###\n\n"
        "The following fragment comes from the \"biomed\" and "
        "after reading the fragment \" Moreover, acute dosing does not recapitulate the "
        "marked learning deficits produced in rodents [15,16] by chronic exposure to dopamine D2R "
        "antagonists [6,7] \" . I find that word \"antagonists\" is difficult\n\n###\n\n"
        "The following fragment comes from the \"biomed\" and after reading the fragment "
        "\" Thrombus formation on fissured atherosclerotic "
        "plaques is the precipitating event in the transition from a stable or subclinical atherosclerotic"
        "disease and leads to acute myocardial infarction, ischemic stroke or peripheral arterial occlusion. "
        "\". I find that word \"Thrombus\" is very difficult\n\n###\n\n"
        "The following fragment comes from the \"bible\" and "
        "after reading the fragment \" Mount Sinai, all it, smoked, because Yahweh descended "
        "on it in fire; and its smoke ascended like the smoke of a furnace, and the whole mountain quaked "
        "greatly. \". I find that word \"fire\" is very easy\n\n###\n\n"
        "The following fragment comes from the @recurso and after reading the "
        "fragment @oracion I find that word @aEvaluar is")

    df = pd.read_excel(file_corpus_train)
    df.loc[df["token"].isnull(), "token"] = "null"
    datos = df.loc[minimo:maximo, ["id", "source", "sentence", "token", "complexity", "escala"]]

    gpt = Gpt3(datos, prompt, "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", load=args.load)
    # calcular_total_pagar(prompt, datos)
    gpt.process_all(file_path=file_corpus_train, save_result=True, percent=args.percent)



