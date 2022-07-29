import pandas as pd
from tokens import calcular_total_pagar
from func import palabras_complejas
from func import promedio_valor_escala

if __name__ == "__main__":
    prompt = ("I'm reading fragments from some source, and some words are not easy to understand. "
              "I'm classifying these words into \"very easy\", \"easy\", "
              "\"neutral\", \"difficult\" and \"very difficult\".\n"
              "Text: I will sprinkle clean water on you, and you shall be clean: from all your filthiness, "
              "and from all your idols, will I cleanse you.\n"
              "Word: filthiness\n"
              "Class: easy\n\n###\n\n"
              "Text: However, no defects in axon pathfinding along the monosynaptic reflex arc or in muscle "
              "spindle differentiation have been noted in PV KO mice, which develop normally and "
              "show no apparent changes in their behavior or physical activity (Schwaller et al. 1999).\n"
              "Word: spindle\n"
              "Class: neutral\n\n###\n\n"
              "Text: Thrombus formation on fissured atherosclerotic plaques is the precipitating event in the "
              "transition from a stable or subclinical atherosclerotic disease and leads to acute myocardial "
              "infarction, schemic stroke or peripheral arterial occlusion.\n"
              "Word: Thrombus\n"
              "Class: very difficult\n\n###\n\n"
              "Text: @oracion\n"
              "Word: @aEvaluar\n"
              "Class: ")
    # print(prompt)

    df = pd.read_excel("corpus/lcp_single_train_arreglo_escala.xlsx")

    df.loc[df["token"].isnull(), "token"] = "null"

    datos = df.loc[0:30, ["source", "sentence", "token", "complexity", "escala"]]

    # calcular_total_pagar(datos, prompt, 5, True)

    resultado = palabras_complejas(datos, prompt, promedio_valor_escala(df))

    # # # resultado.to_excel('resultados/resultadoPrueba.xlsx')
