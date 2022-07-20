import pandas as pd
from func import palabras_complejas
from func import filtro
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # prompt = """ I'm reading fragments from the Bible, and some words are not easy to understand.
    # I'm classifying these words into \"very easy\", \"easy\", \"neutral\", \"difficult\" and \"very difficult\".\n
    #
    # Text: However, no defects in axon pathfinding along the monosynaptic reflex arc or in muscle
    # spindle differentiation have been noted in PV KO mice, which develop normally and show no apparent changes in their
    # behavior or physical activity (Schwaller et al. 1999).\n
    #
    # Response: \"spindle\" is neutral\n
    #
    # \n ### \n
    #
    # Text: I will sprinkle clean water on you, and you shall be clean: from all your filthiness,
    # and from all your idols, will I cleanse you.\n
    #
    # Response: \"filthiness\" is easy\n
    #
    # \n ### \n
    #
    # Text: @oracion\n
    #
    # Response: @aEvaluar is"""
    #
    # df = pd.read_excel("corpus/labeled-lcp_single_test.xlsx")
    #
    # datos = df.loc[0:10, ["source", "sentence", "token", "complexity"]]
    #
    # resultado = palabras_complejas(datos, prompt)
    #
    # resultado.to_excel('corpus/resultadoPrueba.xlsx')

    df = pd.read_excel("corpus/lcp_single_train_arreglo.xlsx")

    df.loc[df["token"].isnull(), "token"] = "null"

    df.to_excel("corpus/prueba.xlsx")

    print(df)
