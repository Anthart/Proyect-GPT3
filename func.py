import openai
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys

lista_escalas = ['very easy', 'easy', 'neutral', 'difficult', 'very difficult']
plantilla = "{:^5} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20}"


def imprimir_fila(cuenta, indice, dframe, respuesta_gpt3, rango, complejidad_gpt3,
                  complejidad, complejidad_escala, comparacion):
    token = dframe["token"][indice]

    print(plantilla.format(cuenta, token, respuesta_gpt3, rango,
                           complejidad_gpt3, complejidad, complejidad_escala,
                           comparacion))


def asig_valor(valor):
    escala = ""

    if valor == 0:
        escala = "very easy"
    elif 0 < valor <= 0.25:
        escala = "easy"
    elif 0.25 < valor <= 0.50:
        escala = "neutral"
    elif 0.50 < valor <= 0.75:
        escala = "difficult"
    elif 0.75 < valor <= 1:
        escala = "very difficult"

    return escala


def asig_medio(valor_escala):
    valor_medio = 0

    if valor_escala == lista_escalas[0]:
        valor_medio = 0
    elif valor_escala == lista_escalas[1]:
        valor_medio = (0.01 + 0.25) / 2
    elif valor_escala == lista_escalas[2]:
        valor_medio = (0.26 + 0.50) / 2
    elif valor_escala == lista_escalas[3]:
        valor_medio = (0.51 + 0.75) / 2
    elif valor_escala == lista_escalas[4]:
        valor_medio = (0.76 + 1) / 2

    return valor_medio


def asig_rango(escala):
    rango = ""

    if escala == "very easy":
        rango = "0"
    if escala == "easy":
        rango = "0.01 - 0.25"
    if escala == "neutral":
        rango = "0.26 - 0.50"
    if escala == "difficult":
        rango = "0.51 - 0.75"
    if escala == "very difficult":
        rango = "0.76 - 1"

    return rango


def promedio_valor_escala(dframe):
    diccionario = {}

    for valor in lista_escalas:
        aux = dframe.loc[dframe["escala"] == valor]
        calculo = aux["complexity"].mean()
        diccionario[valor] = calculo

    return diccionario


def filtro(respuesta_gpt3):
    resultado = ""

    for valor_escala in lista_escalas:
        n_palabras = len(valor_escala.split())
        if n_palabras == 2 and respuesta_gpt3.count(valor_escala) >= 1:
            return valor_escala
        elif respuesta_gpt3.count(valor_escala) >= 1:
            resultado = valor_escala

    return resultado


def evaluar(orden):
    openai.api_key = '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'
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=orden,
        temperature=0,
        max_tokens=10,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )
    return response.choices[0].text


def palabras_complejas(dframe, orden, dic_escalas):
    resultado = dframe
    resultado["Respuesta GPT3"] = None
    resultado["Rango GPT3"] = None
    resultado["Complejidad GPT3"] = 0.0
    resultado["comparacion"] = None

    print(plantilla.format("N", "Token", "Respuesta GPT3", "Rango GPT3", "Complejidad GPT3",
                           "Complejidad compLex", "Rango compLex", "Comparacion") + "\n")

    cuenta = 0
    for indice in dframe.index:
        temp = orden
        temp = temp.replace("@recurso", "\"" + dframe["source"][indice] + "\"")
        temp = temp.replace("@oracion", "\"" + dframe["sentence"][indice] + "\"")
        temp = temp.replace("@aEvaluar", "\"" + dframe["token"][indice] + "\"")

        try:
            respuesta_gpt3 = evaluar(temp)
        except openai.error.OpenAIError:
            resultado.to_csv("resultados/resultados_tem.csv")
            sys.exit("Error openai")

        respuesta_gpt3 = filtro(respuesta_gpt3)
        rango = asig_rango(respuesta_gpt3)
        complejidad_gpt3 = dic_escalas[respuesta_gpt3]
        complejidad = dframe["complexity"][indice]
        escala_complex = dframe["escala"][indice]

        resultado.at[indice, "Respuesta GPT3"] = respuesta_gpt3
        resultado.at[indice, "Rango GPT3"] = rango
        resultado.at[indice, "Complejidad GPT3"] = complejidad_gpt3

        if respuesta_gpt3 == escala_complex:
            comparacion = "Si"
        else:
            comparacion = "No"

        resultado.at[indice, "comparacion"] = comparacion

        imprimir_fila(cuenta, indice, dframe, respuesta_gpt3, rango, complejidad_gpt3,
                      complejidad, escala_complex, comparacion)

        cuenta = cuenta + 1

    true = resultado.loc[:, "complexity"]
    predicted = resultado.loc[:, "Complejidad GPT3"]

    mae = round(mean_absolute_error(true, predicted), 4)
    mse = round(mean_squared_error(true, predicted), 4)
    rmse = round(mean_squared_error(true, predicted, squared=False), 4)
    r2 = round(r2_score(true, predicted), 4)
    pearson = round(true.corr(predicted, method='pearson'), 4)
    spearman = round(true.corr(predicted, method='spearman'), 4)

    print("MAE: " + str(mae))
    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse))
    print("R2: " + str(r2))
    print("Pearson: " + str(pearson))
    print("Spearman: " + str(spearman))
    print("\n")

    resultado = resultado[["sentence", "token", "Respuesta GPT3", "Rango GPT3", "Complejidad GPT3",
                           "complexity", "escala", "comparacion"]]

    resultado["MAE"] = mae
    resultado["MSE"] = mse
    resultado["RMSE"] = rmse
    resultado["R2"] = r2
    resultado["Pearson"] = pearson
    resultado["Sperman"] = spearman

    return resultado
