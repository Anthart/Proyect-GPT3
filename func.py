import openai
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

lista_escalas = ['very easy', 'easy', 'neutral', 'difficult', 'very difficult']
titulos = "N\tToken\t\tRespuesta GPT\tRango\t\tComplejidad - compLex"
plantilla = "{:^5} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20}"


def imprimir_fila(cuenta, indice, dframe, respuesta_gpt3, rango, valor_medio_gpt3,
                  complejidad, complejidad_escala):

    token = dframe["token"][indice]

    if respuesta_gpt3 == complejidad_escala:
        comparacion = "Si"
    else:
        comparacion = "No"

    print(plantilla.format(cuenta, token, respuesta_gpt3, rango,
                           valor_medio_gpt3, complejidad, complejidad_escala,
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


def filtro(respuesta_gpt3):
    resultado = ''

    for valorEscala in lista_escalas:
        nPalabras = len(valorEscala.split())
        if nPalabras == 2 and respuesta_gpt3.count(valorEscala) >= 1:
            return valorEscala
        elif respuesta_gpt3.count(valorEscala) >= 1:
            resultado = valorEscala

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
        presence_penalty=0
    )
    return response.choices[0].text


def palabras_complejas(dframe, orden):
    resultado = dframe
    resultado["Respuesta GPT3"] = 0.0
    resultado["Rango"] = None
    resultado["Valor medio"] = 0.0
    resultado["compLex (complejidad a escala)"] = None

    print(plantilla.format("N", "Token", "Respuesta GPT", "Rango GPT", "Valor medio",
                           "Complejidad compLex", "Rango compLex", "Comparacion") + "\n")

    cuenta = 0
    for indice in dframe.index:
        temp = orden
        temp = temp.replace("@recurso", "\"" + dframe["source"][indice] + "\"")
        temp = temp.replace("@oracion", "\"" + dframe["sentence"][indice] + "\"")
        temp = temp.replace("@aEvaluar", "\"" + dframe["token"][indice] + "\"")
        respuesta_gpt3 = evaluar(temp)

        respuesta_gpt3 = filtro(respuesta_gpt3)
        rango = asig_rango(respuesta_gpt3)
        valor_medio_gpt3 = asig_medio(respuesta_gpt3)
        complejidad = dframe["complexity"][indice]
        escala_complex = asig_valor(complejidad)

        resultado.at[indice, "Respuesta GPT3"] = respuesta_gpt3
        resultado.at[indice, "Rango"] = rango
        resultado.at[indice, "Valor medio"] = valor_medio_gpt3
        resultado.at[indice, "compLex (complejidad a escala)"] = escala_complex

        imprimir_fila(cuenta, indice, dframe, respuesta_gpt3, rango, valor_medio_gpt3,
                      complejidad, escala_complex)

        cuenta = cuenta + 1

    true = resultado.loc[:, "complexity"]
    predicted = resultado.loc[:, "Valor medio"]

    print("MAE: " + str(round(mean_absolute_error(true, predicted), 4)))
    print("MSE: " + str(round(mean_squared_error(true, predicted), 4)))
    print("RMSE: " + str(round(mean_squared_error(true, predicted, squared=False), 4)))
    print("R2: " + str(round(r2_score(true, predicted), 4)))
    print("Pearson: " + str(round(true.corr(predicted), 4)))
    print("spearman: " + str(round(true.corr(predicted, method='spearman'), 4)))
    print("\n")

    return resultado
