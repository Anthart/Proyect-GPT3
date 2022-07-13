import openai
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

escala = ['very easy', 'easy', 'neutral', 'difficult', 'very difficult']
titulos = "N\tToken\t\tRespuesta GPT\tRango\t\tComplejidad - compLex"


def imprimirFila(cuenta, indice, dframe, respuestaGPT3, rango, complejidad):
    token = dframe["token"][indice]
    if len(token) >= 8:
        token = token + "\t"
    else:
        token = token + "\t\t"

    respuestaGPT3 = str(round(respuestaGPT3, 3))
    if len(respuestaGPT3) >= 8:
        respuestaGPT3 = respuestaGPT3 + "\t"
    else:
        respuestaGPT3 = respuestaGPT3 + "\t\t"

    if len(rango) >= 8:
        rango = rango + "\t"
    else:
        rango = rango + "\t\t"

    complejidad = str(round(complejidad, 3))
    if len(complejidad) >= 8:
        complejidad = complejidad + "\t\t\t"
    else:
        complejidad = complejidad + "\t\t\t\t"

    print(str(cuenta) + "\t" + token + respuestaGPT3 + rango + complejidad)


def asigValor(valor):
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

def asigMedio(valor_escala):
    valor_medio = 0

    if valor_escala == escala[0]:
        valor_medio = 0
    elif valor_escala == escala[1]:
        valor_medio = (0.01 + 0.25) / 2
    elif valor_escala == escala[2]:
        valor_medio = (0.26 + 0.50) / 2
    elif valor_escala == escala[3]:
        valor_medio = (0.51 + 0.75) / 2
    elif valor_escala == escala[4]:
        valor_medio = (0.76 + 1) / 2

    return valor_medio

def asigRango(escala):

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


def filtro(respuestaGPT3):
    resultado = ''

    for valorEscala in escala:
        nPalabras = len(valorEscala.split())
        if nPalabras == 2 and respuestaGPT3.count(valorEscala) >= 1:
            return valorEscala
        elif respuestaGPT3.count(valorEscala) >= 1:
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

    print(titulos)

    cuenta = 0
    for indice in dframe.index:
        temp = orden
        temp = temp.replace("@recurso", "\"" + dframe["source"][indice] + "\"")
        temp = temp.replace("@oracion", "\"" + dframe["sentence"][indice] + "\"")
        temp = temp.replace("@aEvaluar", "\"" + dframe["token"][indice] + "\"")
        respuestaGPT3 = evaluar(temp)
        respuestaGPT3 = filtro(respuestaGPT3)
        rango = asigRango(respuestaGPT3)
        respuestaGPT3 = asigMedio(respuestaGPT3)

        complejidad = dframe["complexity"][indice]

        resultado.at[indice, "Respuesta GPT3"] = respuestaGPT3
        resultado.at[indice, "Rango"] = rango

        imprimirFila(cuenta, indice, dframe, respuestaGPT3, rango, complejidad)

        cuenta = cuenta + 1

    true = resultado.loc[:, "complexity"]
    predicted = resultado.loc[:, "Respuesta GPT3"]

    print("MAE: " + str(round(mean_absolute_error(true, predicted), 4)))
    print("MSE: " + str(round(mean_squared_error(true, predicted), 4)))
    print("RMSE: " + str(round(mean_squared_error(true, predicted, squared=False), 4)))
    print("R2: " + str(round(r2_score(true, predicted), 4)))
    return resultado
