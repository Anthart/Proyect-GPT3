import openai

escala = ['very easy', 'easy', 'neutral', 'difficult', 'very difficult']
titulos = "N\tToken\t\tRespuesta GPT\t\tRango\t\tComplejidad - compLex\t\tCoincidencia\n"


def imprimirFila(cuenta, indice, dframe, respuestaGPT3, rango, complejidad, comparacion):
    token = dframe["token"][indice]
    if len(token) >= 8:
        token = token + "\t"
    else:
        token = token + "\t\t"

    if len(respuestaGPT3) >= 8:
        respuestaGPT3 = respuestaGPT3 + "\t\t"
    else:
        respuestaGPT3 = respuestaGPT3 + "\t\t\t"

    if len(rango) >= 8:
        rango = rango + "\t"
    else:
        rango = rango + "\t\t"

    if len(complejidad) >= 8:
        complejidad = complejidad + "\t\t\t"
    else:
        complejidad = complejidad + "\t\t\t\t"

    print(str(cuenta) + "\t" + token + respuestaGPT3 + rango + complejidad + comparacion)


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

def asigRango(escala):

    rango = ""

    if escala == "very easy":
        rango = "0"
    if escala == "easy":
        rango = "0.1 - 0.25"
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
    resultado = dframe.loc[:, ["source", "sentence", "token"]]
    resultado["Escala GPT3"] = None
    resultado["Rango"] = None
    resultado["Escala corpus"] = None
    resultado["Coincidencia"] = None

    print(titulos)

    cuenta = 0
    nCoincidencia = 0
    for indice in dframe.index:
        temp = orden
        temp = temp.replace("@recurso","\"" + dframe["source"][indice] + "\"")
        temp = temp.replace("@oracion", "\"" + dframe["sentence"][indice] + "\"")
        temp = temp.replace("@aEvaluar", "\"" + dframe["token"][indice] + "\"")
        respuestaGPT3 = evaluar(temp)
        respuestaGPT3 = filtro(respuestaGPT3)
        rango = asigRango(respuestaGPT3)

        complejidad = dframe["complexity"][indice]
        complejidad = asigValor(complejidad)
        if complejidad == respuestaGPT3:
            comparacion = "SI"
            nCoincidencia = nCoincidencia + 1
        else:
            comparacion = "NO"

        resultado.at[indice, "Escala GPT3"] = respuestaGPT3
        resultado.at[indice, "Rango"] = rango
        resultado.at[indice, "Escala corpus"] = complejidad
        resultado.at[indice, "Coincidencia"] = comparacion

        imprimirFila(cuenta, indice, dframe, respuestaGPT3, rango, complejidad, comparacion)

        cuenta = cuenta + 1
        # time.sleep(0.5)

    coincidencia = (nCoincidencia / dframe.shape[0]) * 100
    print("Porcentaje coincidencia: " + str(round(coincidencia, 2)) + " %")

    return resultado
