import operator
import os
import openai
import sys
import json
import pickle
import time
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from transformers import GPT2Tokenizer


class Gpt3:

    def __init__(self, datos, prompt, key):
        self.__lista_escalas = ['very easy', 'easy', 'neutral', 'difficult', 'very difficult']
        self.__plantilla_resultados = "{:^5} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20}"
        self.__plantilla_porcentaje = "{:^5} {:^20} {:^20} {:^30} {:^30} {:^30} {:^30} {:^30}"
        self.__means = {}
        self.__datos = datos
        self.__prompt = prompt
        self.__key = key

    def __imprimir_fila(self, indice, respuesta_gpt3, rango, complejidad_gpt3,
                        complejidad, complejidad_escala, comparacion):
        token = self.__datos["token"][indice]

        print(self.__plantilla_resultados.format(indice, token, respuesta_gpt3, rango,
                                                 complejidad_gpt3, complejidad, complejidad_escala,
                                                 comparacion))

    def __imprimir_fila_porcent(self, indice, respuesta_gpt3, respuesta_complex, opciones):
        token = self.__datos["token"][indice]

        print(self.__plantilla_porcentaje.format(indice, token, respuesta_gpt3, respuesta_complex, opciones[0],
                                                 opciones[1], opciones[2], opciones[3]))

    def __asig_valor(self, valor):
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

    def __asig_medio(self, valor_escala):
        valor_medio = 0

        if valor_escala == self.__lista_escalas[0]:
            valor_medio = 0
        elif valor_escala == self.__lista_escalas[1]:
            valor_medio = (0.01 + 0.25) / 2
        elif valor_escala == self.__lista_escalas[2]:
            valor_medio = (0.26 + 0.50) / 2
        elif valor_escala == self.__lista_escalas[3]:
            valor_medio = (0.51 + 0.75) / 2
        elif valor_escala == self.__lista_escalas[4]:
            valor_medio = (0.76 + 1) / 2

        return valor_medio

    def __asig_rango(self, escala):
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

    def promedio_valor_escala(self, name_file):
        diccionario = {}

        try:
            tf = open("promedio.json", "r")
            diccionario = json.load(tf)
            no_file = False
        except FileNotFoundError:
            no_file = True

        if no_file:
            dframe = pd.read_excel(name_file)
            for valor in self.__lista_escalas:
                aux = dframe.loc[dframe["escala"] == valor]
                calculo = aux["complexity"].mean()
                diccionario[valor] = calculo
            tf = open("promedio.json", "w")
            json.dump(diccionario, tf)
            tf.close()

        self.__means = diccionario

    def __filtro(self, respuesta_gpt3):
        resultado = ""

        for valor_escala in self.__lista_escalas:
            n_palabras = len(valor_escala.split())
            if n_palabras == 2 and respuesta_gpt3.count(valor_escala) >= 1:
                return valor_escala
            elif respuesta_gpt3.count(valor_escala) >= 1:
                resultado = valor_escala

        return resultado

    def __load_data(self):
        with open("temp/datos_temp.pkl", "rb") as tf:
            dicc = pickle.load(tf)

        minimo = dicc["minimo"]
        maximo = dicc["maximo"]
        nombre_archivo = dicc["archivo"]

        df = pd.read_csv(f"temp/{nombre_archivo}")

        return df, minimo, maximo

    def __temporal_storage(self, minimo, maximo, data):
        try:
            os.mkdir('temp')
        except OSError:
            print("Correpto !!\tCarpeta temp exite")

        now = datetime.now()
        nombre_archivo = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.second}.csv"
        data.to_csv(f"temp/{nombre_archivo}")

        dicc = {"minimo": minimo, "maximo": maximo, "archivo": nombre_archivo}

        with open("temp/datos_temp.pkl", "wb") as tf:
            pickle.dump(dicc, tf)

    def __guardar_metricas(self, metricas):
        file = 'resultados_metricas.xlsx'
        folder = 'resultados'
        path = f'{folder}/{file}'

        if not os.path.isdir(folder):
            os.mkdir(folder)

        if os.path.isfile(path):
            pandas_metrics = pd.read_excel(path, index_col=[0])
            pandas_metrics = pd.concat([pandas_metrics, metricas], ignore_index=True)
        else:
            pandas_metrics = metricas

        pandas_metrics.to_excel(path)

    def __evaluar(self, orden):
        openai.api_key = self.__key
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=orden,
            temperature=0,
            max_tokens=5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=4,
            stop=["\n"]
        )
        respuesta = response.choices[0].text
        prob_tokens = response.choices[0].logprobs.top_logprobs
        return respuesta, prob_tokens

    def __ordenar_probs(self, dicc: dict[str, float]):
        new_dicc = {}
        dic_sort = sorted(dicc.items(), key=operator.itemgetter(1), reverse=True)
        for item in dic_sort:
            new_dicc[item[0]] = item[1]
        return new_dicc

    def __pre_data_prob(self, dicc: dict[str, float]) -> dict:
        new_dicc = {}
        r = 0
        for d in dicc:
            no_space = d.replace(" ", "")
            lista = list(dicc.keys())
            lista.remove(d)
            if no_space in lista:
                r = np.exp(dicc[no_space])
            val = np.exp(dicc[d])
            new_dicc[no_space] = round((val + r) * 100, 2)
            new_dicc = self.__ordenar_probs(new_dicc)
        return new_dicc

    def __logprobs_to_percent(self, prob: list[dict[str, float]]):
        new_prob = []
        for i in range(len(prob)):
            item = prob[i]
            self.__pre_data_prob(item)
            new_prob.append(self.__pre_data_prob(item))
        return new_prob

    def __logprobs_display(self, logprobs: list[dict[str, float]]) -> list:
        probs = self.__logprobs_to_percent(logprobs)
        lista = ["", "", "", "", ""]
        size = len(probs)
        count = 5
        for i in range(size):
            items = list(probs[i].items())
            text = "" if i == 0 else ","
            for j in range(count):
                if j <= len(items) - 1:
                    lista[j] = lista[j] + text + str(items[j][0]) + ":" + str(items[j][1]) + "%"
                else:
                    lista[j] = lista[j] + text + "None"
        return lista

    def __prob_for_label(self, label: str, logprobs: list[dict[str, float]]) -> float:
        prob = 0.0
        next_logprobs = logprobs[0]
        for s, logprob in next_logprobs.items():
            s = s.lower()
            if label.lower() == s:
                prob += np.exp(logprob)
            elif label.lower().startswith(s):
                rest_of_label = label[len(s):]
                remaining_logprobs = logprobs[1:]
                prob += logprob * self.__prob_for_label(
                    rest_of_label,
                    remaining_logprobs,
                )
        return prob

    def calcular_total_pagar(self, cost=0.02, completion_length=10, to_file=False):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        corpus_token = pd.DataFrame()

        corpus_token["prompts"] = None
        corpus_token["tokens"] = None
        corpus_token["Max response token"] = None
        corpus_token["Total"] = None

        for indice in self.__datos.index:
            temp = self.__prompt
            temp = temp.replace("@recurso", self.__datos["source"][indice])
            temp = temp.replace("@oracion", "\"" + self.__datos["sentence"][indice] + "\"")
            temp = temp.replace("@aEvaluar", "\"" + self.__datos["token"][indice] + "\"")
            tokens_prompt = len(tokenizer(temp)['input_ids'])
            corpus_token.at[indice, "prompts"] = temp
            corpus_token.at[indice, "tokens"] = tokens_prompt
            corpus_token.at[indice, "Max response token"] = completion_length
            corpus_token.at[indice, "Total"] = tokens_prompt + completion_length

        total_prompt_mas_res = corpus_token["tokens"].sum() + corpus_token["Max response token"].sum()
        total_prompt_mas_res *= (cost / 1000)

        print(f"Total a pagar: {round(total_prompt_mas_res, 2)} $")

        if to_file:
            corpus_token.to_excel("resultados/precios_gpt3.xlsx")

    @staticmethod
    def guardar_prompt(prompt, version_prompt=None):
        nombre_archivo = "versiones_prompt"
        ruta = f"prompt_examples/{nombre_archivo}.xlsx"

        try:
            if version_prompt is not None:
                int(version_prompt)
        except ValueError:
            return 'Error: se ha ingresado un valor no numero como version'

        if not os.path.isdir("prompt_examples"):
            os.mkdir("prompt_examples")

        if os.path.isfile(ruta):
            lista = Gpt3.cargar_lista_prompt()
            total_prompts = lista.shape[0]
            if version_prompt is None:
                version_prompt = total_prompts + 1
                datos = {
                    'version': [f'version {version_prompt}'],
                    'prompt': [prompt]
                }
                datos = pd.DataFrame(datos)
                lista = pd.concat([lista, datos], ignore_index=True)
            elif 0 < version_prompt <= total_prompts:
                lista.at[version_prompt - 1, 'prompt'] = prompt
            else:
                return 'Error: numero de version no valido'

        else:
            datos = {
                'version': ['version 1'],
                'prompt': [prompt]
            }
            lista = pd.DataFrame(datos)

        lista.to_excel(ruta)

    @staticmethod
    def cargar_lista_prompt():
        nombre_archivo = "versiones_prompt"
        ruta = f"prompt_examples/{nombre_archivo}.xlsx"
        try:
            lista = pd.read_excel(ruta, index_col=[0])
        except FileNotFoundError:
            print("No existe lista")
            return False
        return lista

    def process(self, version=False, save_result=False, load=False,
                percent=False):

        load_data = self.__load_data() if load else None

        if load_data is None:
            resultado = self.__datos
            resultado["Respuesta GPT3"] = None
            resultado["Rango GPT3"] = None
            resultado["Complejidad GPT3"] = 0.0
            resultado["comparacion"] = None
            for i in range(5):
                resultado[f"Porcentaje {i + 1}"] = ""
        elif load_data is not None:
            resultado = load_data
            frame = [resultado, self.__datos.loc[:]]
            resultado = pd.concat(frame)
            resultado = resultado.loc[:, ~resultado.columns.str.contains("Unnamed")]
        else:
            sys.exit("Error de ingreso de parametros")

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        last = time.time()
        tokens_prompt = 0
        peticiones = 0

        if percent:
            print(self.__plantilla_porcentaje.format("N", "Token", "Respuesta GPT3", "Respuesta CompLex", "Opcion 1",
                                                     "Opcion 2", "Opcion 3", "Opcion 4", "Opcion 5"))
        else:
            print(self.__plantilla_resultados.format("N", "Token", "Respuesta GPT3", "Rango GPT3", "Complejidad GPT3",
                                                     "Complejidad compLex", "Rango compLex", "Comparacion") + "\n")

        for indice in self.__datos.index:
            temp = self.__prompt
            temp = temp.replace("@recurso", "\"" + self.__datos["source"][indice] + "\"")
            temp = temp.replace("@oracion", "\"" + self.__datos["sentence"][indice] + "\"")
            temp = temp.replace("@aEvaluar", "\"" + self.__datos["token"][indice] + "\"")

            try:
                respuesta_gpt3, prob_tokens = self.__evaluar(temp)
            except openai.error.RateLimitError as limit_rate_error:
                if indice - 1 != -1:
                    self.__temporal_storage(indice, self.__datos.tail(1).index[0], resultado.loc[0:indice - 1])
                sys.exit(str(limit_rate_error))
            except openai.error.OpenAIError as error_openai:
                if indice - 1 != -1:
                    self.__temporal_storage(indice, self.__datos.tail(1).index[0], resultado.loc[0:indice - 1])
                sys.exit(str(error_openai))

            try:
                respuesta_gpt3 = self.__filtro(respuesta_gpt3)
                cant_palabras = len(respuesta_gpt3.split())
                prob = self.__logprobs_display(prob_tokens[0: cant_palabras])
                if respuesta_gpt3 == "":
                    raise KeyError
            except KeyError:
                self.__temporal_storage(indice, self.__datos.tail(1).index[0], resultado.loc[0:indice - 1])
                sys.exit("No se encontro el resultado esperado"
                         " por GPT3")

            tokens_prompt += len(tokenizer(temp)['input_ids'])
            peticiones += 1

            rango = self.__asig_rango(respuesta_gpt3)
            complejidad_gpt3 = round(self.__means[respuesta_gpt3], 15)
            complejidad = self.__datos["complexity"][indice]
            escala_complex = self.__datos["escala"][indice]

            resultado.at[indice, "Respuesta GPT3"] = respuesta_gpt3
            resultado.at[indice, "Rango GPT3"] = rango
            resultado.at[indice, "Complejidad GPT3"] = complejidad_gpt3

            if respuesta_gpt3 == escala_complex:
                comparacion = "Si"
            else:
                comparacion = "No"

            resultado.at[indice, "comparacion"] = comparacion

            for i in range(len(prob)):
                resultado.at[indice, f"Porcentaje {i + 1}"] = prob[i]

            if percent:
                self.__imprimir_fila_porcent(indice, respuesta_gpt3, escala_complex, prob)
            else:
                self.__imprimir_fila(indice, respuesta_gpt3, rango, complejidad_gpt3,
                                     complejidad, escala_complex, comparacion)

            # ****************************** Control de Peticiones ***********************************
            actual = time.time() - last

            if actual >= 60:
                tokens_prompt = 0
                peticiones = 0
                last = time.time()

            if tokens_prompt >= 150000 or peticiones >= 55:
                # if peticiones >= 55:
                seconds_to_wait = 60 - actual
                tokens_prompt = 0
                peticiones = 0
                time.sleep(seconds_to_wait)
                last = time.time()
            # ************************* Cierre de control de Peticiones ******************************

        true = resultado.loc[:, "complexity"]
        predicted = resultado.loc[:, "Complejidad GPT3"]

        metrics = {
            "MAE": round(mean_absolute_error(true, predicted), 4),
            "MSE": round(mean_squared_error(true, predicted), 4),
            "RMSE": round(mean_squared_error(true, predicted, squared=False), 4),
            "R2": round(r2_score(true, predicted), 4),
            "Pearson": round(true.corr(predicted, method='pearson'), 4),
            "Spearman": round(true.corr(predicted, method='spearman'), 4)
        }

        for m in metrics:
            resultado[m] = metrics[m]
            print(f"{m}: {metrics[m]}")
        print("\n")

        resultado = resultado[["id", "sentence", "token", "Respuesta GPT3", "Rango GPT3", "Complejidad GPT3",
                               "complexity", "escala", "comparacion", "MAE", "MSE", "RMSE", "R2", "Pearson", "Spearman",
                               "Porcentaje 1", "Porcentaje 2", "Porcentaje 3", "Porcentaje 4", "Porcentaje 5"]]

        if version:
            resultado_metricas = {"Version": [version]}
            resultado_metricas.update(metrics)
            resultado_metricas = pd.DataFrame(resultado_metricas)
            self.__guardar_metricas(resultado_metricas)

        if save_result:
            if version:
                resultado.to_excel(f'resultados/resultado_{version}.xlsx')
            else:
                resultado.to_excel('resultados/resultado_version 12.xlsx')

        return resultado
