import openai
import sys
import json
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from functools import reduce
from proyect_modules.probs_response import *
from proyect_modules.file_operation import *
from proyect_modules.total_pagar import *


class Gpt3:

    STRAT_1 = 'strat1'
    STRAT_2 = 'strat2'
    STRAT_3 = 'strat3'

    def __init__(self, datos, prompt, key, load=False):
        self.__plantilla_resultados = "{:^5} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20}"
        self.__plantilla_porcentaje = "{:^5} {:^20} {:^20} {:^30} {:^30} {:^30} {:^30} {:^30}"
        self.__means = {}
        self.__datos = datos
        self.__prompt = prompt
        self.__key = key
        self.load = load
        self.__rango_escalas = {
            'very easy': (0, 0),
            'easy': (0, 0.25),
            'neutral': (0.25, 0.50),
            'difficult': (0.50, 0.75),
            'very difficult': (0.75, 1)
        }

    def __prompt_format(self, source, sentence, token):
        prompt = self.__prompt
        prompt = prompt.replace("@recurso", f"\"{source}\"")
        prompt = prompt.replace("@oracion", f"\"{sentence}\"")
        prompt = prompt.replace("@aEvaluar", f"\"{token}\"")
        # prompt = prompt.replace("@aEvaluar", "\"" + self.__datos["token"][indice] + "\"")
        return prompt

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

    def __asig_etiqueta(self, valor):
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

    def __strat_1(self, valor_escala):
        return reduce(lambda x, y: (x + y) / 2, self.__rango_escalas.get(valor_escala))

    def __strat_2(self, name_file):
        diccionario = {}
        try:
            tf = open("../promedio.json", "r")
            diccionario = json.load(tf)
            no_file = False
        except FileNotFoundError:
            no_file = True

        if no_file:
            dframe = pd.read_excel(name_file)
            for valor in list(self.__rango_escalas.keys()):
                aux = dframe.loc[dframe["escala"] == valor]
                calculo = aux["complexity"].mean()
                diccionario[valor] = calculo
            tf = open("../promedio.json", "w")
            json.dump(diccionario, tf)
            tf.close()

        self.__means = diccionario

    def __strat_3(self, respuesta_gpt3, prob_dicc):
        valor_gpt3 = 0
        dicc_puntos = {"very easy": 0, "easy": 0.25, "neutral": 0.5, "difficult": 0.75, "very difficult": 1}
        list_keys = list(dicc_puntos.keys())
        total_entre_puntos = 1 / (len(list_keys) - 1)
        criterio = 1

        if respuesta_gpt3 not in list_keys:
            print("La respuesta de GPT-3 no se encuentran en la lista de probabilidades")
            return valor_gpt3

        if respuesta_gpt3 in ["difficult", "very difficult"]:
            criterio *= -1

        if respuesta_gpt3 not in ["neutral"]:
            punto_es = list_keys[list_keys.index(respuesta_gpt3) + 1 * criterio]
            punto_val = dicc_puntos.get(punto_es)
            valor_gpt3 = (punto_val - total_entre_puntos * prob_dicc.get(respuesta_gpt3) * criterio)
        else:
            valor_gpt3 = dicc_puntos.get(respuesta_gpt3)

        for key in prob_dicc.keys():
            if key in list_keys:
                if list_keys.index(respuesta_gpt3) < list_keys.index(key):
                    valor_gpt3 += (total_entre_puntos * prob_dicc.get(key)) * criterio
                elif list_keys.index(respuesta_gpt3) > list_keys.index(key):
                    valor_gpt3 -= (total_entre_puntos * prob_dicc.get(key)) * criterio

        return valor_gpt3

    def __strat_3_2(self, respuesta_gpt3, prod_dicc):
        complejidad_gpt3 = 0
        rango = self.__rango_escalas.get(respuesta_gpt3)
        list_keys = list(self.__rango_escalas.keys())

        if respuesta_gpt3 not in list_keys:
            print("La respuesta de GPT-3 no se encuentran en la lista de probabilidades")
            return complejidad_gpt3

        v_entre_rango = rango[1] - rango[0]

        if respuesta_gpt3 == 'easy':
            complejidad_gpt3 = rango[1] - (v_entre_rango * prod_dicc.get(respuesta_gpt3))

        if respuesta_gpt3 in ['difficult', 'very difficult']:
            complejidad_gpt3 = rango[0] + (v_entre_rango * prod_dicc.get(respuesta_gpt3))

        if respuesta_gpt3 == 'neutral':
            prod_dicc_keys = list(prod_dicc.keys())[1:]
            complejidad_gpt3 = (rango[1] + rango[0]) / 2
            for key in prod_dicc_keys:
                if key in list_keys:
                    if list_keys.index('neutral') > list_keys.index(key):
                        complejidad_gpt3 -= ((v_entre_rango / 2) * prod_dicc.get(respuesta_gpt3))
                    else:
                        complejidad_gpt3 += ((v_entre_rango / 2) * prod_dicc.get(respuesta_gpt3))

        return complejidad_gpt3

    def __filtro(self, respuesta_gpt3):
        resultado = ""

        for valor_escala in list(self.__rango_escalas.keys()):
            n_palabras = len(valor_escala.split())
            if n_palabras == 2 and respuesta_gpt3.count(valor_escala) >= 1:
                return valor_escala
            elif respuesta_gpt3.count(valor_escala) >= 1:
                resultado = valor_escala

        return resultado

    def __evaluar(self, orden):
        openai.api_key = self.__key
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=orden,
            temperature=0,
            max_tokens=5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=5,
            stop=["\n"]
        )
        respuesta = response.choices[0].text
        prob_tokens = response.choices[0].logprobs.top_logprobs
        return respuesta, prob_tokens

    def data_to_process(self):
        load_da = load_data_temp() if self.load else None

        if load_da is None:
            to_process = self.__datos
            to_process["Respuesta GPT3"] = None
            to_process["Rango GPT3"] = None
            to_process["Complejidad GPT3"] = 0.0
            to_process["comparacion"] = None
            for i in range(5):
                to_process[f"Porcentaje {i + 1}"] = ""
        elif load_da is not None:
            to_process = load_da
            frame = [to_process, self.__datos.loc[:]]
            to_process = pd.concat(frame)
            to_process = to_process.loc[:, ~to_process.columns.str.contains("Unnamed")]
        else:
            sys.exit("Error de ingreso de parametros")

        return to_process

    def process_data(self, indice, resultado):
        temp = self.__prompt_format(
            self.__datos["source"][indice],
            self.__datos["sentence"][indice],
            self.__datos["token"][indice]
        )

        try:
            respuesta_gpt3, prob_tokens = self.__evaluar(temp)
        except openai.error.RateLimitError as limit_rate_error:
            if indice - 1 != -1:
                temporal_storage(indice, self.__datos.tail(1).index[0], resultado.loc[0:indice - 1])
            sys.exit(str(limit_rate_error))
        except openai.error.OpenAIError as error_openai:
            if indice - 1 != -1:
                temporal_storage(indice, self.__datos.tail(1).index[0], resultado.loc[0:indice - 1])
            sys.exit(str(error_openai))

        try:
            respuesta_gpt3 = self.__filtro(respuesta_gpt3)
            cant_palabras = len(respuesta_gpt3.split())
            prob = logprobs_to_percent(prob_tokens[0: cant_palabras])
            if respuesta_gpt3 == "":
                raise KeyError
        except KeyError:
            temporal_storage(indice, self.__datos.tail(1).index[0], resultado.loc[0:indice - 1])
            sys.exit("No se encontro el resultado esperado"
                     " por GPT3")
        return temp, respuesta_gpt3, prob

    def process_all(self, file_path="", version=False, save_result=False, percent=False):

        if file_path != "":
            self.__strat_2(file_path)

        resultado = self.data_to_process()
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

            temp, respuesta_gpt3, prob = self.process_data(indice, resultado)

            tokens_prompt += len(tokenizer(temp)['input_ids'])
            peticiones += 1

            # complejidad_gpt3 = round(self.__means[respuesta_gpt3], 15)
            complejidad_gpt3 = self.__strat_3_2(respuesta_gpt3, prob[0])
            respuesta_gpt3 = self.__asig_etiqueta(complejidad_gpt3)
            rango = reduce(lambda x, y: f'{x + 0.01} - {y}', self.__rango_escalas.get(respuesta_gpt3))
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

            prob = logprobs_display(prob)

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
            guardar_metricas(resultado_metricas)

        if save_result:
            if version:
                resultado.to_excel(f'resultados/resultado_{version}.xlsx')
            else:
                resultado.to_excel(f'resultados/resultado_prueba.xlsx')
