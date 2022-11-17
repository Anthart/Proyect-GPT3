from transformers import GPT2Tokenizer
import pandas as pd


def calcular_total_pagar(prompt, datos, cost=0.02, completion_length=10, to_file=False):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    corpus_token = pd.DataFrame()

    corpus_token["prompts"] = None
    corpus_token["tokens"] = None
    corpus_token["Max response token"] = None
    corpus_token["Total"] = None

    for indice in datos.index:
        temp = prompt
        temp = temp.replace("@recurso", datos["source"][indice])
        temp = temp.replace("@oracion", "\"" + datos["sentence"][indice] + "\"")
        temp = temp.replace("@aEvaluar", "\"" + datos["token"][indice] + "\"")
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
