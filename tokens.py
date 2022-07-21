from transformers import GPT2Tokenizer
import pandas as pd


def calcular_total_pagar(df, prompt, limit_max, to_file):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    df = df.loc[0:]

    corpus_token = pd.DataFrame()

    count = 0

    corpus_token["prompts"] = None
    corpus_token["tokens"] = None
    corpus_token["Max response token"] = None
    corpus_token["Total"] = None

    for indice in df.index:
        temp = prompt
        temp = temp.replace("@recurso", "\"" + df["source"][indice] + "\"")
        temp = temp.replace("@oracion", "\"" + df["sentence"][indice] + "\"")
        temp = temp.replace("@aEvaluar", "\"" + df["token"][indice] + "\"")
        tokens_prompt = len(tokenizer(temp)['input_ids'])
        corpus_token.at[indice, "prompts"] = temp
        corpus_token.at[indice, "tokens"] = tokens_prompt
        corpus_token.at[indice, "Max response token"] = limit_max
        corpus_token.at[indice, "Total"] = tokens_prompt + limit_max
        count += 1
        print(f"{count}. {tokens_prompt}")

    total_prompt_mas_res = corpus_token["tokens"].sum() + corpus_token["Max response token"].sum()
    total_prompt_mas_res *= (0.06 / 1000)

    print(f"Total a pagar: {round(total_prompt_mas_res, 2)} $")

    if to_file:
        corpus_token.to_excel("resultados/precios_gpt3.xlsx")
