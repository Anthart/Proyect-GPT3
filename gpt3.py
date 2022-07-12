import clr
clr.AddReference('dll/ventanaGPT')
import os
import openai
import pandas as pd
from System.Windows.Forms import *
from ventanaGPT import Form1

def clicIngresar(sender,e):
    prompt = app.txtIngreso.Text
    respuesta = evaluar(prompt)
    app.txtIngreso.AppendText(respuesta)
 
def clicLimpiar(sender,e):
    app.txtIngreso.Clear()

def evaluar(orden):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt= orden,
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response.choices[0].text

openai.api_key = os.getenv("OPENAI_API_KEY")
app = Form1()
app.btIngresar.Click += clicIngresar
app.btlimpiar.Click += clicLimpiar
Application.Run(app)