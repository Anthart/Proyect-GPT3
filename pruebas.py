import os.path

from prompt_file import guardar_prompt
from prompt_file import cargar_lista_prompt
from tokens import calcular_total_pagar
import pandas as pd


# prompt = ("I'm reading fragments from some source, and some words are not easy to understand. "
#               "I'm classifying these words into \"very easy\", \"easy\", "
#               "\"neutral\", \"difficult\" and \"very difficult\".\nAfter reading the fragment \" However, "
#               "no defects in axon pathfinding along the monosynaptic reflex arc or in muscle "
#               "spindle differentiation have been noted in PV KO mice, which develop normally and "
#               "show no apparent changes in their behavior or physical activity (Schwaller et al. 1999)."
#               " \". I find that word \"spindle\" is neutral\n\n###\n\n"
#               "After reading the fragment \" I will sprinkle clean water on you, "
#               "and you shall be clean: from all your filthiness, and from all your idols, will I cleanse you. \"."
#               "I find that word \"filthiness\" is easy\n\n###\n\n"
#               "After reading the fragment \" Moreover, acute dosing does not recapitulate the "
#               "marked learning deficits produced in rodents [15,16] by chronic exposure to dopamine D2R "
#               "antagonists [6,7] \" . I find that word \"antagonists\" is difficult\n\n###\n\n"
#               "After reading the fragment \" Thrombus formation on fissured atherosclerotic "
#               "plaques is the precipitating event in the transition from a stable or subclinical atherosclerotic"
#               "disease and leads to acute myocardial infarction, ischemic stroke or peripheral arterial occlusion. "
#               "\". I find that word \"Thrombus\" is very difficult\n\n###\n\n"
#               "After reading the fragment @oracion I find that word @aEvaluar is")
#
# guardar_prompt(prompt)

if os.path.isdir('resultados'):
    print("existe")
else:
    print("no exite")

