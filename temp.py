import json

data = ""

with open('file.txt', 'r', encoding="UTF-8") as f:
    data = f.read()


data = data.replace("\'", "\"")

data = json.loads(data)


import pandas as pd

df = pd.DataFrame(data)

def formatPrettyResponse(data):
    d = data.describe()

    final_cols = ['Age ', 'Gender', 'BMI', 'Nausea/Vomting']

    for i in d.columns:
        if i not in final_cols:
            d=d.drop([i], axis=1)

    print(d)


print(formatPrettyResponse(df))






