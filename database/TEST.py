"""_summary_
    A Mongo DB Connection
"""
from os import getenv
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
#HOST: str = str(getenv('MONGODB_HOST'))
#PORT: str = str(getenv('MONGODB_PORT'))
USER: str = str(getenv('USER'))
PASSWORD: str = str(getenv('PASSWORD'))
DATABASE: str = str(getenv('DATABASE'))
#MONGO_URI: str = f'mongodb://{USER}:{PASSWORD}@{HOST}:{PORT}'
#MONGO_URI : str = f'mongodb://{HOST}:{PORT}'
MONGO_URI : str = f"mongodb+srv://{USER}:{PASSWORD}@data.qglrnpr.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client[DATABASE]

from bson import ObjectId
id = ObjectId('65790f74348aa7ad03724567')
obj = db['main'].find({"_id": id})

obj = list(obj)
obj = obj[0]['data']
objConvert = obj
objConvert = str(objConvert)
objConvert = objConvert.replace("\'", "\"")
import json
objConvert = json.loads(objConvert)

import pandas as pd
obj = pd.DataFrame(objConvert)
# 

print(obj)

obj = obj[~obj.eq("NaN").any(axis=1)]

print(obj)

"""







_mean = {}


def fillIfIsNumber(data, key):
    data = data[key].fillna(data.mean())

def fillIsText(data, key):
    data = data[key]




# Typo de col
for i in obj.columns.tolist():
    _typeData = obj[i].dtypes

    if "int" in str(_typeData).lower() or "float" in str(_typeData).lower():
        print("Es numero")
    else:
        print("No es numero")


import json

obj = str(obj).replace("\'", "\"")

obj = json.loads(obj)



import pandas as pd

df = pd.DataFrame(obj)

print(len(df))

print(df)

df = df.dropna()

print()

print(df)
"""