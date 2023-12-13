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
id = ObjectId('657925448b5f4880faf705a1')
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



import numpy as np

def allValuesInColBeNumber(self, data, key):
    """
    retrun bool Say if all values in col be (integer or float)
    Iterate a pandas col
    if all values in col be integer or float always expect condition
    and hav inusual value is a error
    """
    _errors = 0
    for i in data[key]:
        if str(i) == "NaN":
            continue
        else:
            if type(i) is float or type(i) is int:
                continue
        _errors = _errors + 1
        
    return _errors == 0

import matplotlib.pyplot as plt

for i in obj.columns:
    if allValuesInColBeNumber(0, obj, i):
        plt.figure(figsize=(8, 4))
        obj[i].plot(kind='density', color='blue')
        plt.title(i)
        plt.xlabel('Valores')
        plt.ylabel('Densidad')
        plt.grid(True)
        plt.show()

            

"""


"""