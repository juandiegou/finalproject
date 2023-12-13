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


def allValuesInColBeNumber(data, key):
    """
    Say if all values in col be integer or float
    Iterate a pandas col,
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


def getValueOfMean(data, key):
    """
    Enter a pandas dataframe 
    and calculate mean 
    """
    _errors = 0
    _lenData = 0
    _sum = 0
    for i in data[key]:
        if str(i) == "NaN":
            continue
        else:
            if type(i) is float or type(i) is int:
                _sum = _sum + float(i)
                _lenData = _lenData + 1
            else:
                _errors = _errors + 1


    if _errors == 0:
        return _sum / _lenData
    
    return 0


for i in obj.columns:
    _data = obj[i]
    if "NaN" in _data.values:
        if allValuesInColBeNumber(obj, i):
            _mean = getValueOfMean(obj, i)
            obj[i] = obj[i].replace('NaN', _mean)


print(obj)

            




"""

for index, row in _update.iterrows():
    print(index)
    print(row)


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