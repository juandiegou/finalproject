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
id = ObjectId('65778e5855ceae4b2d460b3b')
obj = db['main'].find({"_id": id})

obj = list(obj)[0]
obj = obj['file_name']




print(obj)

"""
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