"""_summary_
    A company Reposiory
"""

from typing import Any, List
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo import ReturnDocument
from ..mongo_connection import db


class Repository():
    """_summary_
        This is an repository for a Company
        that allow conect with mongoBD
    """
    collection: str
    database: Database = db

    def __init__(self, collection: str) -> None:
        self.collection = collection

    def get_collection(self) -> Collection[Any]:
        """_summary_
            Get all Company Collection data
        Returns:
            Collection[Any]: All records
        """
        return self.database[self.collection]

    
    def insertDataframe(self, data: Any,file_name: str) -> str:
        """_summary_
            Insert a dataframe to a collection
        Args:
            data (Any): The data to insert
        Returns:
            str: The id of the data inserted
        """
        #print("data", data)
        return self.database[self.collection].insert_one({"data":data,"file_name":file_name}).inserted_id
        
 
    def getColeccion(self,dataset: str) -> Any:
        return self.database[dataset].find()
    

    def getCollectionByID(self, id: Any) -> Any:
        return self.database['main'].find({"_id": id})
    
    def load_file(self, file: str) -> object:
        """_summary_

        Args:
            file (str): The file to load

        Returns:
            object: The file loaded
        """
        return self.database[self.collection].insert_one(file)
    
    def delete_one(self,query_filer,**args):
        """_summary_

        Args:
            query_filer (str): A Filter field to delete company

        Returns:
            CompanyModel: The Company deleted
        """
        return self.database[self.collection].find_one_and_delete(query_filer,**args)

    def aggregate(self, pipeline,**agrs):
        return self.database[self.collection].aggregate(pipeline)
