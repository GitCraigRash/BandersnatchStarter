from os import getenv
from certifi import where
from dotenv import load_dotenv
from MonsterLab import Monster
import pandas
from pandas import DataFrame
from pymongo import MongoClient


class Database:
    """
    Accesses the MongoDB Database
    Attributes:
        db_url - The url to the database
        db_name - The name of the databse
        colleciton_name - name of databse collection
        client - MongoClient from pymongo used to connect to Database
        db - the connected database
        collection - the connected collection
    Methods:
        seed(self,amount): Adds 'amount' number of monsters to Database
        reset(): deletes all monsters in database
        count() -> int: returns number of monsters in database
        dataframe() -> DataFrame: returns DF of Database without _id
        html_table() -> str: returns DF in an html format
    """
    def __init__(self):
        load_dotenv()
        db_url = getenv("URI")
        db_name = "ClusterX"
        collection_name = "characters"
        self.client = MongoClient(db_url, tlsCAFile=where())
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def seed(self, amount):
        """
        Inserts the desired number of monsters into the Database.
        """
        self.collection.insert_many(
            [Monster().to_dict() for _ in range(amount)]
        )

    def reset(self):
        """
        Deletes all monsters in the database.
        """
        return self.collection.delete_many({})

    def count(self) -> int:
        """
        Counts all monsters in the database.
        """
        return self.collection.count_documents({})

    def dataframe(self) -> DataFrame:
        """
        Converts all documents in database to a DataFrame. 
        Removes "_id" column during conversion.
        """
        df = DataFrame(self.collection.find())
        return df.drop(df.columns[0], axis=1)

    def html_table(self) -> str:
        """
        Converts DataFrame of documents into html text for webpage.
        """
        df = self.dataframe()
        return df.to_html()
    
if __name__ == '__main__':
    db = Database()
    db.reset()
    db.seed(1000)
    
