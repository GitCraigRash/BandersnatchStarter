from data import Database
from pymongo import MongoClient
from os import getenv
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()
db_url = getenv("URI")
db_name = "ClusterX"
collection_name = "characters"
client = MongoClient(db_url)
db = client[db_name]
collection = db[collection_name]


def test_default_count():
    '''Tests of output is integer'''
    c = collection.count_documents({})
    d = len(list(collection.find({})))
    report = "problem with count(), " + str(c) + "!=" + str(d)
    if c != d:
        return report
    return ".count() passed .test_default_count()"


def test_default_seed():
    '''
    Tests if seed() adds documents to Database.
    It is important that "if" tests before is less than collection.
    '''
    before = a.count() 
    a.seed(10)
    report = ".test_default_seed() raised error. Current number of docs " 
    report = report + str(before)
    if before < collection.count_documents({}):
        return ".seed() passed .test_default_seed()" 
    return report


def test_default_reset():
    '''Test if Database is empty'''
    a.reset()
    report = ".test_default_reset() failed. Detecting documents in database."
    if collection.count_documents({}) == 0:
        return ".reset() passed .test_default_reset()" 
    return report


def test_default_dataframe():
    '''Tests if output is a pandas DataFrame'''
    db = Database()
    a = DataFrame({'Name':['Alice'],'Age':[12]})
    db.seed(10)
    report = ".test_default_dataframe() failed. Database is "
    report = report + str(type(db.dataframe()))
    if type(db.dataframe()) == type(a):
        return ".test_default_dataframe() passed .dataframe()"
    return report


def test_default_html_table(a):
    '''Detects if output is <html>'''
    a.seed(10)
    report = ".test_default_html_table() failed. Database is "
    report = report + str(a.html_table())
    if str(a.html_table()[:6]) == "<table":
        return ".test_default_html_table() passed .html_table()"
    return report

if __name__ == '__main__':
    a = Database()
    print(test_default_count())
    print(test_default_seed())
    print(test_default_reset())
    print(test_default_dataframe())
    print(test_default_html_table(a))
