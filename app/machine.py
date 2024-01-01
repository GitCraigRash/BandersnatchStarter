from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from app.data import Database

class Machine:

    def __init__(self, df):
        df.loc[:,"Rarity"] = df["Rarity"].apply(lambda x: int(str(x)[-1]))
        n = len(df[df['Rarity'] == 5])*10
        df = df.groupby('Rarity').apply(lambda x: x.head(n) if x['Rarity'].iloc[0] != 5 else x).reset_index(drop=True)
        target = df["Rarity"]
        
        columns_to_drop= ["Rarity"]
        df = df.drop(columns=columns_to_drop)
        
        scaler = StandardScaler().fit(df)
        features = scaler.transform(df)

        self.name = "Random Forest Classifier"
        self.model = RandomForestClassifier(n_estimators=230, random_state=32)
        self.model.fit(features,target)

        

    def __call__(self, feature_basis):
        prediction = self.model.predict(feature_basis)
        prediction_prob = self.model.predict_proba(feature_basis)

        prob = max(prediction_prob[0]).round(4)
        return prediction[0],"{:.2%}".format(prob)
    def save(self, filepath):
        joblib.dump(self.model,'model.joblib')

    @staticmethod
    def open(filepath):
        return joblib.load('model.joblib')

    def info(self):
        db = Database()
        df = db.dataframe
        return db.collection.find_one({"Timestamp"}), db.name
