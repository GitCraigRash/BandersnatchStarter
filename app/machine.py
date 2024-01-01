import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from app.data import Database


class Machine:

    def __init__(self, df):
        df["Rarity"] = pd.DataFrame([int(i[-1]) for i in df["Rarity"]])
        n = len(df[df['Rarity'] == 5])*10
        df = df.groupby('Rarity').apply(lambda x: x.head(n) 
                                        if x['Rarity'].iloc[0] != 5
                                        else x).reset_index(drop=True)
        target = df["Rarity"]
        columns_to_drop= ["Rarity"]
        df = df.drop(columns=columns_to_drop)
        self.name = "Random Forest Classifier"
        self.model = RandomForestClassifier(n_estimators=230, random_state=32)
        self.model.fit(df,target)

    def __call__(self, feature_basis):
        prediction = self.model.predict(feature_basis)
        prediction_prob = self.model.predict_proba(feature_basis)
        prob = max(prediction_prob[0]).round(4)
        return "Rank "+ str(prediction[0]),prob
    
    def save(self, filepath):
        joblib.dump(self.model,'model.joblib')

    @staticmethod
    def open(filepath):
        return joblib.load('model.joblib')

    def info(self):
        db = Database()
        df = db.dataframe
        doc = db.collection.find_one()
        time = doc.get("Timestamp")
        time = datetime.strptime(time,"%Y-%m-%d %H:%M:%S")
        time = time.strftime("%Y-%m-%d %I:%M:%S %p")
        info = "Base Model: " + self.name  + "<br" + "/>" + "Timestamp: " + time
        return info
