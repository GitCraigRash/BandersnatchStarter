
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from data import Database
# Create Monsters
db = Database()
db.reset()
db.seed(1000)
df = db.dataframe()



# Clean DataFrame
df["Rarity"] = pd.DataFrame([int(i[-1]) for i in df["Rarity"]]) 
n = len(df[df['Rarity'] == 5])*9
df = df.groupby('Rarity').apply(lambda x: x.head(n) 
                                        if x['Rarity'].iloc[0] != 5
                                        else x).reset_index(drop=True)
# Designate Target
target = df["Rarity"]

columns_to_drop= ["Name","Rarity","Damage","Type","Timestamp"]
df=df.drop(columns= columns_to_drop)

# Split Training and Test data
X_train,X_test,y_train,y_test = train_test_split(
                                    df,target,test_size=0.1, random_state=42)


# DummyClassifier baseline model
baseline_model = DummyClassifier(strategy='most_frequent') 
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)
# Evaluate the baseline model
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
base_cm = confusion_matrix(y_test, baseline_predictions)


# SVC model
regr = SVC()
svc_model = regr.fit(X_train, y_train)
scv_predictions = svc_model.predict(X_test)
# Evaluate the SVC model
svc_accuracy = accuracy_score(y_test, scv_predictions)
SVC_cm = confusion_matrix(y_test, scv_predictions)


# RandomForestClassifier model
rf_classifier = RandomForestClassifier(
                            n_estimators=230, random_state=32)
rf_classifier.fit(X_train, y_train)
rfc_predictions = rf_classifier.predict(X_test)
# Evaluate the RandomForestClassifier model
rfc_accuracy = accuracy_score(y_test, rfc_predictions)
RFC_cm = confusion_matrix(y_test, rfc_predictions)


# XGBoost model
xgb_cl = xgb()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001]
}
scoring_metric = make_scorer(accuracy_score)
grid_search = GridSearchCV(
    estimator=xgb_cl,
    param_grid=param_grid,
    scoring=scoring_metric,
    cv=3,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
# Train with the best parameters
xgb_bp = xgb(**best_params)
xgb_bp.fit(X_train, y_train)
xgb_bp_predictions = xgb_bp.predict(X_test)
# Evaluate the XGB model
xgb_bp_accuracy_score = accuracy_score(y_test, xgb_bp_predictions)
XGB_cm = confusion_matrix(y_test, xgb_bp_predictions)


# Baseline
print("\nBaseline Confusion Matrix:")
for i in range(len(base_cm)):
    row_str = "|".join([f"{count:2d}" for count in base_cm[i]])
    print(f"| {row_str} |")
print(f"Baseline Accuracy: {baseline_accuracy:.2}")
# SVC
print("\nSVC Confusion Matrix:")
for i in range(len(SVC_cm)):
    row_str = "|".join([f"{count:2d}" for count in SVC_cm[i]])
    print(f"| {row_str} |")
print(f"SVC: {svc_accuracy:.2%}")

# RFC
print("\nRFC Confusion Matrix:")
for i in range(len(RFC_cm)):
    row_str = "|".join([f"{count:2d}" for count in RFC_cm[i]])
    print(f"| {row_str} |")
print(f"RFC Accuracy: {rfc_accuracy:.2}")

# XGB
print("\nXGB Confusion Matrix:")
for i in range(len(XGB_cm)):
    row_str = "|".join([f"{count:2d}" for count in XGB_cm[i]])
    print(f"| {row_str} |")
print(f"XGB Accuracy: {rfc_accuracy:.2}")
