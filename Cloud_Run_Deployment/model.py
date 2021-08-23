import joblib
import sklearn
import xgboost

def run_():
    model = xgboost.XGBClassifier()
    trained_model = joblib.load('./best_model.sav')
    return trained_model
