import xgboost
import sklearn
def run_():
    import joblib
    model = xgboost.XGBClassifier()
    trained_model = joblib.load('./best_model.sav')
    print(trained_model)
    return trained_model
