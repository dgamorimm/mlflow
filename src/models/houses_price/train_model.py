from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost
import pandas as pd
import math
import mlflow


df = pd.read_csv('data/processed/casas.csv')
print(df.head())

X = df.drop('preco', axis = 1)
y = df['preco'].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

xgb_params = {
    'learning_rate' : 0.2,
    'seed' : 42
}

mlflow.set_experiment('house-prices-script')
mlflow.xgboost.autolog()
with mlflow.start_run():
    xgb = xgboost.train(xgb_params, dtrain, evals=[(dtrain, 'train')])
    xgb_predict = xgb.predict(dtest)
    mlflow.log_metrics({
    'mse' : mean_squared_error(y_test, xgb_predict),
    'rmse' : math.sqrt(mean_squared_error(y_test, xgb_predict)),
    'r2' : r2_score(y_test, xgb_predict)
    })