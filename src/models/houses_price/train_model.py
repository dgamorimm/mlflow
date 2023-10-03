from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost
import pandas as pd
import math
import mlflow
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='House Prices ML')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.3,
        help='Taxa de aprendizado para atualizar o tamanho de cada passo do boosting'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Profundidade máxima das árvores'
    )
    
    return parser.parse_args()

df = pd.read_csv('data/processed/casas.csv')

X = df.drop('preco', axis = 1)
y = df['preco'].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)


def main():
    args = parse_args()
    xgb_params = {
        'learning_rate' : args.learning_rate,
        'max_depth' : args.max_depth,
        'seed' : 42
    }

    mlflow.set_tracking_uri('http://localhost:5000')
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

if __name__ == '__main__':
    main()