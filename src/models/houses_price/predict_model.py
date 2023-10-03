import mlflow
logged_model = 'runs:/238d7720b89d4ca5ae333812f78e5f3b/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv('data/processed/casas_X.csv')
predicted = loaded_model.predict(pd.DataFrame(data))

data['predicted'] = predicted
data.to_csv('precos.csv')

"""
Através da linha de comando

mlflow models predict -m [id-da-execucao] -i [base-de-entrada] -t [formato] -o [arquivo-saída]
mlflow models predict -m 'runs:/238d7720b89d4ca5ae333812f78e5f3b/model' -i 'data/processed/casas_X.csv' -t 'csv' -o 'preco2.csv'

Podemos servir o modelo também

mlflow models serve -m [id-da-execução]
mlflow models serve -m 'runs:/238d7720b89d4ca5ae333812f78e5f3b/model' -p 5001

"""