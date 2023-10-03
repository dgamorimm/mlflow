- mlflow models build-docker -m 'models:/House Prices/Production' -n 'house-prices'

- docker run -p 5001:8080 'house-prices'