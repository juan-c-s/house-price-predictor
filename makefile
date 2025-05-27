## Run MLflow server (already in your Makefile)
mlflow-server:
	uv run mlflow server --host 127.0.0.1 --port 8080

.PHONY: dash-app


dash-app:
	uv run src/house_price_predictor/dash_app/main.py 