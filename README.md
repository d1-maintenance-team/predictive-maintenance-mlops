# predictive-maintenance-mlops

Predictive maintenance MLOps pipeline: real-time RUL prediction, model
registry, drift detection, and automated retraining.

## Architecture

- `data/` - Raw, processed, and feature data.
- `notebooks/` - Exploration and analysis notebooks.
- `ml/common/` - Shared preprocessing, labels, and time-series windowing.
- `ml/xgboost/` - XGBoost feature preparation, training, evaluation, and inference.
- `ml/lstm/` - PyTorch LSTM dataset, model, training, evaluation, and inference.
- `evaluation/` - Cross-model evaluation utilities.
- `simulator/` - Synthetic equipment and sensor-data generation.
- `streaming/` - Real-time ingestion and event-processing components.
- `api/` - Application and prediction API.
- `monitoring/` - Metrics, drift detection, and retraining.
- `config/` - Mosquitto, Node-RED, and Grafana configuration.
- `tests/` - Unit, integration, and end-to-end tests.
- `docker-compose.yml` - Local multi-service orchestration.
- `requirements.txt` - Python dependencies.

