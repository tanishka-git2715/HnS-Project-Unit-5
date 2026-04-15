# Customer Churn Prediction MLOps Pipeline 🚀

This repository contains a complete, production-ready MLOps pipeline for predicting customer churn. It demonstrates an end-to-end workflow from data generation and training to deployment and monitoring.

## 🎯 Project Goals
- **Experiment Tracking:** Use MLflow to track model parameters and metrics.
- **REST API Serving:** Serve models via FastAPI with Prometheus metrics.
- **Containerization:** Package the application using Docker and Docker Compose.
- **CI/CD:** Automate workflows with GitHub Actions.
- **Orchestration:** Deploy to Kubernetes with liveness/readiness probes.
- **Monitoring:** Collect metrics with Prometheus and visualize with Grafana.

---

## 🏗️ Architecture
The system is built with a modular architecture:
1. **Data Layer:** Synthetic data generation and preprocessing.
2. **Training Layer:** Scikit-learn model training with MLflow logging.
3. **Serving Layer:** FastAPI application for low-latency inference.
4. **DevOps Layer:** Docker containerization and Kubernetes orchestration.
5. **Observability Layer:** Prometheus metrics and Grafana dashboards.

---

## 📁 Project Structure
```text
churn-mlops/
├── app/               # FastAPI application
├── data/              # Raw data storage
├── k8s/               # Kubernetes manifests
├── mlruns/            # MLflow experiment data
├── models/            # Local model artifacts
├── monitoring/        # Prometheus & Grafana config
├── src/               # Data processing & training scripts
├── tests/             # Pytest suite
├── Dockerfile         # Container definition
├── docker-compose.yml # Local orchestration
├── Makefile           # Task automation
└── README.md          # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation
Ensure you have Python 3.11+ installed.
```bash
make install
```

### 2. Run Training
This will generate synthetic data, train the model, and log metrics to MLflow.
```bash
make train
```
*View MLflow UI: `mlflow ui`*

### 3. Start Services locally
Launch the API, MLflow, Prometheus, and Grafana using Docker Compose.
```bash
make docker-up
```

---

## 🔌 API Documentation
The API is available at `http://localhost:8000`.

### Endpoints:
- `GET /health` - Check API and model status.
- `POST /predict` - Send customer data to get a churn prediction.
- `GET /metrics` - Prometheus metrics endpoint.
- `POST /retrain` - Manually trigger a model retraining pipeline.

### Sample Prediction Request (cURL):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

---

## 📊 Monitoring
- **Prometheus UI:** `http://localhost:9090`
- **Grafana Dashboard:** `http://localhost:3000` (User: `admin`, Pass: `admin`)
  - Once logged in, go to `Dashboards` -> `Browse` to find the "Customer Churn API Dashboard".

---

## ☸️ Kubernetes Deployment
To deploy to a cluster (e.g., Minikube):
```bash
make k8s-deploy
```

---

## 🎤 Viva Preparation Section

### 1. What is MLOps?
MLOps (Machine Learning Operations) is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently. It combines ML development (Dev) and ML system operations (Ops).

### 2. Difference between DevOps and MLOps
While DevOps focuses on traditional software, MLOps adds unique challenges like:
- **Data Versioning:** Changes in input data can affect model performance.
- **Model Drift:** Models degrade over time as real-world data changes.
- **Experiment Tracking:** Need to track hyperparameters and artistic choices.

### 3. Why MLflow is used?
MLflow provides a unified platform for the ML lifecycle, including tracking experiments, packaging code into reproducible runs, and managing/deploying models.

### 4. Why Docker is needed?
Docker ensures that the application runs identically across different environments (dev, staging, prod) by packaging code, libraries, and dependencies together in a container.

### 5. Kubernetes Scaling
Kubernetes allows horizontal scaling of the API. If traffic increases, K8s can automatically spin up more replicas of the `churn-api` pod to handle the load.

---

## 📸 Viva Screenshots Instructions
To present this project, show the following:
1. **Terminal:** Show `make train` successfully completing.
2. **MLflow UI:** Show the logged metrics (Accuracy, Recall) and the artifacts.
3. **FastAPI Docs:** Show the `/docs` page (Swagger UI) and run a test prediction.
4. **Docker Desktop:** Show all services running in the compose stack.
5. **Grafana:** Show the "Latency P95" or "Request Rate" graphs updating in real-time.

---

## ⚖️ License
Open-source Project under MIT License.
