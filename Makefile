.PHONY: install train serve test docker-build docker-up k8s-deploy clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

train:
	python src/train.py

serve:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

docker-build:
	docker build -t churn-api:latest .

docker-up:
	docker-compose up --build

k8s-deploy:
	kubectl apply -f k8s/

clean:
	rm -rf mlruns/
	rm -rf models/*.pkl
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
