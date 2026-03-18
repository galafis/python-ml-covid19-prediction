.PHONY: install test lint run docker clean

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -v --tb=short

lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503

run:
	PYTHONPATH=. python main.py --state SP --model-type all

docker-build:
	docker build -t covid-ml-prediction .

docker-run:
	docker-compose up covid-prediction

docker-test:
	docker-compose up test

clean:
	rm -rf output/ __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
