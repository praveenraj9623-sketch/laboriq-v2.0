install:
	pip install -r requirements.txt

run-pipeline:
	python pipelines/run_pipeline.py

app:
	streamlit run app.py

test:
	pytest -q

clean:
	rm -rf reports/*.csv reports/*.json models/*.joblib .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
