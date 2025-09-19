CONDA_ENV=oncotarget-lite

setup:
	conda env create -f environment.yml --force

lint:
	conda run -n $(CONDA_ENV) ruff check .
	conda run -n $(CONDA_ENV) black --check .

test:
	conda run -n $(CONDA_ENV) pytest --cov=oncotarget --cov-report=term-missing

run-app:
	conda run -n $(CONDA_ENV) streamlit run app/streamlit_app.py

build-img:
	docker build -t oncotarget-lite:latest .
