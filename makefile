
.PHONY: all clean update-env

TOC_URL = "https://raw.githubusercontent.com/fares-ipgp/toce/main/data/external/toce_sichuan.csv"

TOC_URL = "https://raw.githubusercontent.com/fares-ipgp/toce/main/data/external/toce_mayer.csv"

all: data/raw/toce.csv data/processed/processed.pickle models/svr.model

update-env:
	conda env update -f ./.devcontainer/environment.yml -n toce 

data/raw/toce.csv:
	python src/data/download.py $(TOC_URL) $@

reports/figures/exploratory.png: data/processed/processed.pickle
	python src/visualization/exploratory.py $< $@

data/processed/processed.pickle: data/raw/toce.csv
	python src/data/preprocess.py $< $@ 

models/svr.model: data/processed/processed.pickle
	python src/models/train.py $< $@

clean:
	# python cache
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	
	# clean targets
	rm -f data/raw/toce.csv
	rm -f data/processed/*.pickle
	rm -f reports/figures/*.png
	rm -f models/*.model

	# mlflow
	rm -rf mlruns/.trash