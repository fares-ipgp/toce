
.PHONY: all clean update-env

BASE_DATA_URL = "https://raw.githubusercontent.com/fares-ipgp/toce/main/data/external/"

DATA = data/raw/toce_mayer.csv
DATA_PROC = data/processed/processed_toce_mayer.pickle
MODELS = models/svr.model

all: $(DATA) $(DATA_PROC) $(MODELS)

update-env:
	conda env update -f ./.devcontainer/environment.yml -n toce 

data/raw/%.csv:
	python src/data/download.py $(BASE_DATA_URL)$*.csv $@

data/processed/processed_%.pickle: data/raw/%.csv
	python src/data/preprocess.py $< $@ 

models/%.model: $(DATA_PROC) 
	python src/models/train.py $< $@ $*

reports/figures/%.png: $(DATA_PROC) 
	python src/visualization/$*.py $< $@

clean:
	# python cache
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	
	# clean targets
	rm -f data/raw/toce.csv
	rm -f data/processed/*.pickle
	rm -f reports/figures/*.png
	rm -f models/*.model

	# mlflow trash
	rm -rf mlruns/.trash