
.PHONY: all clean

TOC_URL = "https://raw.githubusercontent.com/fares-ipgp/toce/main/data/external/toce_data.csv"

all: data/raw/toce.csv

clean:
	rm -f data/raw/toce.csv

data/raw/toce.csv:
	python src/data/download.py $(TOC_URL) $@