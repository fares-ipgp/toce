
.PHONY: all clean

TOC_URL = "https://raw.githubusercontent.com/fares-ipgp/toce/main/data/csv/toc_data.csv"

all: data/raw/toc_data.csv

clean:
 rm -f data/raw/*.csv

data/raw/toc.csv:
	python src/data/download.py $(TOC_URL) $@