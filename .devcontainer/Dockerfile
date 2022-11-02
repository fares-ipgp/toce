FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y make

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml -n toce

# Make RUN commands use the new environment:
RUN echo "conda activate toce" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure numpy is installed:"
RUN python -c "import numpy"

# The code to run when container is started:
COPY entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]
