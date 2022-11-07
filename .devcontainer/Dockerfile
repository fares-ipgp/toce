FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y make

# set user and workdir
# RUN useradd -rm -d /home/dv -s /bin/bash -g root -G sudo -u 1001 dv
# USER dv
#WORKDIR /home/dv

# conda environment
COPY environment.yml .
RUN conda env create -f environment.yml -n toce

RUN conda init

# Make RUN commands use the new environment:
RUN echo "conda activate toce" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure numpy is installed:"
RUN python -c "import numpy"

EXPOSE 5000

# The code to run when container is started:
COPY entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]
