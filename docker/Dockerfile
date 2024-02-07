FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN apt-get -y update
RUN apt-get -y install git
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
