FROM ubuntu:22.04

MAINTAINER Yuri <inspiration@g.skku.edu>

RUN apt-get update
RUN apt-get install -y python3-pip

ADD . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn", "api_server:app", "--reload"]