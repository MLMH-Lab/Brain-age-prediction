FROM python:3.7-slim-buster

ADD ./requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt
