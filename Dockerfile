FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

# RUN useradd -m ranjeet

# RUN chown -R ranjeet:ranjeet /home/ranjeet/

COPY . .

# USER ranjeet

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

# WORKDIR /home/ranjeet/app

# EXPOSE 8080

ENTRYPOINT python main.py
