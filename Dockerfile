FROM python:3.10-slim

WORKDIR /app

# Copy files and dirs
COPY requirements.txt requirements.txt
COPY configs configs
COPY models models
COPY train.py train.py

# Update pip and install requirements
RUN python3 -m pip install -U pip wheel setuptools
RUN pip3 install -r requirements.txt

# Entry development mode
ENTRYPOINT ["/bin/bash"]

# CMD ["app.py"] # optional, add commands here to start or initialize something when you run