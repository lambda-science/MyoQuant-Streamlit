FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/lambda-science/MyoQuant.git .
RUN wget https://lbgi.fr/~meyer/SDH_models/model.h5

RUN python -m pip install poetry
RUN python -m poetry config virtualenvs.create false
RUN python -m poetry install --no-root --no-interaction && rm -rf ~/.cache/pypoetry/{cache,artifacts}

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]