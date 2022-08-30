# HistoQuant-Streamlit

HistoQuant-Streamlit is a web application for quantifying the number of cells in a histological image.  
It is built using CellPose, Stardist and custom models and image analysis techniques to automatically analyse myopathy histology images.  
This web application is intended for demonstration purposes only.

## How to install

A streamlit cloud demo instance should be deployed at https://lambda-science-histoquant-streamlit-home-39mwbj.streamlitapp.com/. I am currently working on proper docker images and tutorial to deploy the application. Meanwhile you can still use the following instructions:

### Docker

You can build the docker image by running `docker build -t streamlit .` and launch the container using `docker run -p 8501:8501 streamlit`.

### Non-Docker

If you do not want to use docker you can install the poetry package in a miniconda (python 3.9) base env, run `poetry install` to install the python env, activate the env with `poetry shell` and launch the app by running `streamlit run Home.py`.

## How to use

A streamlit cloud demo instance should be deployed at https://lambda-science-histoquant-streamlit-home-39mwbj.streamlitapp.com/. I am currently working on docker images and tutorial to deploy the application.  
Once on the demo, click on on the corresponding staining analysis on the sidebar, and upload your histology image. Results will be displayed in the main area automatically.  
For HE Staining analysis you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_he.jpg)  
For SDH Staining analysis you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_sdh.jpg)

## Who and how

- Creator and Maintainer: [Corentin Meyer, 3rd year PhD Studient in the CSTB Team, ICube - CNRS - Unistra](https://lambda-science.github.io/)
- Source code for this appliaction is availiable [HERE](https://github.com/lambda-science/HistoQuant-Streamlit)
