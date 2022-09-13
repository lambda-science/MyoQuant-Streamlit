import streamlit as st

st.set_page_config(
    page_title="MyoQuant",
    page_icon="🔬",
)

st.write("# Welcome to MyoQuant! 👋")

st.sidebar.success("Select the corresponding staining analysis above.")

st.markdown(
    """
# MyoQuant

MyoQuant is a web application for quantifying the number of cells in a histological image.  
It is built using CellPose, Stardist and custom models and image analysis techniques to automatically analyze myopathy histology images.  
This web application is intended for demonstration purposes only.

## How to Use

A Streamlit cloud demo instance should be deployed at https://lbgi.fr/MyoQuant/. I am currently working on docker images and tutorial to deploy the application.  
Once on the demo, click on the corresponding staining analysis on the sidebar, and upload your histology image. Results will be displayed in the main area automatically.  
For HE Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_he.jpg)  
For SDH Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_sdh.jpg)

## Who and how

- Creator and Maintainer: [Corentin Meyer, 3rd year PhD Student in the CSTB Team, ICube—CNRS—Unistra] (https://lambda-science.github.io/)
- The source code for this application is available [HERE] (https://github.com/lambda-science/MyoQuant)

"""
)
