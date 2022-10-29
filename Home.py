import streamlit as st

st.set_page_config(
    page_title="MyoQuant",
    page_icon="🔬",
)

st.write("# Welcome to MyoQuant demo ! 👋")

st.sidebar.success("Select the corresponding staining analysis above.")

st.markdown(
    """
# MyoQuant-Streamlit🔬

MyoQuant-Streamlit🔬 is a demo web application to showcase usage of MyoQuant.  
MyoQuant is a command line tool to quantify pathological feature in histology images.  
It is built using CellPose, Stardist, custom neural-network models and image analysis techniques to automatically analyze myopathy histology images.  
This web application is intended for demonstration purposes only.  

## How to Use

Once on the demo, click on the corresponding staining analysis on the sidebar, and upload your histology image. Results will be displayed in the main area automatically.  
For HE Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_he.jpg)  
For SDH Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_sdh.jpg)

## Who and how

- Creator and Maintainer: [Corentin Meyer, 3rd year PhD Student in the CSTB Team, ICube — CNRS — Unistra](https://lambda-science.github.io/)
- The source code for MyoQuant is available [HERE](https://github.com/lambda-science/MyoQuant), for the demo website it is available [HERE](https://github.com/lambda-science/MyoQuant-Streamlit)

"""
)
