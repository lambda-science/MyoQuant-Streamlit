import streamlit as st

st.set_page_config(
    page_title="HistoQuant-Streamlit",
    page_icon="🔬",
)

st.write("# Welcome to HistoQuant-Streamlit! 👋")

st.sidebar.success("Select the corresponding staining analysis above.")

st.markdown(
    """
    HistoQuant-Streamlit is a web application for quantifying the number of cells in a histological image.   
    It is built using CellPose, Stardist and custom models and image analysis techniques to automatically analyse myopathy histology images.  
    This web application is intended for demonstration purposes only.  
    ### How to use
    Click on on the corresponding staining analysis on the sidebar, and upload your histology image. Results will be displayed in the main area automatically.
    ### Who and how
    - Creator and Maintainer: [Corentin Meyer, 3rd year PhD Studient in the CSTB Team, ICube - CNRS - Unistra](https://lambda-science.github.io/)
    - Source code for this appliaction is availiable [HERE](https://github.com/lambda-science/HistoQuant-Streamlit)
"""
)
