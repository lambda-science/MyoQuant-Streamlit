![Twitter Follow](https://img.shields.io/twitter/follow/corentinm_py?style=social) ![Demo Version](https://img.shields.io/badge/Demo-https%3A%2F%2Flbgi.fr%2FMyoQuant%2F-9cf) ![PyPi](https://img.shields.io/badge/PyPi-https%3A%2F%2Fpypi.org%2Fproject%2Fmyoquant%2F-blueviolet) ![Pypi verison](https://img.shields.io/pypi/v/myoquant)

# MyoQuant-Streamlit🔬: a demo web interface for the MyoQuant tool.

## Please refer to the [MyoQuant GitHub Repository](https://github.com/lambda-science/MyoQuant) or the [MyoQuant PyPi repository](https://pypi.org/project/myoquant/) for full documentation.

MyoQuant-Streamlit🔬 is a demo web interface to showcase the usage of MyoQuant.

<p align="center">
  <img src="https://i.imgur.com/mzALgZL.png" alt="IMPatienT Banner" style="border-radius: 25px;" />
</p>

MyoQuant🔬 is a command-line tool to automatically quantify pathological features in muscle fiber histology images.  
It is built using CellPose, Stardist, custom neural-network models and image analysis techniques to automatically analyze myopathy histology images. Currently MyoQuant is capable of quantifying centralization of nuclei in muscle fiber with HE staining and anomaly in the mitochondria distribution in muscle fibers with SDH staining.  
This web application is intended for demonstration purposes only.

## How to install or deploy the interface

The demo version is deployed at https://lbgi.fr/MyoQuant/. You can deploy your own demo version using Docker, your own python environment or google Colab for GPU support.

### Docker

You can build the docker image by running `docker build -t streamlit .` and launch the container using `docker run -p 8501:8501 streamlit`.

### Non-Docker

If you do not want to use Docker you can install the poetry package in a miniconda (python 3.9, 3.10) base env, run `poetry install` to install the python env, activate the env with `poetry shell` and launch the app by running `streamlit run Home.py`.

### Deploy on Google Colab for GPU

As this application uses various deep-learning model, you could benefit from using a deployment solution that provides a GPU.  
To do so, you can leverage Google Colab free GPU to boost this Streamlit application.  
To run this app on Google Colab, simply clone the notebook called `google_colab_deploy.ipynb` into Colab and run the four cells. It will automatically download the latest code version, install dependencies and run the app. A link will appear in the output of the lat cell with a structure like `https://word1-word2-try-01-234-567-890.loca.lt`. Click it and the click continue and you’re ready to use the app!

## How to Use

Once on the demo, click on the corresponding staining analysis on the sidebar, and upload your histology image. Results will be displayed in the main area automatically.  
For HE Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_he.jpg)  
For SDH Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_sdh.jpg)

## Contact

Creator and Maintainer: [**Corentin Meyer**, 3rd year PhD Student in the CSTB Team, ICube — CNRS — Unistra](https://lambda-science.github.io/) <corentin.meyer@etu.unistra.fr>  
The source code for MyoQuant is available [HERE](https://github.com/lambda-science/MyoQuant), for the demo website it is available [HERE](https://github.com/lambda-science/MyoQuant-Streamlit)

## Citing MyoQuant🔬

[placeholder]

## Partners

<p align="center">
  <img src="https://i.imgur.com/m5OGthE.png" alt="Partner Banner" style="border-radius: 25px;" />
</p>

IMPatienT is born within the collaboration between the [CSTB Team @ ICube](https://cstb.icube.unistra.fr/en/index.php/Home) led by Julie D. Thompson, the [Morphological Unit of the Institute of Myology of Paris](https://www.institut-myologie.org/en/recherche-2/neuromuscular-investigation-center/morphological-unit/) led by Teresinha Evangelista, the [imagery platform MyoImage of Center of Research in Myology](https://recherche-myologie.fr/technologies/myoimage/) led by Bruno Cadot, [the photonic microscopy platform of the IGMBC](https://www.igbmc.fr/en/plateformes-technologiques/photonic-microscopy) led by Bertrand Vernay and the [Pathophysiology of neuromuscular diseases team @ IGBMC](https://www.igbmc.fr/en/igbmc/a-propos-de-ligbmc/directory/jocelyn-laporte) led by Jocelyn Laporte
