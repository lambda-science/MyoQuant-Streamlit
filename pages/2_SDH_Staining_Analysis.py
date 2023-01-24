import streamlit as st
from streamlit.components.v1 import html

try:
    from imageio.v2 import imread
except:
    from imageio import imread
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from os import path
import urllib.request
from myoquant.src.SDH_analysis import (
    predict_all_cells,
    predict_single_cell,
    paint_full_image,
)
from myoquant.src.common_func import (
    load_cellpose,
    run_cellpose,
    is_gpu_availiable,
    df_from_cellpose_mask,
    load_sdh_model,
    extract_single_image,
)

st.set_page_config(
    page_title="MyoQuant SDH Analysis",
    page_icon="🔬",
)

use_GPU = is_gpu_availiable()


@st.experimental_singleton
def st_load_sdh_model(model_path):
    return load_sdh_model(model_path)


@st.experimental_singleton
def st_load_cellpose():
    return load_cellpose()


@st.experimental_memo
def st_run_cellpose(image_ndarray, _model):
    return run_cellpose(image_ndarray, _model)


@st.experimental_memo
def st_df_from_cellpose_mask(mask):
    return df_from_cellpose_mask(mask)


@st.experimental_memo
def st_predict_all_cells(image_ndarray, cellpose_df, _model_SDH):
    return predict_all_cells(image_ndarray, cellpose_df, _model_SDH)


@st.experimental_memo
def st_extract_single_image(image_ndarray, cellpose_df, index):
    return extract_single_image(image_ndarray, cellpose_df, index)


@st.experimental_memo
def st_predict_single_cell(image_ndarray, _model_SDH):
    return predict_single_cell(image_ndarray, _model_SDH)


@st.experimental_memo
def st_paint_full_image(image_sdh, df_cellpose, class_predicted_all):
    return paint_full_image(image_sdh, df_cellpose, class_predicted_all)


labels_predict = ["control", "sick"]

tf.random.set_seed(42)
np.random.seed(42)


if path.exists("model.h5"):
    st.success("SDH Model ready to use !")
    pass
else:
    with st.spinner("Please wait we are downloading the SDH Model."):
        urllib.request.urlretrieve(
            "https://lbgi.fr/~meyer/SDH_models/model.h5", "model.h5"
        )
    st.success("SDH Model have been downloaded !")

model_cellpose = st_load_cellpose()

model_SDH = st_load_sdh_model("model.h5")

st.title("SDH Staining Analysis")
st.write(
    "This demo will automatically detect cells classify the SDH stained cell as sick or healthy using our deep-learning model."
)
st.write("Upload your SDH Staining image")
uploaded_file_sdh = st.file_uploader("Choose a file")

if uploaded_file_sdh is not None:
    image_ndarray_sdh = imread(uploaded_file_sdh)

    st.write("Raw Image")
    image = st.image(uploaded_file_sdh)

    mask_cellpose = st_run_cellpose(image_ndarray_sdh, model_cellpose)

    st.header("Segmentation Results")
    st.subheader("CellPose results")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(mask_cellpose, cmap="viridis")
    ax.axis("off")
    st.pyplot(fig)

    st.header("SDH Cell Classification Results")
    df_cellpose = st_df_from_cellpose_mask(mask_cellpose)
    df_cellpose_results = st_predict_all_cells(
        image_ndarray_sdh, df_cellpose, model_SDH
    )
    class_predicted_all = df_cellpose_results["class_predicted"].values
    proba_predicted_all = df_cellpose_results["proba_predicted"].values
    count_per_label = np.unique(class_predicted_all, return_counts=True)
    class_and_proba_df = pd.DataFrame(
        list(zip(class_predicted_all, proba_predicted_all)),
        columns=["class", "proba"],
    )
    st.dataframe(
        df_cellpose_results.drop(
            [
                "centroid-0",
                "centroid-1",
                "bbox-0",
                "bbox-1",
                "bbox-2",
                "bbox-3",
                "image",
            ],
            axis=1,
        )
    )
    st.write("Total number of cells detected: ", len(class_predicted_all))
    for elem in count_per_label[0]:
        st.write(
            "Number of cells classified as ",
            labels_predict[int(elem)],
            ": ",
            count_per_label[1][int(elem)],
            " ",
            100 * count_per_label[1][int(elem)] / len(class_predicted_all),
            "%",
        )

    st.header("Single Cell Grad-CAM")
    selected_fiber = st.selectbox("Select a cell", list(range(len(df_cellpose))))
    selected_fiber = int(selected_fiber)
    single_cell_img = st_extract_single_image(
        image_ndarray_sdh, df_cellpose, selected_fiber
    )

    grad_img, class_predicted, proba_predicted = st_predict_single_cell(
        single_cell_img, model_SDH
    )

    fig2, (ax1, ax2) = plt.subplots(1, 2)
    resized_single_cell_img = tf.image.resize(single_cell_img, (256, 256))
    ax1.imshow(single_cell_img)
    ax2.imshow(grad_img)
    ax1.axis("off")
    # ax2.axis("off")

    xlabel = (
        labels_predict[int(class_predicted)]
        + " ("
        + str(round(proba_predicted, 2))
        + ")"
    )
    ax2.set_xlabel(xlabel)
    st.pyplot(fig2)

    st.header("Painted predicted image")
    st.write(
        "Green color indicates cells classified as control, red color indicates cells classified as sick"
    )
    paint_img = st_paint_full_image(image_ndarray_sdh, df_cellpose, class_predicted_all)
    fig3, ax3 = plt.subplots(1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "green", "red"]
    )
    ax3.imshow(image_ndarray_sdh)
    ax3.imshow(paint_img, cmap=cmap, alpha=0.5)
    ax3.axis("off")
    st.pyplot(fig3)

html(
    f"""
    <script defer data-domain="lbgi.fr/myoquant" src="https://plausible.cmeyer.fr/js/script.js"></script>
    """
)
