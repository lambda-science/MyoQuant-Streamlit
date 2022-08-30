import streamlit as st
from cellpose.core import use_gpu
from cellpose.models import Cellpose
from imageio.v2 import imread
from skimage.measure import regionprops_table
import pandas as pd
import matplotlib.pyplot as plt
from stardist import random_label_cmap
import tensorflow as tf
from tensorflow.config import list_physical_devices
from tensorflow import keras

from gradcam import *
from os import path
import urllib.request

labels_predict = ["control", "sick"]

st.set_page_config(
    page_title="HistoQuant-Streamlit",
    page_icon="🔬",
)

if path.exists("model.h5"):
    st.success("SDH Model ready to use !")
    pass
else:
    with st.spinner("Please wait we are downloading the SDH Model."):
        urllib.request.urlretrieve(
            "https://lbgi.fr/~meyer/SDH_models/model.h5", "model.h5"
        )
    st.success("SDH Model have been downloaded !")

if len(list_physical_devices("GPU")) >= 1:
    use_GPU = True
else:
    use_GPU = False


@st.experimental_singleton
def load_cellpose():
    model_c = Cellpose(gpu=use_GPU, model_type="cyto2")
    return model_c


@st.experimental_singleton
def load_sdh_model():
    model_sdh = keras.models.load_model("model.h5")
    return model_sdh


@st.experimental_memo
def run_cellpose(image):
    channel = [[0, 0]]
    mask_cellpose, flow, style, diam = model_cellpose.eval(
        image, diameter=None, channels=channel
    )
    return mask_cellpose


@st.experimental_memo
def predict_single_cell(single_cell_img):
    img_array = np.empty((1, 256, 256, 3))
    # img_array[0] = (
    #    keras.preprocessing.image.smart_resize(single_cell_img, (256, 256)) / 255.0
    # )
    img_array[0] = tf.image.resize(single_cell_img, (256, 256)) / 255.0
    predicted_class = model_SDH.predict(img_array * 255).argmax()
    predicted_proba = round(np.amax(model_SDH.predict(img_array * 255)), 2)
    heatmap = make_gradcam_heatmap(
        img_array, model_SDH.get_layer("resnet50v2"), "conv5_block3_3_conv"
    )
    grad_cam_img = save_and_display_gradcam(img_array[0], heatmap)
    return grad_cam_img, predicted_class, predicted_proba


@st.experimental_memo
def predict_all_cells(histo_img, cellpose_mask, cellpose_df):
    img_array = np.empty((len(cellpose_df), 256, 256, 3))
    grad_cam_array = []
    single_img_array = np.empty((1, 256, 256, 3))
    predicted_class_array = np.empty((len(cellpose_df)))
    predicted_proba_array = np.empty((len(cellpose_df)))
    my_bar = st.progress(0)
    for index in range(len(cellpose_df)):
        single_cell_img = image_ndarray_sdh[
            cellpose_df.iloc[index, 5] : cellpose_df.iloc[index, 7],
            cellpose_df.iloc[index, 6] : cellpose_df.iloc[index, 8],
        ]

        single_cell_mask = cellpose_df.iloc[index, 9]
        single_cell_img[~single_cell_mask] = 0

        # img_array[index] = (
        #     keras.preprocessing.image.smart_resize(single_cell_img, (256, 256)) / 255.0
        # )
        img_array[index] = tf.image.resize(single_cell_img, (256, 256)) / 255.0
        single_img_array[0] = img_array[index]
        heatmap = make_gradcam_heatmap(
            single_img_array, model_SDH.get_layer("resnet50v2"), "conv5_block3_3_conv"
        )
        grad_cam_array.append(save_and_display_gradcam(img_array[index], heatmap))
        predicted_class_array[index] = model_SDH.predict(
            single_img_array * 255
        ).argmax()
        predicted_proba_array[index] = np.amax(
            model_SDH.predict(single_img_array * 255)
        )
        my_bar.progress((index + 1 / (len(cellpose_df) / 100)) / 100)
    return grad_cam_array, predicted_class_array, predicted_proba_array


model_cellpose = load_cellpose()

model_SDH = load_sdh_model()

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

    mask_cellpose = run_cellpose(image_ndarray_sdh)

    st.header("Segmentation Results")
    st.subheader("CellPose results")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(mask_cellpose, cmap="viridis")
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("All cells detected by CellPose")

    props_cellpose = regionprops_table(
        mask_cellpose,
        properties=[
            "label",
            "area",
            "centroid",
            "eccentricity",
            "bbox",
            "image",
            "perimeter",
        ],
    )
    df_cellpose = pd.DataFrame(props_cellpose)
    st.dataframe(df_cellpose.drop("image", axis=1))

    st.header("SDH Cell Classification Results")

    grad_img_all, class_predicted_all, proba_predicted_all = predict_all_cells(
        image_ndarray_sdh, mask_cellpose, df_cellpose
    )

    count_per_label = np.unique(class_predicted_all, return_counts=True)
    label_count = dict()

    for index, label in enumerate(labels_predict):
        label_count[count_per_label[0][index]] = count_per_label
    class_predicted_all
    st.write("Total number of cells detected: ", len(class_predicted_all))
    st.write(
        "Number of cells classified as control: ",
        count_per_label[1][0],
        " ",
        100 * count_per_label[1][0] / len(class_predicted_all),
        "%",
    )
    st.write(
        "Number of cells classified as sick: ",
        count_per_label[1][1],
        " ",
        100 * count_per_label[1][1] / len(class_predicted_all),
        "%",
    )

    st.header("Single Cell Grad-CAM")
    selected_fiber = st.selectbox("Select a cell", list(range(len(df_cellpose))))
    selected_fiber = int(selected_fiber)
    single_cell_img = image_ndarray_sdh[
        df_cellpose.iloc[selected_fiber, 5] : df_cellpose.iloc[selected_fiber, 7],
        df_cellpose.iloc[selected_fiber, 6] : df_cellpose.iloc[selected_fiber, 8],
    ]

    single_cell_mask = df_cellpose.iloc[selected_fiber, 9]
    single_cell_img[~single_cell_mask] = 0

    # grad_img, class_predicted, proba_predicted = predict_single_cell(single_cell_img)

    fig2, (ax1, ax2) = plt.subplots(1, 2)
    # resized_single_cell_img = keras.preprocessing.image.smart_resize(
    #     single_cell_img, (256, 256)
    # )
    resized_single_cell_img = tf.image.resize(single_cell_img, (256, 256))
    ax1.imshow(single_cell_img)
    ax2.imshow(grad_img_all[selected_fiber])
    ax1.axis("off")
    # ax2.axis("off")

    xlabel = (
        labels_predict[int(class_predicted_all[selected_fiber])]
        + " ("
        + str(round(proba_predicted_all[selected_fiber], 2))
        + ")"
    )
    ax2.set_xlabel(xlabel)
    st.pyplot(fig2)
