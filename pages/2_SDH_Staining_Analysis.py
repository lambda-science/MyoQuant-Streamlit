import streamlit as st
from streamlit.components.v1 import html
from cellpose.core import use_gpu
from cellpose.models import Cellpose

try:
    from imageio.v2 import imread
except:
    from imageio import imread
from skimage.measure import regionprops_table
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from stardist import random_label_cmap
import tensorflow as tf
from tensorflow.config import list_physical_devices
from tensorflow import keras

from gradcam import *
from os import path
import urllib.request
from random_brightness import *

labels_predict = ["control", "sick"]

tf.random.set_seed(42)
np.random.seed(42)

st.set_page_config(
    page_title="MyoQuant SDH Analysis",
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
    model_sdh = keras.models.load_model(
        "model.h5", custom_objects={"RandomBrightness": RandomBrightness}
    )
    return model_sdh


@st.experimental_memo
def run_cellpose(image):
    channel = [[0, 0]]
    mask_cellpose, flow, style, diam = model_cellpose.eval(
        image, diameter=None, channels=channel
    )
    return mask_cellpose


@st.experimental_memo
def predict_single_cell(single_cell_img, _model_SDH):
    img_array = np.empty((1, 256, 256, 3))
    img_array[0] = tf.image.resize(single_cell_img, (256, 256))
    prediction = _model_SDH.predict(img_array)
    predicted_class = prediction.argmax()
    predicted_proba = round(np.amax(prediction), 2)
    heatmap = make_gradcam_heatmap(
        img_array, _model_SDH.get_layer("resnet50v2"), "conv5_block3_3_conv"
    )
    grad_cam_img = save_and_display_gradcam(img_array[0], heatmap)
    return grad_cam_img, predicted_class, predicted_proba


@st.experimental_memo
def resize_batch_cells(histo_img, cellpose_df):
    img_array_full = np.empty((len(cellpose_df), 256, 256, 3))
    for index in range(len(cellpose_df)):
        single_cell_img = histo_img[
            cellpose_df.iloc[index, 5] : cellpose_df.iloc[index, 7],
            cellpose_df.iloc[index, 6] : cellpose_df.iloc[index, 8],
        ].copy()

        single_cell_mask = cellpose_df.iloc[index, 9].copy()
        single_cell_img[~single_cell_mask] = 0

        img_array_full[index] = tf.image.resize(single_cell_img, (256, 256))
    return img_array_full


@st.experimental_memo
def predict_all_cells(histo_img, cellpose_df, _model_SDH):
    predicted_class_array = np.empty((len(cellpose_df)))
    predicted_proba_array = np.empty((len(cellpose_df)))
    img_array_full = resize_batch_cells(histo_img, cellpose_df)
    prediction = _model_SDH.predict(img_array_full)
    index_counter = 0
    for prediction_result in prediction:
        predicted_class_array[index_counter] = prediction_result.argmax()
        predicted_proba_array[index_counter] = np.amax(prediction_result)
        index_counter += 1
    return predicted_class_array, predicted_proba_array


@st.experimental_memo
def paint_full_image(image_sdh, df_cellpose, class_predicted_all):
    image_sdh_paint = np.zeros((image_sdh.shape[0], image_sdh.shape[1]), dtype=np.uint8)
    for index in range(len(df_cellpose)):
        single_cell_mask = df_cellpose.iloc[index, 9].copy()
        if class_predicted_all[index] == 0:
            image_sdh_paint[
                df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
                df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
            ][single_cell_mask] = 1
        elif class_predicted_all[index] == 1:
            image_sdh_paint[
                df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
                df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
            ][single_cell_mask] = 2
    return image_sdh_paint


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

    class_predicted_all, proba_predicted_all = predict_all_cells(
        image_ndarray_sdh, df_cellpose, model_SDH
    )

    count_per_label = np.unique(class_predicted_all, return_counts=True)
    class_and_proba_df = pd.DataFrame(
        list(zip(class_predicted_all, proba_predicted_all)),
        columns=["class", "proba"],
    )
    class_and_proba_df
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
    single_cell_img = image_ndarray_sdh[
        df_cellpose.iloc[selected_fiber, 5] : df_cellpose.iloc[selected_fiber, 7],
        df_cellpose.iloc[selected_fiber, 6] : df_cellpose.iloc[selected_fiber, 8],
    ].copy()

    single_cell_mask = df_cellpose.iloc[selected_fiber, 9].copy()
    single_cell_img[~single_cell_mask] = 0

    grad_img, class_predicted, proba_predicted = predict_single_cell(
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
    paint_img = paint_full_image(image_ndarray_sdh, df_cellpose, class_predicted_all)
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
