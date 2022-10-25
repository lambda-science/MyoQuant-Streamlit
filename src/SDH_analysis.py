import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from cellpose.core import use_gpu
from cellpose.models import Cellpose
from skimage.measure import regionprops_table
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.config import list_physical_devices
from tensorflow import keras

from gradcam import *
from os import path
from random_brightness import *

labels_predict = ["control", "sick"]
tf.random.set_seed(42)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(42)
if len(list_physical_devices("GPU")) >= 1:
    use_GPU = True
else:
    use_GPU = False


def is_gpu_availiable():
    return use_GPU


def load_cellpose():
    model_c = Cellpose(gpu=use_GPU, model_type="cyto2")
    return model_c


def load_sdh_model(model_path):
    model_sdh = keras.models.load_model(
        model_path, custom_objects={"RandomBrightness": RandomBrightness}
    )
    return model_sdh


def run_cellpose(image, model_cellpose):
    channel = [[0, 0]]
    mask_cellpose, flow, style, diam = model_cellpose.eval(
        image, diameter=None, channels=channel
    )
    return mask_cellpose


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


def run_cli_analysis(image_array, model_SDH, mask_cellpose):
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
    class_predicted_all, proba_predicted_all = predict_all_cells(
        image_array, df_cellpose, model_SDH
    )

    count_per_label = np.unique(class_predicted_all, return_counts=True)
    class_and_proba_df = pd.DataFrame(
        list(zip(class_predicted_all, proba_predicted_all)),
        columns=["class", "proba"],
    )

    # Result table dict
    results_classification_dict = {}
    results_classification_dict["Muscle Fibers"] = [len(class_predicted_all), 100]
    for elem in count_per_label[0]:
        results_classification_dict[labels_predict[int(elem)]] = [
            count_per_label[1][int(elem)],
            100 * count_per_label[1][int(elem)] / len(class_predicted_all),
        ]

    # Paint The Full Image
    full_label_map = paint_full_image(image_array, df_cellpose, class_predicted_all)
    # paint_fig, ax3 = plt.subplots(1, 1)
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    #     "", ["white", "green", "red"]
    # )
    # ax3.imshow(image_array)
    # ax3.imshow(full_label_map, cmap=cmap, alpha=0.5)
    # ax3.axis("off")

    return results_classification_dict, full_label_map

    # # Paint The Grad Cam
    # selected_fiber = st.selectbox("Select a cell", list(range(len(df_cellpose))))
    # selected_fiber = int(selected_fiber)
    # single_cell_img = image_ndarray_sdh[
    #     df_cellpose.iloc[selected_fiber, 5] : df_cellpose.iloc[selected_fiber, 7],
    #     df_cellpose.iloc[selected_fiber, 6] : df_cellpose.iloc[selected_fiber, 8],
    # ].copy()

    # single_cell_mask = df_cellpose.iloc[selected_fiber, 9].copy()
    # single_cell_img[~single_cell_mask] = 0

    # grad_img, class_predicted, proba_predicted = predict_single_cell(
    #     single_cell_img, model_SDH
    # )
