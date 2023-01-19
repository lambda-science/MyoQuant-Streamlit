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
from os import path
from skimage.measure import regionprops_table
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from tensorflow.config import list_physical_devices
import numpy as np

labels_predict = {1: "fiber type 1", 2: "fiber type 2"}

if len(list_physical_devices("GPU")) >= 1:
    use_GPU = True
else:
    use_GPU = False

np.random.seed(42)

st.set_page_config(
    page_title="MyoQuant ATP Analysis",
    page_icon="🔬",
)


@st.experimental_singleton
def load_cellpose():
    model_c = Cellpose(gpu=use_GPU, model_type="cyto2")
    return model_c


@st.experimental_memo
def run_cellpose(image):
    channel = [[0, 0]]
    mask_cellpose, flow, style, diam = model_cellpose.eval(
        image, diameter=None, channels=channel
    )
    return mask_cellpose


@st.experimental_memo
def get_all_intensity(image_array, df_cellpose):
    all_cell_median_intensity = []
    for index in range(len(df_cellpose)):
        single_cell_img = image_array[
            df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
            df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
        ].copy()

        single_cell_mask = df_cellpose.iloc[index, 9].copy()
        single_cell_img[~single_cell_mask] = 0
        # Calculate median pixel intensity of the cell but ignore 0 values
        single_cell_median_intensity = np.median(single_cell_img[single_cell_img > 0])
        all_cell_median_intensity.append(single_cell_median_intensity)
    return all_cell_median_intensity


@st.experimental_memo
def estimate_threshold(intensity_list):
    density = gaussian_kde(intensity_list)
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()

    # Create a vector of 256 values going from 0 to 256:
    xs = np.linspace(0, 255, 256)
    density_xs_values = density(xs)
    gmm = GaussianMixture(n_components=2).fit(np.array(intensity_list).reshape(-1, 1))

    # Find the x values of the two peaks
    peaks_x = np.sort(gmm.means_.flatten())
    # Find the minimum point between the two peaks
    min_index = np.argmin(density_xs_values[(xs > peaks_x[0]) & (xs < peaks_x[1])])
    threshold = peaks_x[0] + xs[min_index]

    return threshold


@st.experimental_memo
def predict_all_cells(histo_img, cellpose_df, intensity_threshold):
    all_cell_median_intensity = get_all_intensity(histo_img, cellpose_df)
    if intensity_threshold == 0:
        intensity_threshold = estimate_threshold(all_cell_median_intensity)

    muscle_fiber_type_all = [
        1 if x > intensity_threshold else 2 for x in all_cell_median_intensity
    ]
    return muscle_fiber_type_all, all_cell_median_intensity


@st.experimental_memo
def paint_full_image(image_ATP, df_cellpose, class_predicted_all):
    image_ATP_paint = np.zeros((image_ATP.shape[0], image_ATP.shape[1]), dtype=np.uint8)
    for index in range(len(df_cellpose)):
        single_cell_mask = df_cellpose.iloc[index, 9].copy()
        if class_predicted_all[index] == 1:
            image_ATP_paint[
                df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
                df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
            ][single_cell_mask] = 1
        elif class_predicted_all[index] == 2:
            image_ATP_paint[
                df_cellpose.iloc[index, 5] : df_cellpose.iloc[index, 7],
                df_cellpose.iloc[index, 6] : df_cellpose.iloc[index, 8],
            ][single_cell_mask] = 2
    return image_ATP_paint


@st.experimental_memo
def plot_density(all_cell_median_intensity, intensity_threshold):
    if intensity_threshold == 0:
        intensity_threshold = estimate_threshold(all_cell_median_intensity)
    fig, ax = plt.subplots(figsize=(10, 5))
    density = gaussian_kde(all_cell_median_intensity)
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()

    # Create a vector of 256 values going from 0 to 256:
    xs = np.linspace(0, 255, 256)
    density_xs_values = density(xs)
    ax.plot(xs, density_xs_values, label="Estimated Density")
    ax.axvline(x=intensity_threshold, color="red", label="Threshold")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


model_cellpose = load_cellpose()

with st.sidebar:
    st.write("Threshold Parameters")
    intensity_threshold = st.slider("Intensity Threshold (0=auto)", 0, 255, 0, 5)

st.title("ATP Staining Analysis")
st.write(
    "This demo will automatically quantify the number of type 1 muscle fibers vs the number of type 2 muscle fiber on ATP stained images."
)
st.write("Upload your ATP Staining image")
uploaded_file_atp = st.file_uploader("Choose a file")

if uploaded_file_atp is not None:
    image_ndarray_atp = imread(uploaded_file_atp)

    st.write("Raw Image")
    image = st.image(uploaded_file_atp)

    mask_cellpose = run_cellpose(image_ndarray_atp)

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
    st.header("Cell Intensity Plot")
    all_cell_median_intensity = get_all_intensity(image_ndarray_atp, df_cellpose)
    figure_intensity = plot_density(all_cell_median_intensity, intensity_threshold)
    st.pyplot(figure_intensity)

    st.header("ATP Cell Classification Results")

    class_predicted_all, proba_predicted_all = predict_all_cells(
        image_ndarray_atp, df_cellpose, intensity_threshold
    )

    count_per_label = np.unique(class_predicted_all, return_counts=True)
    class_and_proba_df = pd.DataFrame(
        list(zip(class_predicted_all, proba_predicted_all)),
        columns=["Muscle Fiber Type", "Intensity"],
    )
    class_and_proba_df
    st.write("Total number of cells detected: ", len(class_predicted_all))
    for index, elem in enumerate(count_per_label[0]):
        st.write(
            "Number of cells classified as ",
            labels_predict[int(elem)],
            ": ",
            count_per_label[1][int(index)],
            " ",
            100 * count_per_label[1][int(index)] / len(class_predicted_all),
            "%",
        )

    st.header("Painted predicted image")
    st.write(
        "Green color indicates cells classified as control, red color indicates cells classified as sick"
    )
    paint_img = paint_full_image(image_ndarray_atp, df_cellpose, class_predicted_all)
    fig3, ax3 = plt.subplots(1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "green", "red"]
    )
    ax3.imshow(image_ndarray_atp)
    ax3.imshow(paint_img, cmap=cmap, alpha=0.5)
    ax3.axis("off")
    st.pyplot(fig3)

html(
    f"""
    <script defer data-domain="lbgi.fr/myoquant" src="https://plausible.cmeyer.fr/js/script.js"></script>
    """
)
