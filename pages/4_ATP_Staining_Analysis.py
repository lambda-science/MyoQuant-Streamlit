import streamlit as st
from streamlit.components.v1 import html

try:
    from imageio.v2 import imread
except:
    from imageio import imread
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table
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
