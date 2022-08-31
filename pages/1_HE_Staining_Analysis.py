import streamlit as st
from cellpose import models, core
from stardist.models import StarDist2D
from csbdeep.utils import normalize

try:
    from imageio.v2 import imread
except:
    from imageio import imread
from skimage.measure import regionprops_table
import pandas as pd
import matplotlib.pyplot as plt
from stardist import random_label_cmap
from tensorflow.config import list_physical_devices
from draw_line import *
from skimage.draw import line

st.set_page_config(
    page_title="HistoQuant-Streamlit",
    page_icon="🔬",
)

if len(list_physical_devices("GPU")) >= 1:
    use_GPU = True
else:
    use_GPU = False


@st.experimental_singleton
def load_cellpose():
    model_c = models.Cellpose(gpu=use_GPU, model_type="cyto2")
    return model_c


@st.experimental_singleton
def load_stardist():
    model_s = StarDist2D.from_pretrained("2D_versatile_he")
    return model_s


@st.experimental_memo
def run_cellpose(image):
    channel = [[0, 0]]
    mask_cellpose, flow, style, diam = model_cellpose.eval(
        image, diameter=None, channels=channel
    )
    return mask_cellpose


@st.experimental_memo
def run_stardist(image, nms_thresh=0.4, prob_thresh=0.5):
    img_norm = image / 255
    img_norm = normalize(img_norm, 1, 99.8)
    mask_stardist, details = model_stardist.predict_instances(
        img_norm, nms_thresh=nms_thresh, prob_thresh=prob_thresh
    )
    return mask_stardist


with st.sidebar:
    st.write("Models Parameters")
    nms_thresh = st.slider("Stardist NMS Tresh", 0.0, 1.0, 0.4, 0.1)
    prob_thresh = st.slider("Stardist Prob Tresh", 0.5, 1.0, 0.5, 0.05)

model_cellpose = load_cellpose()
model_stardist = load_stardist()

st.title("HE Staining Analysis")
st.write(
    "This demo will automatically detect cells and nucleus in the image and try to quantify a certain number of features."
)
st.write("Upload your HE Staining image")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    image_ndarray = imread(uploaded_file)
    st.write("Raw Image")
    image = st.image(uploaded_file)

    mask_cellpose = run_cellpose(image_ndarray)
    mask_stardist = run_stardist(image_ndarray, nms_thresh, prob_thresh)
    mask_stardist_copy = mask_stardist.copy()

    st.header("Segmentation Results")
    st.subheader("CellPose and Stardist overlayed results")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(mask_cellpose, cmap="viridis")
    lbl_cmap = random_label_cmap()
    ax.imshow(mask_stardist, cmap=lbl_cmap, alpha=0.5)
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

    selected_fiber = st.selectbox("Select a cell", list(range(len(df_cellpose))))
    selected_fiber = int(selected_fiber)
    single_cell_img = image_ndarray[
        df_cellpose.iloc[selected_fiber, 5] : df_cellpose.iloc[selected_fiber, 7],
        df_cellpose.iloc[selected_fiber, 6] : df_cellpose.iloc[selected_fiber, 8],
    ]
    nucleus_single_cell_img = mask_stardist_copy[
        df_cellpose.iloc[selected_fiber, 5] : df_cellpose.iloc[selected_fiber, 7],
        df_cellpose.iloc[selected_fiber, 6] : df_cellpose.iloc[selected_fiber, 8],
    ]
    single_cell_mask = df_cellpose.iloc[selected_fiber, 9]
    single_cell_img[~single_cell_mask] = 0
    nucleus_single_cell_img[~single_cell_mask] = 0

    props_nuc_single = regionprops_table(
        nucleus_single_cell_img,
        intensity_image=single_cell_img,
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
    df_nuc_single = pd.DataFrame(props_nuc_single)
    st.markdown(
        """
        * White point represent cell centroid. 
        * Green point represent nucleus centroid. Green dashed line represent the fiber centrer - nucleus distance. 
        * Red point represent the cell border from a straight line between the cell centroid and the nucleus centroid. The red dashed line represent distance between the nucelus and the cell border. 
        * The periphery ratio is calculated by the division of the distance centroid - nucleus and the distance centroid - cell border."""
    )
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(single_cell_img)
    ax2.imshow(nucleus_single_cell_img, cmap="viridis")
    # Plot Fiber centroid
    x_fiber = df_cellpose.iloc[selected_fiber, 3] - df_cellpose.iloc[selected_fiber, 6]
    y_fiber = df_cellpose.iloc[selected_fiber, 2] - df_cellpose.iloc[selected_fiber, 5]
    ax3.scatter(x_fiber, y_fiber, color="white")
    # Plot nucleus centroid
    for index, value in df_nuc_single.iterrows():
        ax3.scatter(value[3], value[2], color="blue", s=2)
        # Extend line and find closest point
        m, b = line_equation(x_fiber, y_fiber, value[3], value[2])

        intersections_lst = calculate_intersection(
            m, b, (single_cell_img.shape[0], single_cell_img.shape[1])
        )
        border_point = calculate_closest_point(value[3], value[2], intersections_lst)
        ax3.plot(
            (x_fiber, border_point[0]),
            (y_fiber, border_point[1]),
            "ro--",
            linewidth=1,
            markersize=1,
        )
        ax3.plot(
            (x_fiber, value[3]),
            (y_fiber, value[2]),
            "go--",
            linewidth=1,
            markersize=1,
        )

        rr, cc = line(
            int(y_fiber),
            int(x_fiber),
            int(border_point[1]),
            int(border_point[0]),
        )
        for index, coords in enumerate(list(zip(rr, cc))):
            try:
                if single_cell_mask[coords] == 0:
                    dist_nuc_cent = calculate_distance(
                        x_fiber, y_fiber, value[3], value[2]
                    )
                    dist_out_of_fiber = calculate_distance(
                        x_fiber, y_fiber, coords[1], coords[0]
                    )
                    ratio_dist = dist_nuc_cent / dist_out_of_fiber
                    ax3.scatter(coords[1], coords[0], color="red", s=10)
                    break
            except IndexError:
                coords = list(zip(rr, cc))[index - 1]
                dist_nuc_cent = calculate_distance(x_fiber, y_fiber, value[3], value[2])
                dist_out_of_fiber = calculate_distance(
                    x_fiber, y_fiber, coords[1], coords[0]
                )
                ratio_dist = dist_nuc_cent / dist_out_of_fiber
                ax3.scatter(coords[1], coords[0], color="red", s=10)
                break

        st.write("Nucleus #{} has a periphery ratio of: {}".format(index, ratio_dist))
    ax3.imshow(single_cell_img)
    ax3.imshow(nucleus_single_cell_img, cmap="viridis", alpha=0.5)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    st.pyplot(fig2)

    st.subheader("All nucleus inside selected cell")

    st.dataframe(df_nuc_single.drop("image", axis=1))
