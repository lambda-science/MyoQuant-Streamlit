import streamlit as st
from streamlit.components.v1 import html
import matplotlib

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
import numpy as np

st.set_page_config(
    page_title="MyoQuant Breast Analysis",
    page_icon="🔬",
)

if len(list_physical_devices("GPU")) >= 1:
    use_GPU = True
else:
    use_GPU = False

with st.sidebar:
    st.write("Nuclei detection Parameters (Stardist)")
    nms_thresh = st.slider("Stardist NMS Tresh", 0.0, 1.0, 0.4, 0.1)
    prob_thresh = st.slider("Stardist Prob Tresh", 0.5, 1.0, 0.5, 0.05)
    st.write("Nuclei Classification Parameter")
    center_thresh = st.slider("Eccentricity Score Tresh", 0.0, 1.0, 0.75, 0.05)

model_cellpose = load_cellpose()
model_stardist_fluo = load_stardist_fluo()

st.title("Breast Muscle Fiber Analysis")
st.write(
    "This demo will automatically detect cells and nucleus in the image and try to quantify a certain number of features."
)

st.write("Upload your single channel cytoplasm file (tif of fluo image)")
uploaded_file_cyto = st.file_uploader("Choose a file (cyto)")
st.write("Upload your single channel nuclei file (tif of fluo image)")
uploaded_file_nuc = st.file_uploader("Choose a file (nuc)")

if uploaded_file_cyto is not None and uploaded_file_nuc is not None:
    image_ndarray_cyto = imread(uploaded_file_cyto)
    image_ndarray_nuc = imread(uploaded_file_nuc)
    mergedarrays = np.maximum(image_ndarray_cyto, image_ndarray_nuc)
    st.write("Raw Image (Cyto)")
    image_cyto = st.image(image_ndarray_cyto)
    st.write("Raw Image Nuc")
    image_nuc = st.image(image_ndarray_nuc)

    mask_cellpose = run_cellpose(image_ndarray_cyto)
    mask_stardist = run_stardist(image_ndarray_nuc, nms_thresh, prob_thresh)
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

    st.header("Full Nucleus Analysis Results")
    df_nuc_analysis = predict_all_cells(
        image_ndarray_cyto, df_cellpose, mask_stardist, center_thresh
    )
    st.dataframe(df_nuc_analysis)
    st.write("Total number of nucleus : ", df_nuc_analysis["N° Nuc"].sum())
    st.write(
        "Total number of internalized nucleus : ",
        df_nuc_analysis["N° Nuc Intern"].sum(),
        " (",
        round(
            100
            * df_nuc_analysis["N° Nuc Intern"].sum()
            / df_nuc_analysis["N° Nuc"].sum(),
            2,
        ),
        "%)",
    )
    st.write(
        "Total number of peripherical nucleus : ",
        df_nuc_analysis["N° Nuc Periph"].sum(),
        " (",
        round(
            100
            * df_nuc_analysis["N° Nuc Periph"].sum()
            / df_nuc_analysis["N° Nuc"].sum(),
            2,
        ),
        "%)",
    )
    st.write(
        "Number of cell with at least one internalized nucleus : ",
        df_nuc_analysis["N° Nuc Intern"].astype(bool).sum(axis=0),
        " (",
        round(
            100
            * df_nuc_analysis["N° Nuc Intern"].astype(bool).sum(axis=0)
            / len(df_nuc_analysis),
            2,
        ),
        "%)",
    )
    st.header("Single Nucleus Analysis Details")
    selected_fiber = st.selectbox("Select a cell", list(range(len(df_cellpose))))
    selected_fiber = int(selected_fiber)
    single_cell_img = mergedarrays[
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
    ax1.imshow(single_cell_img, cmap="gray")
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

    st.header("Painted predicted image")
    st.write(
        "Green color indicates cells with only peripherical nuclei, red color indicates cells with at least one internal nucleus (regeneration)."
    )
    painted_img = paint_histo_img(image_ndarray_cyto, df_cellpose, df_nuc_analysis)
    fig3, ax3 = plt.subplots(1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "green", "red"]
    )
    ax3.imshow(mergedarrays)
    ax3.imshow(painted_img, cmap=cmap, alpha=0.5)
    ax3.axis("off")
    st.pyplot(fig3)

html(
    f"""
    <script defer data-domain="lbgi.fr/myoquant" src="https://plausible.cmeyer.fr/js/script.js"></script>
    """
)
