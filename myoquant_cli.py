import typer
from os import path
from pathlib import Path
from rich.console import Console
from rich.table import Table
import time
from src.SDH_analysis import (
    load_cellpose,
    load_sdh_model,
    run_cellpose,
    run_cli_analysis,
    is_gpu_availiable,
)
import urllib.request
import numpy as np
from PIL import Image

try:
    from imageio.v2 import imread
except:
    from imageio import imread

console = Console()

table = Table(title="Analysis Results")
table.add_column("Feature", justify="left", style="cyan")
table.add_column("Raw Count", justify="center", style="magenta")
table.add_column("Proportion (%)", justify="right", style="green")


app = typer.Typer(
    name="MyoQuant",
    add_completion=False,
    help="MyoQuant Analysis Command Line Interface",
    pretty_exceptions_show_locals=False,
)


def check_file_exists(path):
    if path is None:
        return path
    if not path.exists():
        console.print(f"The path you've supplied {path} does not exist.", style="red")
        raise typer.Exit(code=1)
    return path


@app.command()
def run(
    image_path: Path = typer.Argument(
        ..., help="The image file path to analyse.", callback=check_file_exists
    ),
    model_path: Path = typer.Option(
        None,
        help="The SDH model path to use for analysis. Will download latest one if no path provided.",
        callback=check_file_exists,
    ),
    cellpose_path: Path = typer.Option(
        None,
        help="The pre-computed CellPose mask to use for analysis. Will run Cellpose if no path provided. Required as an image file.",
        callback=check_file_exists,
    ),
    output_path: Path = typer.Option(
        None,
        help="The path to the folder to save the results. Will save in the current folder if not specified.",
    ),
    cellpose_diameter: int = typer.Option(
        None,
        help="Approximative single cell diameter in pixel for CellPose detection. If not specified, Cellpose will try to deduce it.",
    ),
):
    """Run the SDH analysis and quantification on the image."""

    console.print(f"Welcome to the SDH Analysis CLI tools.", style="magenta")
    console.print(f"Running SDH Quantification on image : {image_path}", style="blue")
    start_time = time.time()

    if output_path is None:
        output_path = image_path.parents[0]
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    if is_gpu_availiable():
        console.print(f"GPU is available.", style="green")
    else:
        console.print(f"GPU is not available.", style="red")

    if model_path is None:
        console.print("No SDH model provided, will download latest one.", style="blue")
        if not path.exists("model.h5"):
            urllib.request.urlretrieve(
                "https://lbgi.fr/~meyer/SDH_models/model.h5", "model.h5"
            )
        console.print("SDH Model have been downloaded !", style="blue")
        model_path = "model.h5"
    else:
        console.print(f"SDH Model used: {model_path}", style="blue")

    if cellpose_path is None:
        console.print(
            "No CellPose mask provided, will run CellPose during the analysis.",
            style="blue",
        )
        model_cellpose = load_cellpose()
        console.print("CellPose Model loaded !", style="blue")
    else:
        console.print(f"CellPose mask used: {cellpose_path}", style="blue")
    console.print("Reading image...", style="blue")

    image_ndarray_sdh = imread(image_path)
    console.print("Image loaded.", style="blue")
    console.print("Starting the Analysis. This may take a while...", style="blue")
    if cellpose_path is None:
        console.print("Running CellPose...", style="blue")
        mask_cellpose = run_cellpose(
            image_ndarray_sdh, model_cellpose, cellpose_diameter
        )
        mask_cellpose = mask_cellpose.astype(np.uint16)
        cellpose_mask_filename = image_path.stem + "_cellpose_mask.tiff"
        Image.fromarray(mask_cellpose).save(output_path / cellpose_mask_filename)
        console.print(
            f"CellPose mask saved as {output_path/cellpose_mask_filename}",
            style="green",
        )
    else:
        mask_cellpose = imread(cellpose_path)

    model_SDH = load_sdh_model(model_path)
    console.print("SDH Model loaded !", style="blue")
    result_df, full_label_map = run_cli_analysis(
        image_ndarray_sdh, model_SDH, mask_cellpose
    )
    console.print("Analysis completed ! ", style="green")
    for index, row in result_df.iterrows():
        table.add_row(
            str(row[0]),
            str(row[1]),
            str(row[2]),
        )
    console.print(table)
    csv_name = image_path.stem + "_results.csv"
    result_df.to_csv(
        output_path / csv_name,
        index=False,
    )
    console.print(
        f"Table saved as a .csv file named {output_path/csv_name}", style="green"
    )
    label_map_name = image_path.stem + "_label_map.tiff"
    Image.fromarray(full_label_map).save(output_path / label_map_name)
    console.print(
        f"Labelled image saved as {output_path/label_map_name}", style="green"
    )
    painted_img_name = image_path.stem + "_painted.tiff"
    # paint_fig.savefig(output_path / painted_img_name, bbox_inches="tight")
    # console.print(
    #     f"Painted image saved as {output_path/painted_img_name}", style="green"
    # )
    console.print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    app()
