import zipfile
import requests
from hocap_toolkit.utils import *

PROJ_ROOT = Path(__file__).parent.parent


def download_box_file(box_link, save_file_path):
    output_path = Path(save_file_path)
    resume_header = {}
    downloaded_size = 0

    with requests.get(box_link, headers=resume_header, stream=True) as response:
        # Check if the request was successful
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
        else:
            print(f"Failed to retrieve file info. Status code: {response.status_code}")
            return

    if output_path.exists():
        downloaded_size = output_path.stat().st_size
        # Check if there's a partial download and get its size
        resume_header = {"Range": f"bytes={downloaded_size}-"}

    # Check if the file is already fully downloaded
    if downloaded_size == total_size:
        tqdm.write(f"  ** {output_path.name} is already downloaded.")
        return

    # Send a GET request with the range header if needed
    with requests.get(box_link, headers=resume_header, stream=True) as response:
        # Check if the request was successful
        if response.status_code in [200, 206]:
            # Initialize tqdm progress bar
            with tqdm(
                total=total_size,
                initial=downloaded_size,
                unit="B",
                unit_scale=True,
                ncols=80,
            ) as pbar:
                # Download the file in chunks
                with output_path.open("ab") as file:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):  # 1 MB chunks
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
        else:
            print(f"Failed to download file. Status code: {response.status_code}")


def unzip_file(zip_file, output_dir):
    zip_file = Path(zip_file)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def main():
    dataset_files = read_data_from_yaml(PROJ_ROOT / "config/hocap_recordings.yaml")

    tqdm.write(f"- Downloading 'calibration.zip'...")
    download_box_file(
        dataset_files["calibration"], PROJ_ROOT / "datasets/calibration.zip"
    )

    tqdm.write(f"- Downloading 'models.zip'...")
    download_box_file(dataset_files["models"], PROJ_ROOT / "datasets/models.zip")

    tqdm.write(f"- Downloading 'poses.zip'...")
    download_box_file(dataset_files["poses"], PROJ_ROOT / "datasets/poses.zip")

    subject_ids = (
        [f"subject_{i}" for i in range(1, 10)]
        if args.subject_id == "all"
        else [args.subject_id]
    )

    for subject_id in subject_ids:
        tqdm.write(f"- Downloading '{subject_id}.zip'...")
        download_box_file(
            dataset_files[subject_id], PROJ_ROOT / "datasets" / f"{subject_id}.zip"
        )

    # Extract the downloaded zip files
    zip_files = list(PROJ_ROOT.glob("datasets/*.zip"))
    tqdm.write(f"- Extracting downloaded zip files...")
    for zip_file in zip_files:
        tqdm.write(f"  ** Extracting '{zip_file.name}'...")
        unzip_file(zip_file, zip_file.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset files")
    parser.add_argument(
        "--subject_id",
        type=str,
        default="all",
        choices=[
            "all",
            "subject_1",
            "subject_2",
            "subject_3",
            "subject_4",
            "subject_5",
            "subject_6",
            "subject_7",
            "subject_8",
            "subject_9",
        ],
        help="The subject id to download",
    )
    args = parser.parse_args()

    main()
