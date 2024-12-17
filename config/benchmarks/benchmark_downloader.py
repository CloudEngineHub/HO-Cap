import requests
from hocap_toolkit.utils import *

PROJ_ROOT = Path(__file__).parent.parent.parent


def download_box_file(box_link, output_file):
    output_path = Path(output_file)
    file_name = output_file.name

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
        tqdm.write(f"  ** {file_name} is already downloaded.")
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


if __name__ == "__main__":
    behchmark_data = read_data_from_yaml("config/hocap_benchmarks.yaml")

    for file_name, file_link in behchmark_data.items():
        tqdm.write(f"- Downloading {file_name}...")
        if "demo" in file_name:
            save_path = PROJ_ROOT / "results" / f"{file_name}.json"
        else:
            save_path = PROJ_ROOT / "config" / "benchmarks" / f"{file_name}.json"
        download_box_file(file_link, save_path)
