from .common_imports import *


def make_clean_folder(folder_path: Union[str, Path]) -> None:
    """Delete the folder if it exists and create a new one."""
    if Path(folder_path).is_dir():
        shutil.rmtree(str(folder_path))
    try:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create folder '{folder_path}': {e}")


def read_data_from_json(file_path: Union[str, Path]) -> Any:
    """Read data from a JSON file and return it."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(str(file_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON from {file_path}: {e}")


def write_data_to_json(file_path: Union[str, Path], data: Union[list, Dict]) -> None:
    """Write data to a JSON file."""
    try:
        with open(str(file_path), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
    except IOError as e:
        raise IOError(f"Failed to write JSON data to {file_path}: {e}")


def read_data_from_yaml(file_path: Union[str, Path]) -> Any:
    """Read data from a YAML file and return it."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(str(file_path), "r", encoding="utf-8") as f:
            return yaml.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading YAML file from {file_path}: {e}")


def write_data_to_yaml(file_path: Union[str, Path], data: Any) -> None:
    """Write data to a YAML file."""
    try:
        with open(str(file_path), "w", encoding="utf-8") as f:
            yaml.dump(data, f)
    except IOError as e:
        raise IOError(f"Failed to write YAML data to {file_path}: {e}")


def read_rgb_image(file_path: Union[str, Path]) -> np.ndarray:
    """Read an RGB image from the specified file path."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Image file '{file_path}' does not exist.")
    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Failed to load image from '{file_path}'.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_rgb_image(file_path: Union[str, Path], image: np.ndarray) -> None:
    """Write an RGB image to the specified file path."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with 3 channels.")
    success = cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError(f"Failed to write RGB image to '{file_path}'.")


def read_depth_image(file_path: Union[str, Path], scale: float = 1.0) -> np.ndarray:
    """Read a depth image from the specified file path."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Depth image file '{file_path}' does not exist.")
    image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise ValueError(f"Failed to load depth image from '{file_path}'.")
    image = image.astype(np.float32) / scale
    return image


def write_depth_image(file_path: Union[str, Path], image: np.ndarray) -> None:
    """Write a depth image to the specified file path."""
    if image.dtype not in [np.uint16, np.uint8]:
        raise ValueError("Depth image must be of type uint16 or uint8.")
    success = cv2.imwrite(str(file_path), image)
    if not success:
        raise ValueError(f"Failed to write depth image to '{file_path}'.")


def read_mask_image(file_path: Union[str, Path]) -> np.ndarray:
    """Read a mask image from the specified file path."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Mask image file '{file_path}' does not exist.")
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load mask image from '{file_path}'.")
    return image


def write_mask_image(file_path: Union[str, Path], image: np.ndarray) -> None:
    """Write a mask image to the specified file path."""
    success = cv2.imwrite(str(file_path), image)
    if not success:
        raise ValueError(f"Failed to write mask image to '{file_path}'.")
