from .common_imports import *
from .color_info import (
    OBJ_CLASS_COLORS,
    HAND_COLORS,
    HAND_BONE_COLORS,
    HAND_JOINT_COLORS,
    HO_CAP_SEG_COLOR,
)
from .mano_info import HAND_BONES


def _apply_morphology(
    mask: np.ndarray, operation: str, kernel_size: int = 3, iterations: int = 1
) -> np.ndarray:
    """Helper function to apply a morphological operation (erode/dilate) on the mask."""
    if mask.ndim not in [2, 3]:
        raise ValueError("Mask must be a 2D or 3D numpy array.")
    if kernel_size <= 1:
        raise ValueError("Kernel size must be greater than 1.")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dtype = mask.dtype
    mask = mask.astype(np.uint8)
    if operation == "erode":
        result = cv2.erode(mask, kernel, iterations=iterations)
    elif operation == "dilate":
        result = cv2.dilate(mask, kernel, iterations=iterations)
    else:
        raise ValueError(f"Invalid operation: {operation}. Use 'erode' or 'dilate'.")
    return result.astype(mask_dtype)


def _plot_image(ax, image, name, facecolor, titlecolor, fontsize):
    """Helper function to plot an image in the grid."""
    if image.ndim == 3 and image.shape[2] == 3:  # RGB image
        ax.imshow(image)
    elif image.ndim == 2 and image.dtype == np.uint8:  # Grayscale/mask image
        unique_values = np.unique(image)
        cmap = "tab10" if len(unique_values) <= 10 else "gray"
        ax.imshow(image, cmap=cmap)
    elif image.ndim == 2 and image.dtype == bool:  # Binary image
        ax.imshow(image, cmap="gray")
    else:  # Depth or other image
        ax.imshow(image, cmap="viridis")

    if name:
        ax.text(
            5,
            5,
            name,
            fontsize=fontsize,
            color=titlecolor,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor=facecolor, alpha=0.5, edgecolor="none", pad=3),
        )


def erode_mask(
    mask: np.ndarray, kernel_size: int = 3, iterations: int = 1
) -> np.ndarray:
    """Apply erosion to the mask."""
    return _apply_morphology(
        mask, operation="erode", kernel_size=kernel_size, iterations=iterations
    )


def dilate_mask(
    mask: np.ndarray, kernel_size: int = 3, iterations: int = 1
) -> np.ndarray:
    """Apply dilation to the mask."""
    return _apply_morphology(
        mask, operation="dilate", kernel_size=kernel_size, iterations=iterations
    )


def get_depth_colormap(image: np.ndarray) -> np.ndarray:
    """Convert a depth image to a colormap representation."""
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    d_min, d_max = image.min(), image.max()
    if d_min == d_max:
        return np.zeros_like(image, dtype=np.uint8)
    # Normalize the depth image to range [0, 255]
    img = (image - d_min) / (d_max - d_min) * 255
    img = img.astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def draw_image_overlay(
    rgb_image: np.ndarray, overlay_image: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Draw an overlay image on top of an RGB image."""
    return cv2.addWeighted(rgb_image, 1 - alpha, overlay_image, alpha, 0)


def draw_image_grid(
    images,
    names=None,
    figsize=(1920, 1080),
    max_cols=4,
    facecolor="white",
    titlecolor="black",
    fontsize=12,
    bar_width=0.2,
):
    """Display a list of images in a grid and draw the title name on each image's top-left corner."""
    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images provided to display.")
    num_cols = min(num_images, max_cols)
    num_rows = (num_images + num_cols - 1) // num_cols
    # Default to no names if not provided
    if names is None or len(names) != num_images:
        names = [None] * num_images
    # Create figure and axis grid
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(figsize[0] / 100.0, figsize[1] / 100.0),
        dpi=100,
        facecolor=facecolor,
    )
    axs = np.atleast_1d(axs).flat  # Ensure axs is always iterable
    # Plot each image
    for i, (image, name) in enumerate(zip(images, names)):
        _plot_image(axs[i], image, name, facecolor, titlecolor, fontsize)
        axs[i].axis("off")
    # Hide unused axes
    for ax in axs[i + 1 :]:
        ax.axis("off")
    # Adjust layout and spacing
    plt.tight_layout(pad=bar_width, h_pad=bar_width, w_pad=bar_width)
    # Convert the figure to an RGB array
    fig.canvas.draw()
    rgb_image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    # Close the figure
    plt.close(fig)
    return rgb_image


def draw_hand_landmarks(rgb_image, landmarks, hand_side=None, box=None):
    """Draw hand landmarks on an image."""
    img = rgb_image.copy()
    # draw bones
    for idx, bone in enumerate(HAND_BONES):
        if np.any(landmarks[bone[0]] == -1) or np.any(landmarks[bone[1]] == -1):
            continue
        cv2.line(
            img,
            landmarks[bone[0]],
            landmarks[bone[1]],
            HAND_BONE_COLORS[idx].rgb,
            2,
        )
    # draw joints
    for idx, mark in enumerate(landmarks):
        if np.any(mark == -1):
            continue
        cv2.circle(img, mark, 5, [255, 255, 255], -1)
        cv2.circle(
            img,
            mark,
            3,
            HAND_JOINT_COLORS[idx].rgb,
            -1,
        )

    # draw hand box
    if box is not None:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # draw hand side text
    if hand_side is not None:
        text = hand_side.lower()
        text_x = np.min(landmarks[:, 0])
        text_y = np.min(landmarks[:, 1]) - 5  # add margin to top
        text_color = HAND_COLORS[1] if text == "right" else HAND_COLORS[2]
        cv2.putText(
            img,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            text_color.rgb,
            1,
            cv2.LINE_AA,
        )
    return img


def draw_all_camera_images(
    images,
    names=None,
    figsize=(1920, 1080),
    facecolor="white",
    titlecolor="black",
    fontsize=12,
    bar_width=0.2,
    show_only=False,
):
    """Draw nine images in a grid (8 from RealSense cameras and 1 from HoloLens) in a 3x4 layout.

    Args:
        images (list of np.ndarray): List of 9 images to be displayed.
        names (list of str, optional): List of image names to display on top-left. Defaults to None.
        figsize (tuple, optional): Figure size in pixels. Defaults to (1920, 1080).
        facecolor (str, optional): Background color of the figure. Defaults to "white".
        titlecolor (str, optional): Color of the image titles. Defaults to "black".
        fontsize (int, optional): Font size for the image titles. Defaults to 12.
        bar_width (float, optional): Padding between subplots. Defaults to 0.2.

    Returns:
        np.ndarray: The final figure rendered as an RGB image.
    """
    num_images = len(images)
    if num_images != 9:
        raise ValueError(f"Expected exactly 9 images, but got {num_images}.")
    if len(names) != num_images:
        raise ValueError(
            f"Number of 'names' must match the number of images. Expected 9, but got {len(names)}."
        )
    if names is None:
        names = [None] * num_images
    fig = plt.figure(
        figsize=(figsize[0] / 100.0, figsize[1] / 100.0), dpi=100, facecolor=facecolor
    )
    gs = GridSpec(3, 4, figure=fig)
    # Plot the first eight images in a 2x4 grid
    for i in range(8):
        row, col = divmod(i, 4)  # Divide by 4 to get row, modulo 4 to get column
        ax = fig.add_subplot(gs[row, col])
        _plot_image(ax, images[i], names[i], facecolor, titlecolor, fontsize)
        ax.axis("off")
    # Plot the ninth image in the third row, spanning columns 1 and 2
    center_ax = fig.add_subplot(gs[2, 1:3])
    _plot_image(center_ax, images[8], names[8], facecolor, titlecolor, fontsize)
    center_ax.axis("off")
    # Adjust layout and spacing
    plt.tight_layout(pad=bar_width, h_pad=bar_width, w_pad=bar_width)
    if show_only:
        plt.show()
        plt.close(fig)
        return

    # Convert figure to RGB image
    fig.canvas.draw()
    rgb_image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    # Close the figure to free memory
    plt.close(fig)
    return rgb_image


def get_rgb_difference(rgb1, rgb2, scale=255.0):
    """Compute L2 error between RGB 1 and RGB2."""
    # Convert to float32 and normalize
    im1 = rgb1.astype(np.float32) / scale
    im2 = rgb2.astype(np.float32) / scale
    # Compute the normalized L2 error
    diff = np.sqrt(np.mean((im1 - im2) ** 2))
    return diff


def get_mask_iou(mask1, mask2):
    """Compute Intersection over Union (IoU) between two binary masks."""
    # Convert to boolean masks
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    # Compute intersection and union
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    # Calculate IoU score
    score = intersection / union if union != 0 else 0.0
    return score


def get_mask_dice_coefficient(mask1, mask2):
    """Compute Dice coefficient between two binary masks."""
    # Convert to boolean masks
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    # Compute intersection and sum of masks
    intersection = np.logical_and(m1, m2).sum()
    sum_masks = m1.sum() + m2.sum()
    # Calculate Dice coefficient
    score = 2 * intersection / sum_masks if sum_masks != 0 else 0.0
    return score


def create_video_from_rgb_images(
    file_path: Union[str, Path], rgb_images: List[np.ndarray], fps: int = 30
) -> None:
    """Create a video from a list of RGB images."""
    if not rgb_images:
        raise ValueError("The list of RGB images is empty.")
    height, width = rgb_images[0].shape[:2]
    container = None
    try:
        container = av.open(str(file_path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.thread_type = "FRAME"  # Parallel processing of frames
        stream.thread_count = os.cpu_count()  # Number of threads to use
        for image in rgb_images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as e:
        raise IOError(f"Failed to write video to '{file_path}': {e}")
    finally:
        if container:
            container.close()


def create_video_from_depth_images(
    file_path: Union[str, Path], depth_images: list[np.ndarray], fps: int = 30
) -> None:
    """Create a video from a list of depth images."""
    # Validate image dimensions
    height, width = depth_images[0].shape[:2]
    container = None
    try:
        container = av.open(str(file_path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.thread_type = "FRAME"  # Parallel processing of frames
        stream.thread_count = os.cpu_count()  # Number of threads to use

        for depth_image in depth_images:
            image = get_depth_colormap(depth_image)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as e:
        raise IOError(f"Failed to write video to '{file_path}': {e}")
    finally:
        if container:
            container.close()


def create_video_from_image_files(
    file_path: Union[str, Path],
    image_files: List[Union[str, Path]],
    fps: int = 30,
    preload: bool = False,
) -> None:
    """Create a video from a list of image files (RGB or Depth images).

    Args:
        file_path (str | Path): Path to save the output video.
        image_files (list[str | Path]): List of image file paths.
        fps (int, optional): Frames per second for the video. Defaults to 30.
        preload (bool, optional): Preload all images into memory before creating the video. Defaults to False.
    """

    def worker_read_image_file(image_file):
        """Helper to read the image file, handle depth images, and return an RGB image."""
        img = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image file: {image_file}")
        # If depth image (2D), apply colormap, otherwise assume it's an RGB image
        if img.ndim == 2:
            img = get_depth_colormap(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not image_files:
        raise ValueError("The list of image files is empty.")

    # Load all images into memory if preload is True
    if preload:
        images = [None] * len(image_files)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(worker_read_image_file, image_file): i
                for i, image_file in enumerate(image_files)
            }
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    images[i] = future.result()
                except Exception as e:
                    raise ValueError(f"Error loading image: {e}")
    else:
        images = None

    first_image = worker_read_image_file(image_files[0])
    height, width = first_image.shape[:2]
    container = None
    try:
        container = av.open(str(file_path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.thread_type = "FRAME"  # Parallel processing of frames
        stream.thread_count = os.cpu_count()  # Number of threads to use
        for i in range(len(image_files)):
            image = images[i] if preload else worker_read_image_file(image_files[i])
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as e:
        raise IOError(f"Failed to write video to '{file_path}': {e}")
    finally:
        if container:
            container.close()


def write_points_to_ply(
    points: np.ndarray, save_path: Union[str, Path], colors: np.ndarray = None
) -> None:
    """Write a point cloud to a PLY file."""
    if colors is None:  # Default to green color
        colors = np.tile([0, 1, 0], (points.shape[0], 1)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(save_path), pcd, write_ascii=True)


def read_points_from_ply(file_path: Union[str, Path]) -> np.ndarray:
    """Read a point cloud from a PLY file."""
    pcd = o3d.io.read_point_cloud(str(file_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    return points


def get_xyz_from_uvd(u, v, d, fx, fy, cx, cy):
    if d == 0:  # Handle division by zero
        return [0.0, 0.0, 0.0]
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    return [x, y, z]


def get_uv_from_xyz(x, y, z, fx, fy, cx, cy):
    if z == 0:  # Prevent division by zero
        return [-1.0, -1.0]
    u = x * fx / z + cx
    v = y * fy / z + cy
    return [u, v]


def get_bbox_from_landmarks(landmarks, width, height, margin=3):
    """Get the xyxy bounding box from hand landmarks."""
    # Filter landmarks where both x and y are valid (i.e., not -1)
    marks = np.array(landmarks)
    valid_mask = ~np.all(marks == -1, axis=1)
    if valid_mask.sum() == 0:
        # If no valid landmarks, return a full image bounding box
        return [-1, -1, -1, -1]
    # Get the bounding box using cv2.boundingRect
    x, y, w, h = cv2.boundingRect(marks[valid_mask])
    bbox = [x, y, x + w, y + h]
    # Apply margin while ensuring the bounding box stays within image bounds
    bbox[0] = max(0, bbox[0] - margin)
    bbox[1] = max(0, bbox[1] - margin)
    bbox[2] = min(width - 1, bbox[2] + margin)
    bbox[3] = min(height - 1, bbox[3] + margin)
    return bbox.astype(int).tolist()


def get_bbox_from_mask(mask, margin=3):
    """Get the xyxy bounding box from a binary mask."""
    height, width = mask.shape[:2]
    if not np.any(mask):
        return [-1.0, -1.0, -1.0, -1.0]
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    bbox = [x, y, x + w, y + h]
    bbox[0] = max(0, bbox[0] - margin)
    bbox[1] = max(0, bbox[1] - margin)
    bbox[2] = min(width - 1, bbox[2] + margin)
    bbox[3] = min(height - 1, bbox[3] + margin)
    return bbox.astype(float).tolist()


def get_mask_from_seg_image(seg_img, color_to_idx_map):
    H, W, _ = seg_img.shape
    flat_seg_img = seg_img.reshape(-1, 3)
    flat_mask_img = np.zeros((H * W), dtype=np.uint8)
    for color, idx in color_to_idx_map.items():
        matching_pixels = np.all(flat_seg_img == color, axis=1)
        flat_mask_img[matching_pixels] = idx
    mask_img = flat_mask_img.reshape(H, W)
    return mask_img


def draw_debug_image(
    rgb_image,
    hand_mask=None,
    object_mask=None,
    prompt_points=None,
    prompt_labels=None,
    hand_marks=None,
    alpha=0.5,
    draw_boxes=False,
    draw_hand_sides=False,
    reduce_background=False,
):
    """
    Draws debug information on an RGB image.

    Args:
        rgb_image (np.ndarray): The original RGB image.
        hand_mask (np.ndarray, optional): Mask of the hands.
        object_mask (np.ndarray, optional): Mask of the objects.
        prompt_points (list, optional): Points to be drawn on the image.
        prompt_labels (list, optional): Labels for the prompt points.
        hand_marks (list, optional): Hand landmark points.
        alpha (float, optional): Transparency factor for overlay. Defaults to 0.5.
        reduce_background (bool, optional): Whether to reduce the background visibility. Defaults to False.
        draw_boxes (bool, optional): Whether to draw bounding boxes around hands and objects. Defaults to False.
        draw_hand_sides (bool, optional): Whether to draw text indicating left/right hand. Defaults to False.

    Returns:
        np.ndarray: The image with debug information drawn on it.
    """
    height, width = rgb_image.shape[:2]
    overlay = np.zeros_like(rgb_image) if reduce_background else rgb_image.copy()

    def apply_mask(mask, colors):
        for label in np.unique(mask):
            if label == 0:
                continue
            overlay[mask == label] = colors[label].rgb

    def draw_boxes_from_mask(mask, colors):
        for label in np.unique(mask):
            if label == 0:
                continue
            box = get_bbox_from_mask(mask == label)
            cv2.rectangle(
                overlay, (box[0], box[1]), (box[2], box[3]), colors[label].rgb, 2
            )

    # Draw hand mask
    if hand_mask is not None:
        apply_mask(hand_mask, HAND_COLORS)

    # Draw object mask
    if object_mask is not None:
        apply_mask(object_mask, OBJ_CLASS_COLORS)

    # Draw bounding boxes
    if draw_boxes:
        if hand_mask is not None:
            draw_boxes_from_mask(hand_mask, HAND_COLORS)
        if object_mask is not None:
            draw_boxes_from_mask(object_mask, OBJ_CLASS_COLORS)

    # Draw prompt points
    if prompt_points is not None and prompt_labels is not None:
        points = np.array(prompt_points, dtype=np.int32).reshape(-1, 2)
        labels = np.array(prompt_labels, dtype=np.int32).reshape(-1)
        for point, label in zip(points, labels):
            color = COLORS["dark_red"] if label == 0 else COLORS["dark_green"]
            cv2.circle(overlay, tuple(point), 3, color.rgb, -1)

    overlay = cv2.addWeighted(rgb_image, 1 - alpha, overlay, alpha, 0)

    # Draw hand sides
    if draw_hand_sides and hand_mask is not None and hand_marks is None:
        for label in np.unique(hand_mask):
            if label == 0:
                continue
            mask = hand_mask == label
            color = HAND_COLORS[label]
            text = "right" if label == 1 else "left"
            x, y, _, _ = cv2.boundingRect(mask.astype(np.uint8))
            cv2.putText(
                overlay,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                color.rgb,
                1,
                cv2.LINE_AA,
            )

    # Draw hand landmarks
    if hand_marks is not None:
        for ind, marks in enumerate(hand_marks):
            if np.all(marks == -1):
                continue

            # Draw bones
            for bone_idx, (start, end) in enumerate(HAND_BONES):
                if np.any(marks[start] == -1) or np.any(marks[end] == -1):
                    continue
                color = HAND_BONE_COLORS[bone_idx]
                cv2.line(overlay, tuple(marks[start]), tuple(marks[end]), color.rgb, 2)

            # Draw joints
            for i, mark in enumerate(marks):
                if np.any(mark == -1):
                    continue
                color = HAND_JOINT_COLORS[i]
                cv2.circle(overlay, tuple(mark), 5, (255, 255, 255), -1)
                cv2.circle(overlay, tuple(mark), 3, color.rgb, -1)

            if draw_boxes:
                box = get_bbox_from_landmarks(marks, width, height, margin=10)
                color = HAND_COLORS[1] if ind == 0 else HAND_COLORS[2]
                cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color.rgb, 2)

            if draw_hand_sides:
                text = "right" if ind == 0 else "left"
                color = HAND_COLORS[1] if ind == 0 else HAND_COLORS[2]
                x, y, _, _ = cv2.boundingRect(
                    np.array([m for m in marks if np.all(m != -1)], dtype=np.int64)
                )
                cv2.putText(
                    overlay,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    color.rgb,
                    1,
                    cv2.LINE_AA,
                )

    return overlay
