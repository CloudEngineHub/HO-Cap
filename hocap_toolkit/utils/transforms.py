from .common_imports import *


def average_quats(quats: np.ndarray) -> np.ndarray:
    """
    Calculate the average quaternion from a set of quaternions.

    Args:
        quats (np.ndarray): An array of quaternions of shape (N, 4), where N is the number of quaternions.

    Returns:
        np.ndarray: The averaged quaternion of shape (4,).
    """
    if not isinstance(quats, np.ndarray) or quats.shape[-1] != 4:
        raise ValueError("Input must be a numpy array of shape (N, 4).")

    rotations = R.from_quat(quats)
    avg_quat = rotations.mean().as_quat().astype(np.float32)
    return avg_quat


def normalize_quats(qs: np.ndarray) -> np.ndarray:
    """
    Normalize quaternions to have unit length.

    Args:
        qs (np.ndarray): Input quaternion, shape (4,) or (N, 4) where each quaternion is (qx, qy, qz, qw).

    Returns:
        np.ndarray: Normalized quaternion(s), same shape as input.
    """
    # Compute the norm of the quaternion
    norms = np.linalg.norm(qs, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Quaternion norms cannot be zero.")
    return qs / norms


def rvt_to_quat(rvt: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector and translation vector to quaternion and translation vector.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,) for single or (N, 6) for batch.

    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,) for single or (N, 7) for batch,
                    in the format [qx, qy, qz, qw, tx, ty, tz].
    """
    # Ensure the input has the correct shape
    if rvt.ndim == 1 and rvt.shape[0] == 6:
        rv = rvt[:3]
        t = rvt[3:]
        q = R.from_rotvec(rv).as_quat()
        return np.concatenate([q, t], dtype=np.float32)

    elif rvt.ndim == 2 and rvt.shape[1] == 6:
        rv = rvt[:, :3]
        t = rvt[:, 3:]
        q = R.from_rotvec(rv).as_quat()  # Batch process
        return np.concatenate([q, t], axis=-1).astype(np.float32)

    else:
        raise ValueError("Input must be of shape (6,) or (N, 6).")


def quat_to_rvt(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion and translation vector to rotation vector and translation vector.

    Args:
        quat (np.ndarray): Quaternion and translation vector. Shape can be (7,) for single input
                           or (N, 7) for batched input.

    Returns:
        np.ndarray: Rotation vector and translation vector. Shape will be (6,) for single input
                    or (N, 6) for batched input.

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    # Validate input shape
    if not isinstance(quat, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if quat.ndim == 1 and quat.shape[0] == 7:
        batch_mode = False
    elif quat.ndim == 2 and quat.shape[1] == 7:
        batch_mode = True
    else:
        raise ValueError(
            "Input must have shape (7,) for a single quaternion or (N, 7) for a batch of quaternions."
        )

    # Extract quaternion (q) and translation (t)
    q = quat[..., :4]  # Quaternion (4 elements)
    t = quat[..., 4:]  # Translation (3 elements)

    # Convert quaternion to rotation vector
    r = R.from_quat(q)
    rv = r.as_rotvec()  # Convert to rotation vector (3 elements)

    # Concatenate rotation vector and translation vector
    return np.concatenate([rv, t], axis=-1).astype(np.float32)


def rvt_to_mat(rvt: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector and translation vector to pose matrix.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,) for single or (N, 6) for batch.

    Returns:
        np.ndarray: Pose matrix, shape (4, 4) for single or (N, 4, 4) for batch.
    """
    # Single input case (shape (6,))
    if rvt.ndim == 1 and rvt.shape[0] == 6:
        p = np.eye(4)
        rv = rvt[:3]
        t = rvt[3:]
        r = R.from_rotvec(rv)
        p[:3, :3] = r.as_matrix()
        p[:3, 3] = t
        return p.astype(np.float32)

    # Batched input case (shape (N, 6))
    elif rvt.ndim == 2 and rvt.shape[1] == 6:
        N = rvt.shape[0]
        p = np.tile(np.eye(4), (N, 1, 1))  # Create an identity matrix for each batch
        rv = rvt[:, :3]  # Rotation vectors (N, 3)
        t = rvt[:, 3:]  # Translation vectors (N, 3)
        r = R.from_rotvec(rv)
        p[:, :3, :3] = r.as_matrix()  # Set rotation matrices for each batch
        p[:, :3, 3] = t  # Set translation vectors for each batch
        return p.astype(np.float32)

    else:
        raise ValueError("Input must be of shape (6,) or (N, 6).")


def mat_to_rvt(mat_4x4: np.ndarray) -> np.ndarray:
    """
    Convert pose matrix to rotation vector and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4) for single input
                              or (N, 4, 4) for batched input.

    Returns:
        np.ndarray: Rotation vector and translation vector, shape (6,) for single input
                    or (N, 6) for batched input.
    """
    # Single input case (shape (4, 4))
    if mat_4x4.ndim == 2 and mat_4x4.shape == (4, 4):
        r = R.from_matrix(mat_4x4[:3, :3])
        rv = r.as_rotvec()
        t = mat_4x4[:3, 3]
        return np.concatenate([rv, t], dtype=np.float32)

    # Batched input case (shape (N, 4, 4))
    elif mat_4x4.ndim == 3 and mat_4x4.shape[1:] == (4, 4):
        rv = R.from_matrix(mat_4x4[:, :3, :3]).as_rotvec()  # Batch process rotations
        t = mat_4x4[:, :3, 3]  # Batch process translations
        return np.concatenate([rv, t], axis=-1).astype(np.float32)

    else:
        raise ValueError("Input must be of shape (4, 4) or (N, 4, 4).")


def mat_to_quat(mat_4x4: np.ndarray) -> np.ndarray:
    """
    Convert pose matrix to quaternion and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4) for single input or (N, 4, 4) for batched input.

    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,) for single input or (N, 7) for batched input.

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(mat_4x4, np.ndarray) or mat_4x4.shape[-2:] != (4, 4):
        raise ValueError("Input must be a numpy array with shape (4, 4) or (N, 4, 4).")

    if mat_4x4.ndim == 2:  # Single matrix (shape (4, 4))
        r = R.from_matrix(mat_4x4[:3, :3])
        q = r.as_quat()  # Quaternion (shape (4,))
        t = mat_4x4[:3, 3]  # Translation (shape (3,))
        return np.concatenate([q, t], dtype=np.float32)

    elif mat_4x4.ndim == 3:  # Batch of matrices (shape (N, 4, 4))
        r = R.from_matrix(mat_4x4[:, :3, :3])  # Handle batch of rotation matrices
        q = r.as_quat()  # Quaternions (shape (N, 4))
        t = mat_4x4[:, :3, 3]  # Translations (shape (N, 3))
        return np.concatenate([q, t], axis=-1).astype(np.float32)  # Shape (N, 7)

    else:
        raise ValueError("Input dimension is not valid. Must be 2D or 3D.")


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion and translation vector to a pose matrix.

    This function supports converting a single quaternion or a batch of quaternions.

    Args:
        quat (np.ndarray): Quaternion and translation vector. Shape can be (7,) for a single quaternion
                           or (N, 7) for a batch of quaternions, where N is the batch size.

    Returns:
        np.ndarray: Pose matrix. Shape will be (4, 4) for a single quaternion or (N, 4, 4) for a batch of quaternions.

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    # Validate input shape
    if not isinstance(quat, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if quat.ndim == 1 and quat.shape[0] == 7:
        batch_mode = False
    elif quat.ndim == 2 and quat.shape[1] == 7:
        batch_mode = True
    else:
        raise ValueError(
            "Input must have shape (7,) for a single quaternion or (N, 7) for a batch of quaternions."
        )

    # Extract quaternion (q) and translation (t)
    q = quat[..., :4]  # Quaternion (4 elements)
    t = quat[..., 4:]  # Translation (3 elements)

    # Prepare the pose matrix
    if batch_mode:
        N = quat.shape[0]
        p = np.tile(np.eye(4), (N, 1, 1))  # Create N identity matrices
    else:
        p = np.eye(4)  # Single identity matrix

    # Convert quaternion to rotation matrix and fill in the pose matrix
    r = R.from_quat(q)
    p[..., :3, :3] = r.as_matrix()  # Fill rotation part
    p[..., :3, 3] = t  # Fill translation part

    return p.astype(np.float32)


def quat_distance(
    q1: np.ndarray, q2: np.ndarray, in_degree: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate the shortest angular distance in degrees between paired quaternions.

    Args:
        q1 (np.ndarray): First quaternion(s), shape (4,) or (N, 4).
        q2 (np.ndarray): Second quaternion(s), shape (4,) or (N, 4).

    Returns:
        float or np.ndarray: Angular distance in degrees, scalar if single pair, array if multiple pairs.
    """
    # Validate input shapes
    if q1.ndim not in {1, 2} or q2.ndim not in {1, 2}:
        raise ValueError("q1 and q2 must be 1D or 2D arrays.")
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError("Each quaternion must have 4 components (qx, qy, qz, qw).")
    if q1.shape != q2.shape:
        raise ValueError("q1 and q2 must have the same shape.")

    # Normalize quaternions to ensure they are unit quaternions
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)

    # Compute the dot product between paired quaternions
    dot_product = np.sum(q1 * q2, axis=-1)

    # Clamp the dot product to the range [-1, 1] to handle numerical precision issues
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the shortest angular distance in radians
    angular_distance = 2 * np.arccos(np.abs(dot_product))

    # Convert to degrees if needed
    if in_degree:
        return np.degrees(angular_distance)
    return angular_distance


def trans_distance(t1, t2):
    """Calculate the Euclidean distance between two translation vectors or arrays of translation vectors.

    Args:
        t1 (np.ndarray): First translation vector(s) in shape (3,) or (N, 3), where N is the number of vectors.
        t2 (np.ndarray): Second translation vector(s) in shape (3,) or (N, 3), where N is the number of vectors.

    Returns:
        float or np.ndarray: Euclidean distance. Returns a scalar if inputs are 1D vectors, or an array of distances if inputs are 2D arrays.
    Raises:
        ValueError: If the inputs are not valid translation vectors or if their shapes are incompatible.
    """

    # Ensure both inputs are NumPy arrays
    t1 = np.asarray(t1, dtype=np.float32)
    t2 = np.asarray(t2, dtype=np.float32)

    # Check if the shapes of t1 and t2 are compatible
    if t1.shape != t2.shape:
        raise ValueError(
            f"Shape mismatch: t1.shape {t1.shape} and t2.shape {t2.shape} must be the same."
        )

    # Check for valid shapes: (3,) for a single vector or (N, 3) for multiple vectors
    if t1.shape[-1] != 3:
        raise ValueError("Each translation vector must have 3 components (tx, ty, tz).")

    # Compute Euclidean distance
    return np.linalg.norm(t1 - t2, axis=-1)


def angular_difference(q1: np.ndarray, q2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate the angular difference in degrees between two quaternions or arrays of quaternions.

    Args:
        q1 (np.ndarray): First quaternion(s) in [qx, qy, qz, qw] or [N, qx, qy, qz, qw] format.
        q2 (np.ndarray): Second quaternion(s) in [qx, qy, qz, qw] or [N, qx, qy, qz, qw] format.

    Returns:
        float or np.ndarray: Angular difference in degrees, scalar if single pair or array if multiple pairs.
    """
    dim = q1.ndim
    if dim == 1:
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
    else:
        q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
        q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

    q1 = R.from_quat(q1)
    q2 = R.from_quat(q2)
    delta_q = q1.inv() * q2
    delta_q_quat = delta_q.as_quat()

    if dim == 1:
        if delta_q_quat[3] < 0:
            delta_q_quat = -delta_q_quat
    else:
        negative_indices = delta_q_quat[:, 3] < 0
        delta_q_quat[negative_indices] = -delta_q_quat[negative_indices]

    if dim == 1:
        angular_diff = 2 * np.arccos(np.clip(delta_q_quat[3], -1.0, 1.0))
    else:
        angular_diff = 2 * np.arccos(np.clip(delta_q_quat[:, 3], -1.0, 1.0))

    return np.degrees(angular_diff)
