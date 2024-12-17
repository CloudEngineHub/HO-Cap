import pycocotools.mask as mask_util
from hocap_toolkit.utils import *

PROJ_ROOT = Path(__file__).parents[2]
HOCAP_DATASET_ROOT = PROJ_ROOT / "datasets"

HOCAP_INFO = read_data_from_yaml(PROJ_ROOT / "config/hocap_info.yaml")

# The train/valid/test split is defined separately for each task (HPE, ODET, OPE)
#   - The split is defined as a list of items
#   - Each item is a list in the format [subject_index, sequence_index, camera_index, frame_index]
#   - For example, [0, 0, 0, 0] refers "subject_1/20231022_190534/105322251564" folder and frame "color_000000.jpg" & "depth_000000.png"
HPE_CONFIG = read_data_from_json(PROJ_ROOT / "config/hocap_hpe.json")
ODET_CONFIG = read_data_from_json(PROJ_ROOT / "config/hocap_odet.json")
OPE_CONFIG = read_data_from_json(PROJ_ROOT / "config/hocap_ope.json")

COCO_CATEGORIES = [
    {
        "id": i + 1,
        "name": obj_class,
        "supercategory": "object",
    }
    for i, obj_class in enumerate(HOCAP_INFO["object_classes"])
    if "HAND" not in obj_class
]

YOLO_CLASSES = [
    obj_class for obj_class in HOCAP_INFO["object_classes"] if "HAND" not in obj_class
]


class HOCapFactory:
    def __init__(self) -> None:
        self._logger = get_logger(__class__.__name__)

        self._calib_dir = HOCAP_DATASET_ROOT / "calibration"
        self._models_dir = HOCAP_DATASET_ROOT / "models"
        self._rs_width = 640
        self._rs_height = 480
        self._mano_betas = [
            self._read_mano_beta(sub_id) for sub_id in HOCAP_INFO["subject_ids"]
        ]
        self._rs_RTs = self._load_rs_cam_RTs()
        self._rs_RTs_inv = [np.linalg.inv(RT) for RT in self._rs_RTs]

    def _world_mano_pose_to_camera(self, mano_pose, cam_RT_inv):
        if np.all(mano_pose == -1):
            return mano_pose

        pose_c = mano_pose.copy()
        rvt_w = np.concatenate([pose_c[:3], pose_c[-3:]], axis=0)
        mat_w = rvt_to_mat(rvt_w)
        mat_c = cam_RT_inv @ mat_w
        rvt_c = mat_to_rvt(mat_c)
        pose_c[:3] = rvt_c[:3]
        pose_c[-3:] = rvt_c[-3:]
        return pose_c

    def _read_mano_beta(self, sub_id):
        file_path = self._calib_dir / "mano" / f"{sub_id}.yaml"
        mano_data = read_data_from_yaml(file_path)
        mano_betas = np.array(mano_data["betas"]).astype(np.float32)
        return mano_betas

    def _load_pose_m(self, sub_id, seq_id):
        file_path = HOCAP_DATASET_ROOT / sub_id / seq_id / "poses_m.npy"
        poses_m = np.load(file_path).astype(np.float32)
        return poses_m

    def _load_rs_cam_RTs(self):
        def create_mat(values):
            return np.array(
                [values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32
            )

        file_path = self._calib_dir / f"extrinsics/extrinsics_20231014.yaml"
        extrinsics = read_data_from_yaml(file_path)["extrinsics"]
        tag_1 = create_mat(extrinsics["tag_1"])
        tag_1_inv = np.linalg.inv(tag_1)
        rs_RTs_master = [
            create_mat(extrinsics[serial])
            for serial in HOCAP_INFO["device_serials"][:-1]  # Exclude the hololens
        ]
        rs_RTs_world = [tag_1_inv @ RT for RT in rs_RTs_master]
        return rs_RTs_world

    def _get_obj_model_path(self, obj_id):
        mesh_file = self._models_dir / obj_id / "textured_mesh.obj"
        texture_file = self._models_dir / obj_id / "textured_mesh_0.png"
        material_file = self._models_dir / obj_id / "textured_mesh.mtl"
        return (mesh_file, texture_file, material_file)

    def _load_object_vertices(self):
        object_vertices = {}
        for obj_id in HOCAP_INFO["object_classes"]:
            if "HAND" in obj_id:
                continue  # Exclude hands
            mesh_file, _, _ = self._get_obj_model_path(obj_id)
            mesh = trimesh.load(mesh_file)
            object_vertices[obj_id] = mesh.vertices.astype(np.float32)
        return object_vertices

    def _calculate_model_info(self, mesh):
        # Diameter (approximate) as the max distance between any two vertices
        diameter = mesh.bounding_sphere.primitive.radius * 2
        # Bounding box dimensions
        min_bounds, max_bounds = mesh.bounds
        size = max_bounds - min_bounds
        return {
            "diameter": float(diameter),
            "min_x": float(min_bounds[0]),
            "min_y": float(min_bounds[1]),
            "min_z": float(min_bounds[2]),
            "size_x": float(size[0]),
            "size_y": float(size[1]),
            "size_z": float(size[2]),
        }

    def _calculate_projected_bbox(self, cam_K, obj_pose, object_vertices):
        """Calculate the 2D bounding box of the projected 3D object mesh."""
        # Transform vertices to camera space
        object_vertices_homogeneous = np.hstack(
            (object_vertices, np.ones((object_vertices.shape[0], 1)))
        )
        vertices_cam = (obj_pose @ object_vertices_homogeneous.T).T[:, :3]

        # Project vertices into 2D
        vertices_2d = (cam_K @ vertices_cam.T).T
        vertices_2d = (
            vertices_2d[:, :2] / vertices_2d[:, 2:3]
        )  # Normalize by depth to get 2D coordinates

        # Get min/max x and y for the bounding box
        x_min, y_min = np.min(vertices_2d, axis=0)
        x_max, y_max = np.max(vertices_2d, axis=0)

        # Return bbox as [x_min, y_min, width, height]
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        return bbox

    def _binary_mask_to_rle(self, mask):
        """
        Convert binary mask to COCO RLE format using pycocotools.
        """
        # Ensure mask is uint8
        binary_mask = mask.astype(np.uint8)

        rle = mask_util.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("utf-8")  # Convert to string (COCO format)
        return rle

    def create_odet_dataset(self, dataset_type):
        if dataset_type not in ["coco", "yolo"]:
            msg = f"Invalid dataset type: {dataset_type}, choose from 'coco' or 'yolo'"
            self._logger.error(msg)
            raise ValueError(msg)

        self._logger.info(
            f">>>>>>>>>> Creating HOCap Object Detection Dataset ({dataset_type})..."
        )
        output_dir = HOCAP_DATASET_ROOT / f"hocap_odet_{dataset_type}"
        make_clean_folder(output_dir)

        if dataset_type == "yolo":
            yolo_classes = [
                obj_c for obj_c in HOCAP_INFO["object_classes"] if "HAND" not in obj_c
            ]
            dataset_info = {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(yolo_classes),
                "names": yolo_classes,
            }
            write_data_to_yaml(output_dir / "data.yaml", dataset_info)

            for split, split_data in ODET_CONFIG.items():
                self._logger.info(f"Extracting {split} data...")

                save_image_dir = output_dir / split / "images"
                make_clean_folder(save_image_dir)

                save_label_dir = output_dir / split / "labels"
                make_clean_folder(save_label_dir)

                tqbar = tqdm(total=len(split_data), ncols=100)
                for image_idx, (sub_idx, seq_idx, cam_idx, frame_idx) in enumerate(
                    split_data
                ):
                    sub_id = HOCAP_INFO["subject_ids"][sub_idx]
                    seq_id = HOCAP_INFO["sequence_ids"][seq_idx]
                    cam_id = HOCAP_INFO["device_serials"][cam_idx]

                    # Copy image
                    src_img_path = (
                        HOCAP_DATASET_ROOT
                        / f"{sub_id}/{seq_id}/{cam_id}/color_{frame_idx:06d}.jpg"
                    )
                    save_img_name = f"sub{sub_idx:02d}_seq{seq_idx:02d}_cam{cam_idx:02d}_frame{frame_idx:06d}.jpg"
                    shutil.copy(src_img_path, save_image_dir / save_img_name)

                    # Generate yolo annotations
                    yolo_annotations = []
                    label_data = np.load(
                        src_img_path.parent / f"label_{frame_idx:06d}.npz"
                    )
                    seg_mask = label_data["seg_mask"]
                    obj_class_inds = label_data["obj_class_inds"]
                    obj_class_names = label_data["obj_class_names"]
                    for idx, mask_i in enumerate(np.unique(seg_mask)):
                        if mask_i == 0:  # Background
                            continue
                        mask = seg_mask == mask_i
                        if mask.sum() < 10:
                            continue  # Ignore tiny/noisy masks

                        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                        category_id = obj_class_inds[idx - 1].item()
                        cx = (x + w / 2) / self._rs_width
                        cy = (y + h / 2) / self._rs_height
                        w /= self._rs_width
                        h /= self._rs_height

                        yolo_annotations.append(
                            f"{category_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                        )

                    # Save yolo annotations
                    save_label_name = f"sub{sub_idx:02d}_seq{seq_idx:02d}_cam{cam_idx:02d}_frame{frame_idx:06d}.txt"
                    (save_label_dir / save_label_name).write_text(
                        "\n".join(yolo_annotations)
                    )

                    tqbar.update(1)
                tqbar.close()

        elif dataset_type == "coco":
            save_anno_dir = output_dir / "annotations"
            make_clean_folder(save_anno_dir)

            for split, split_data in ODET_CONFIG.items():
                self._logger.info(f"Extracting {split} data...")
                save_image_dir = output_dir / split
                make_clean_folder(save_image_dir)

                save_anno_path = save_anno_dir / f"instances_{split}HOCap.json"

                annotations = []
                images = []
                tqbar = tqdm(total=len(split_data), ncols=100)
                for image_idx, (sub_idx, seq_idx, cam_idx, frame_idx) in enumerate(
                    split_data
                ):
                    sub_id = HOCAP_INFO["subject_ids"][sub_idx]
                    seq_id = HOCAP_INFO["sequence_ids"][seq_idx]
                    cam_id = HOCAP_INFO["device_serials"][cam_idx]

                    src_img_path = (
                        HOCAP_DATASET_ROOT
                        / f"{sub_id}/{seq_id}/{cam_id}/color_{frame_idx:06d}.jpg"
                    )

                    # Copy image
                    save_img_name = f"sub{sub_idx:02d}_seq{seq_idx:02d}_cam{cam_idx:02d}_frame{frame_idx:06d}.jpg"
                    shutil.copy(src_img_path, save_image_dir / save_img_name)

                    # Generate COCO annotations
                    images.append(
                        {
                            "id": image_idx,
                            "file_name": save_img_name,
                            "height": self._rs_height,
                            "width": self._rs_width,
                        }
                    )

                    label_data = np.load(
                        src_img_path.parent / f"label_{frame_idx:06d}.npz"
                    )
                    seg_mask = label_data["seg_mask"]
                    obj_class_inds = label_data["obj_class_inds"]

                    for idx, mask_i in enumerate(np.unique(seg_mask)):
                        if mask_i == 0:
                            continue  # Background
                        mask = (seg_mask == mask_i).astype(np.uint8)
                        area = np.sum(mask).item()
                        if area < 10:
                            continue  # Ignore tiny/noisy masks

                        x, y, w, h = cv2.boundingRect(mask)
                        category_id = (
                            obj_class_inds[idx - 1].item() + 1
                        )  # COCO category id starts from 1

                        annotations.append(
                            {
                                "id": len(annotations),
                                "image_id": image_idx,
                                "category_id": category_id,
                                "bbox": [x, y, w, h],
                                "area": area,
                                "iscrowd": 0,
                                "segmentation": self._binary_mask_to_rle(mask),
                            }
                        )
                    tqbar.update(1)
                tqbar.close()

                coco_data = {
                    "images": images,
                    "annotations": annotations,
                    "categories": COCO_CATEGORIES,
                }
                with open(save_anno_path, "w") as f:
                    json.dump(coco_data, f)

    def create_hpe_dataset(self):
        self._logger.info(">>>>>>>>>> Creating HOCap Hand Pose Estimation Dataset...")
        output_dir = HOCAP_DATASET_ROOT / "hocap_hpe"
        make_clean_folder(output_dir)

        for split, split_data in HPE_CONFIG.items():
            self._logger.info(f"Extracting {split} data...")

            # Create directories
            save_image_dir = output_dir / split / "images"
            save_image_dir.mkdir(parents=True, exist_ok=True)
            save_label_dir = output_dir / split / "labels"
            save_label_dir.mkdir(parents=True, exist_ok=True)

            tqbar = tqdm(total=len(split_data), ncols=100)
            for image_idx, (sub_idx, seq_idx, cam_idx, frame_idx) in enumerate(
                split_data
            ):
                sub_id = HOCAP_INFO["subject_ids"][sub_idx]
                seq_id = HOCAP_INFO["sequence_ids"][seq_idx]
                cam_id = HOCAP_INFO["device_serials"][cam_idx]

                # Copy image
                src_img_path = (
                    HOCAP_DATASET_ROOT
                    / f"{sub_id}/{seq_id}/{cam_id}/color_{frame_idx:06d}.jpg"
                )
                save_img_name = f"sub{sub_idx:02d}_seq{seq_idx:02d}_cam{cam_idx:02d}_frame{frame_idx:06d}.jpg"
                shutil.copy(src_img_path, save_image_dir / save_img_name)

                # Generate hand pose annotations
                label_data = np.load(src_img_path.parent / f"label_{frame_idx:06d}.npz")
                cam_K = label_data["cam_K"]
                hand_joints_2d = label_data["hand_joints_2d"]
                hand_joints_3d = label_data["hand_joints_3d"]
                mano_shape = self._mano_betas[sub_idx]
                mano_poses_w = self._load_pose_m(sub_id, seq_id)[:, frame_idx]
                cam_RT_inv = self._rs_RTs_inv[cam_idx]
                mano_poses = np.stack(
                    [
                        self._world_mano_pose_to_camera(p, cam_RT_inv)
                        for p in mano_poses_w
                    ],
                    axis=0,
                )

                save_label_name = f"sub{sub_idx:02d}_seq{seq_idx:02d}_cam{cam_idx:02d}_frame{frame_idx:06d}.npz"
                np.savez_compressed(
                    save_label_dir / save_label_name,
                    cam_K=cam_K,
                    keypoints_2d=hand_joints_2d,
                    keypoints_3d=hand_joints_3d,
                    mano_betas=mano_shape,
                    mano_poses=mano_poses,
                )
                tqbar.update(1)
            tqbar.close()

    def create_ope_dataset(self):
        self._logger.info(">>>>>>>>>> Creating HOCap Object Pose Estimation Dataset...")
        output_dir = HOCAP_DATASET_ROOT / "hocap_ope"
        make_clean_folder(output_dir)

        object_vertices = self._load_object_vertices()

        for split, split_data in OPE_CONFIG.items():
            self._logger.info(f"Extracting {split} data...")
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            tqbar = tqdm(total=len(split_data), ncols=100)
            gt_data = {}
            for image_idx, (sub_idx, seq_idx, cam_idx, frame_idx) in enumerate(
                split_data
            ):
                sub_id = HOCAP_INFO["subject_ids"][sub_idx]
                seq_id = HOCAP_INFO["sequence_ids"][seq_idx]
                cam_id = HOCAP_INFO["device_serials"][cam_idx]

                save_rgb_dir = split_dir / f"{seq_idx:02d}_{cam_idx:02d}" / "rgb"
                save_mask_dir = split_dir / f"{seq_idx:02d}_{cam_idx:02d}" / "mask"
                save_depth_dir = split_dir / f"{seq_idx:02d}_{cam_idx:02d}" / "depth"
                save_rgb_dir.mkdir(parents=True, exist_ok=True)
                save_mask_dir.mkdir(parents=True, exist_ok=True)
                save_depth_dir.mkdir(parents=True, exist_ok=True)

                src_rgb_path = (
                    HOCAP_DATASET_ROOT
                    / f"{sub_id}/{seq_id}/{cam_id}/color_{frame_idx:06d}.jpg"
                )
                src_depth_path = (
                    HOCAP_DATASET_ROOT
                    / f"{sub_id}/{seq_id}/{cam_id}/depth_{frame_idx:06d}.png"
                )
                label_path = (
                    HOCAP_DATASET_ROOT
                    / f"{sub_id}/{seq_id}/{cam_id}/label_{frame_idx:06d}.npz"
                )

                # Generate gt data
                gt_info = []
                label_data = np.load(label_path)
                cam_K = label_data["cam_K"]
                obj_poses = label_data["obj_poses"]
                seg_mask = label_data["seg_mask"]
                obj_class_inds = label_data["obj_class_inds"]
                obj_class_names = label_data["obj_class_names"]
                obj_seg_mask = np.zeros_like(seg_mask)

                for idx, mask_id in enumerate(np.unique(seg_mask)):
                    if mask_id == 0:  # Background
                        continue
                    obj_idx = (
                        obj_class_inds[idx - 1].item() + 1
                    )  # BOP format starts from 1
                    obj_name = obj_class_names[idx - 1]

                    if "HAND" in obj_name:  # Exclude hands
                        continue

                    obj_pose = obj_poses[idx - 1]
                    # update obj_seg_mask
                    obj_seg_mask[seg_mask == mask_id] = obj_idx

                    gt_entry = {
                        "cam_R_m2c": obj_pose[:3, :3].tolist(),
                        "cam_t_m2c": obj_pose[:3, 3].tolist(),
                        "obj_id": obj_idx,
                        "cam_K": cam_K.tolist(),
                        "obj_bb": self._calculate_projected_bbox(
                            cam_K, obj_pose, object_vertices[obj_name]
                        ),
                    }
                    gt_info.append(gt_entry)

                # Save gt data
                shutil.copy(
                    src_rgb_path, save_rgb_dir / f"{sub_idx:02d}_{frame_idx:06d}.jpg"
                )
                shutil.copy(
                    src_depth_path,
                    save_depth_dir / f"{sub_idx:02d}_{frame_idx:06d}.png",
                )
                write_mask_image(
                    save_mask_dir / f"{sub_idx:02d}_{frame_idx:06d}.png", obj_seg_mask
                )

                # Add gt.yaml
                if f"{seq_idx:02d}_{cam_idx:02d}" not in gt_data:
                    gt_data[f"{seq_idx:02d}_{cam_idx:02d}"] = {}
                gt_data[f"{seq_idx:02d}_{cam_idx:02d}"][
                    f"{sub_idx:02d}_{frame_idx:06d}"
                ] = gt_info

                tqbar.update(1)
            tqbar.close()

            # Save gt.yaml
            self._logger.info(f"Saving gt.yaml for {split} split...")
            for key in gt_data.keys():
                write_data_to_yaml(split_dir / f"{key}/gt.yaml", gt_data[key])

        # Copy object models
        self._logger.info("Generating object models...")
        save_model_dir = output_dir / "models"
        save_model_dir.mkdir(parents=True, exist_ok=True)

        model_info = {}
        for obj_idx, obj_id in enumerate(HOCAP_INFO["object_classes"]):
            if "HAND" in obj_id:
                continue  # Exclude hands
            mesh_file, _, _ = self._get_obj_model_path(obj_id)
            # Convert obj to ply
            mesh = trimesh.load(mesh_file)
            mesh.export(save_model_dir / f"{obj_id}.ply")
            model_info[obj_idx] = self._calculate_model_info(mesh)
        write_data_to_json(save_model_dir / "model_info.json", model_info)
