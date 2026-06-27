# src/pose_module.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class PoseTracker:
    """
    Minimal YOLO11-Pose tracker.

    Model behavior:
        - Check models/ folder.
        - Create models/ if missing.
        - Check models/yolo11n-pose.pt.
        - Download automatically if missing.
        - Reuse local file if already exists.

    Output:
        [
            {
                "track_id": 1,
                "bbox": [x1, y1, x2, y2],
                "keypoints": [[x, y, conf], ...]
            },
            ...
        ]
    """

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        models_dir: str | Path = "models",
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | int | None = None,
        tracker: str | None = "bytetrack.yaml",
        persist: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.model_path = self._ensure_model(model_name)

        self.model = YOLO(str(self.model_path))

        self.conf = conf
        self.iou = iou
        self.device = device
        self.tracker = tracker
        self.persist = persist

    def _ensure_model(self, model_name: str) -> Path:
        """
        Ensure model file exists inside models_dir.

        Args:
            model_name:
                Official Ultralytics model name, for example:
                - yolo11n-pose.pt
                - yolo11s-pose.pt

        Returns:
            Path to local model file inside models_dir.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.models_dir / model_name

        if model_path.exists():
            print(f"[PoseTracker] Found local model: {model_path}")
            return model_path

        print(f"[PoseTracker] Model not found. Downloading to: {model_path}")

        YOLO(str(model_path))

        if not model_path.exists():
            raise FileNotFoundError(
                f"Failed to download model to {model_path}. "
                f"Check internet connection or model name: {model_name}"
            )

        print(f"[PoseTracker] Downloaded model: {model_path}")
        return model_path

    def process_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
            Run pose estimation and multi-object tracking on a single video frame.

            This method receives one OpenCV frame, applies YOLO pose tracking, and
            returns a list of detected people. Each detected person includes a temporary
            tracking ID, a bounding box, and COCO-format pose keypoints.

            Args:
                frame:
                    A single OpenCV image in BGR format with shape (H, W, 3).

            Returns:
                A list of dictionaries, where each dictionary represents one detected
                person in the current frame.

                Each dictionary has the following structure:
                    {
                        "track_id": int | None,
                        "bbox": [x1, y1, x2, y2],
                        "keypoints": [
                            [x, y, confidence],
                            ...
                        ]
                    }

                Notes:
                    - "track_id" may be None when the tracker has not assigned an ID yet.
                    - "bbox" is in xyxy format.
                    - "keypoints" follows the 17-keypoint COCO pose format.
                    - Coordinates are returned in pixel space.
                    - Confidence values are model-estimated keypoint confidence scores.

            Raises:
                TypeError:
                    If frame is not a numpy.ndarray.

                ValueError:
                    If frame is empty or does not have shape (H, W, 3).
            """
        self._validate_frame(frame)

        results = self.model.track(
            source=frame,
            persist=self.persist,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            tracker=self.tracker,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None or result.keypoints is None:
            return []

        boxes_xyxy = self._to_numpy(result.boxes.xyxy)

        if boxes_xyxy.size == 0:
            return []

        track_ids = self._extract_track_ids(result)
        keypoints = self._extract_keypoints(result)

        frame_data: list[dict[str, Any]] = []

        for person_idx in range(len(boxes_xyxy)):
            person_data = {
                "track_id": track_ids[person_idx],
                "bbox": self._round_list(boxes_xyxy[person_idx].tolist()),
                "keypoints": self._round_nested_list(keypoints[person_idx]),
            }

            frame_data.append(person_data)

        return frame_data

    def reset_tracker(self) -> None:
        try:
            self.model.predictor.trackers = None
        except AttributeError:
            pass

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray from OpenCV.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "frame must have shape (H, W, 3). "
                "Expected an OpenCV BGR image."
            )

        if frame.size == 0:
            raise ValueError("frame is empty.")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.array([])

        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()

        if hasattr(value, "cpu"):
            return value.cpu().numpy()

        return np.asarray(value)

    def _extract_track_ids(self, result: Any) -> list[int | None]:
        num_boxes = len(result.boxes.xyxy)

        if result.boxes.id is None:
            return [None for _ in range(num_boxes)]

        ids = self._to_numpy(result.boxes.id).astype(int).tolist()

        if len(ids) < num_boxes:
            ids += [None for _ in range(num_boxes - len(ids))]

        return ids

    def _extract_keypoints(self, result: Any) -> np.ndarray:
        if hasattr(result.keypoints, "data") and result.keypoints.data is not None:
            kpts = self._to_numpy(result.keypoints.data)

            if kpts.ndim == 3 and kpts.shape[-1] >= 3:
                return kpts[:, :, :3]

        xy = self._to_numpy(result.keypoints.xy)

        if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
            conf = self._to_numpy(result.keypoints.conf)
        else:
            conf = np.ones(xy.shape[:2], dtype=np.float32)

        if xy.ndim != 3:
            return np.zeros((0, 17, 3), dtype=np.float32)

        return np.concatenate([xy, conf[..., None]], axis=-1)

    @staticmethod
    def _round_list(values: list[float], ndigits: int = 2) -> list[float]:
        return [round(float(v), ndigits) for v in values]

    @staticmethod
    def _round_nested_list(values: np.ndarray, ndigits: int = 2) -> list[list[float]]:
        return [
            [round(float(item), ndigits) for item in point]
            for point in values.tolist()
        ]