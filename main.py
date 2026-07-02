import argparse
from pathlib import Path

import cv2
import torch

from src.decision_engine import DecisionEngine
from src.pose_module import PoseTracker
from src.rgb_module_r3d_18 import RGBExtractor


COCO_SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def frame_to_rgb_tensor(frame_bgr, size=224):
    """
    OpenCV frame: [H, W, 3] BGR uint8
    RGBExtractor cần tensor: [3, 224, 224] float
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (size, size))

    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def draw_pose(frame, pose_data, violators=None, keypoint_conf=0.3):
    violators = set(violators or [])

    for person in pose_data:
        track_id = person.get("track_id")
        color = (0, 0, 255) if track_id in violators else (0, 255, 255)

        x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {track_id}" if track_id is not None else "ID ?"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        keypoints = person.get("keypoints", [])

        for start_idx, end_idx in COCO_SKELETON:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue

            x_start, y_start, conf_start = keypoints[start_idx]
            x_end, y_end, conf_end = keypoints[end_idx]

            if conf_start < keypoint_conf or conf_end < keypoint_conf:
                continue

            cv2.line(
                frame,
                (int(x_start), int(y_start)),
                (int(x_end), int(y_end)),
                color,
                2,
            )

        for x, y, conf in keypoints:
            if conf >= keypoint_conf:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
                cv2.circle(frame, (int(x), int(y)), 4, color, 1)

    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="dataset/RWF-2000/train/Fight/0vasozvJYvk_0.avi",
        help="Input video path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/model_train.pth",
        help="RGB model checkpoint path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/pose_detection_test.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. Use 0 to process the whole video.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable OpenCV preview window while processing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = args.video
    checkpoint_path = args.checkpoint
    output_path = Path(args.output)

    engine = DecisionEngine(window_size=30, step_size=15)
    pose_tracker = PoseTracker(model_name="yolo26m-pose.pt")
    rgb_extractor = RGBExtractor(checkpoint_path=checkpoint_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được video output: {output_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        pose_data = pose_tracker.process_frame(frame)
        rgb_frame = frame_to_rgb_tensor(frame)

        result = engine.update(
            frame=rgb_frame,
            pose_data_from_duc=pose_data,
            rgb_module=rgb_extractor,
        )

        print(
            f"Frame {frame_idx:05d} | "
            f"Risk: {result['risk_score']}% | "
            f"Alert: {result['alert']} | "
            f"Violators: {result['violators']}"
        )

        draw_pose(frame, pose_data, result["violators"])

        cv2.putText(
            frame,
            f"Risk: {result['risk_score']}% | Alert: {result['alert']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if result["alert"] else (0, 255, 0),
            2,
        )

        writer.write(frame)

        if not args.no_show:
            cv2.imshow("Violence Detection Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    writer.release()

    if not args.no_show:
        cv2.destroyAllWindows()

    print(f"Đã xuất video test: {output_path}")
