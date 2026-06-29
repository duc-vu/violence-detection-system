import cv2
import torch

from src.decision_engine import DecisionEngine
from src.pose_module import PoseTracker
from src.rgb_module_r3d_18 import RGBExtractor


def frame_to_rgb_tensor(frame_bgr, size=224):
    """
    OpenCV frame: [H, W, 3] BGR uint8
    RGBExtractor cần tensor: [3, 224, 224] float
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (size, size))

    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


if __name__ == "__main__":
    video_path = "data/violence.avi"      
    checkpoint_path = "models/model_r3d_18 (train_3_epoch).pth"

    engine = DecisionEngine(window_size=30, step_size=15)
    pose_tracker = PoseTracker()
    rgb_extractor = RGBExtractor(checkpoint_path=checkpoint_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

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

        cv2.putText(
            frame,
            f"Risk: {result['risk_score']}% | Alert: {result['alert']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if result["alert"] else (0, 255, 0),
            2,
        )

        cv2.imshow("Violence Detection Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()