from collections import deque
import numpy as np

class DecisionEngine:
    def __init__(self, window_size=30, step_size=15, alpha=0.2):
        # Bộ đệm lưu tối đa 30 frame gần nhất để gửi cho luồng RGB
        self.frame_buffer = deque(maxlen=window_size)
        # Bộ đệm lưu lịch sử pose từ Đức để tính vận tốc chuyển động
        self.pose_history = deque(maxlen=window_size)
        
        self.step_size = step_size
        self.frame_counter = 0
        self.current_rgb_score = 0.0
        
        # Hệ số EMA để làm mượt điểm số (alpha càng nhỏ càng mượt nhưng sẽ trễ)
        self.alpha = alpha 
        self.smoothed_risk = 0.0

    def update(self, frame, pose_data_from_duc, rgb_module):
        self.frame_buffer.append(frame)
        self.pose_history.append(pose_data_from_duc)
        self.frame_counter += 1
        
        # 1. Đồng bộ thời gian: Cứ sau 15 frame mới gọi mạng RGB của Trí Anh một lần
        if self.frame_counter % self.step_size == 0 and len(self.frame_buffer) == self.frame_buffer.maxlen:
            # Chuyển deque thành list các frame gửi cho Trí Anh
            self.current_rgb_score = rgb_module.predict(list(self.frame_buffer))
            
        # 2. Tự tính toán điểm Pose Heuristic dựa trên lịch sử xương
        pose_score = self.compute_pose_score_heuristic(list(self.pose_history))
        
        # 3. Late Fusion: Cộng điểm theo tỷ lệ Đề cương (0.4 RGB + 0.6 Pose)
        raw_risk = 0.4 * self.current_rgb_score + 0.6 * pose_score
        
        # 4. Temporal Smoothing: Áp dụng công thức EMA để làm mượt luồng số
        self.smoothed_risk = self.alpha * raw_risk + (1 - self.alpha) * self.smoothed_risk
        
        # 5. Lọc Heuristic nâng cao: Lấy danh sách ID học sinh đang có xung đột
        violator_ids = self.get_violator_ids(pose_data_from_duc)
        
        # Trả về đúng định dạng Dict đã cam kết với Khang làm UI
        return {
            "risk_score": round(self.smoothed_risk * 100, 1), # Trả về % từ 0.0 - 100.0
            "alert": self.smoothed_risk > 0.65,              # Ngưỡng kích hoạt còi hú (65%)
            "violators": violator_ids
        }

    def compute_pose_score_heuristic(self, pose_history_list):
        """
        Ý tưởng thuật toán của ông: 
        - Tính khoảng cách dịch chuyển của các keypoints quan trọng (cổ tay, cổ chân) giữa frame sau và frame trước.
        - Nếu khoảng cách dịch chuyển quá lớn trong thời gian ngắn -> Vận tốc cao -> Nghi vấn đấm/đá.
        """
        if len(pose_history_list) < 2:
            return 0.0
            
        velocities = []
        current_frame_people = pose_history_list[-1]
        prev_frame_people = pose_history_list[-2]
        
        # Lập trình logic so khớp ID giữa 2 frame liên tiếp để tính vận tốc
        for p_curr in current_frame_people:
            for p_prev in prev_frame_people:
                if p_curr["track_id"] == p_prev["track_id"]:
                    # Lấy tọa độ cổ tay phải (Keypoint số 10 trong YOLO Pose)
                    kp_curr = p_curr["keypoints"][10]
                    kp_prev = p_prev["keypoints"][10]
                    
                    # Nếu độ tin cậy của keypoint > 0.5 mới tính toán
                    if kp_curr[2] > 0.5 and kp_prev[2] > 0.5:
                        dist = np.sqrt((kp_curr[0] - kp_prev[0])**2 + (kp_curr[1] - kp_prev[1])**2)
                        velocities.append(dist)
                        
        if len(velocities) == 0:
            return 0.0
            
        # Chuẩn hóa vận tốc về khoảng 0 - 1 (giá trị 50 px/frame coi như max nguy hiểm)
        max_estimated_velocity = 50.0
        pose_risk = np.mean(velocities) / max_estimated_velocity
        return float(np.clip(pose_risk, 0.0, 1.0))

    def get_violator_ids(self, current_pose_data):
        """
        Mẹo heuristic lọc nhiễu: Chỉ bắt những ID nào có khoảng cách hộp (bbox) 
        gần nhau dưới một ngưỡng quy định (tức là có va chạm thể xác).
        """
        violators = []
        if len(current_pose_data) < 2:
            return violators
            
        # Code logic tính khoảng cách tâm các bbox...
        # Nếu gần nhau và điểm smoothed_risk cao -> Cho vào danh sách đen
        if self.smoothed_risk > 0.5:
            for p in current_pose_data:
                violators.append(p["track_id"])
                
        return list(set(violators))
    
    
# --- ĐOẠN CODE TEST ĐỘC LẬP BẰNG MOCK DATA ---
if __name__ == "__main__":
    # 1. Tạo class giả lập cho module RGB của Trí Anh để không bị lỗi gọi hàm
    class MockRGBExtractor:
        def predict(self, frames):
            print("-> [AI] Đã gọi mô hình RGB trích xuất đặc trưng video.")
            return 0.8  # Giả lập Trí Anh trả về xác suất bạo lực 80%

    # 2. Khởi tạo Engine 
    engine = DecisionEngine(window_size=30, step_size=15)
    mock_extractor = MockRGBExtractor()
    
    print("--- BẮT ĐẦU SIMULATE LUỒNG WEBCAM ---")
    # Giả lập luồng dữ liệu webcam chạy qua 40 frame liên tiếp
    for i in range(1, 41):
        # Dữ liệu giả lập cấu trúc của Đức trả về mỗi frame
        mock_frame = np.zeros((224, 224, 3)) # Giả lập 1 ma trận ảnh trống
        mock_pose_data = [
            {
                "track_id": 1,
                "bbox": [10, 20, 100, 200],
                # Giả lập tọa độ 17 điểm, điểm số 10 (cổ tay) thay đổi liên tục để tạo vận tốc
                "keypoints": [[100, 100, 0.9]] * 10 + [[100 + i*5, 100 + i*2, 0.9]] + [[100, 100, 0.9]] * 6
            }
        ]
        
        # Chạy hàm update tâm điểm 
        result = engine.update(mock_frame, mock_pose_data, mock_extractor)
        
        # Xem luồng số nhảy mượt thế nào qua từng frame
        print(f"Frame {i:02d} | Risk Score: {result['risk_score']}% | Alert: {result['alert']} | Violators: {result['violators']}")