
# rgb_module.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class FightDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = r3d_18(weights=None) # Khi predict sẽ load checkpoint sau nên để weights=None
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class RGBExtractor:
    def __init__(self, checkpoint_path=None):
        # 1. Xác định thiết bị chạy (Ưu tiên GPU nếu có)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. Khởi tạo mô hình FightDetector với num_classes=2
        self.model = FightDetector(num_classes=2)
        
        # 3. Nạp trọng số từ file checkpoint sau khi bạn train trên Colab xong
        if checkpoint_path:
            print(f"Đang nạp trọng số r3d_18 từ: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Loại bỏ tiền tố 'module.' nếu có (do train DataParallel)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            self.model.load_state_dict(state_dict)
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, frames_window):
        """
        Hàm nhận vào một Python List chứa 16 hoặc 30 frame liên tiếp.
        Định dạng đầu ra bắt buộc: đúng một con số Float từ 0.0 đến 1.0 (Xác suất bạo lực)
        """
        # Nếu danh sách truyền vào nhiều hơn 16 frame, ta chỉ lấy 16 frame đầu tiên 
        # (Vì mô hình r3d_18 của tác giả được thiết kế tối ưu cho T=16)
        if len(frames_window) > 16:
            frames_window = frames_window[:16]
            
        # Bước 1: Ghép các frame trong List lại -> [T, C, H, W]
        video_tensor = torch.stack(frames_window)
        
        # Bước 2: Hoán đổi trục phù hợp với đầu vào CNN 3D -> [C, T, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        # Bước 3: Thêm chiều Batch (B=1) ở đầu -> [1, C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)
        video_tensor = video_tensor.to(self.device)
        
        # Bước 4: Dự đoán không tính gradient
        with torch.no_grad():
            logits = self.model(video_tensor) # Đầu ra trả về Tensor kích thước [1, 2]
            
            # Vì mô hình trả về logits thô của 2 lớp, ta dùng Softmax để tính xác suất
            probabilities = torch.softmax(logits, dim=1) 
            
        # Bước 5: Trích xuất xác suất của lớp "Bạo lực" (Thường là index 1)
        # Ép kiểu về số Float thuần túy từ 0.0 đến 1.0 theo đúng định dạng đầu ra bắt buộc
        rgb_score = float(probabilities[0][1].item())
        
        return rgb_score


if __name__ == "__main__":
    # Truyền file checkpoint vừa tải về vào đây
    extractor = RGBExtractor(checkpoint_path="model_train.pth") 

    # Giả lập 16 frame ảnh để test hàm predict
    mock_frames = [torch.rand(3, 224, 224) for _ in range(16)]

    # Chạy predict thử nghiệm
    score = extractor.predict(mock_frames)
    print(f"\nrgb_score = {score:.4f}")