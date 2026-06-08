import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
import math
import re
import unicodedata
from qrdet import QRDetector

# Sửa lỗi 'ANTIALIAS' do thư viện vietocr dùng code cũ không tương thích với Pillow >= 10.0.0
if not hasattr(Image, 'ANTIALIAS'):
    setattr(Image, 'ANTIALIAS', getattr(Image, 'Resampling', Image).LANCZOS)

from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch

# Đường dẫn tĩnh (tuyệt đối) tới model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHT = os.path.join(BASE_DIR, 'models', 'yolo_best.pt')
VIETOCR_WEIGHT = os.path.join(BASE_DIR, 'models', 'vietocr_best.pth')

# Cấu hình thiết bị
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load mô hình YOLO
yolo_model = YOLO(YOLO_WEIGHT)

# Load mô hình VietOCR
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = VIETOCR_WEIGHT
config['cnn']['pretrained'] = False
config['device'] = device
vietocr_predictor = Predictor(config)

# Khởi tạo QR Detector
qr_detector = QRDetector()

CLASS_NAMES = {
    0: 'card', 1: 'dob', 2: 'expiry', 3: 'gender', 4: 'id_number',
    5: 'name', 6: 'nationality', 7: 'origin', 8: 'residence',
}

FIELD_LABELS = {
    'id_number': 'Số CCCD', 'name': 'Họ và tên', 'dob': 'Ngày sinh',
    'gender': 'Giới tính', 'nationality': 'Quốc tịch', 'origin': 'Quê quán',
    'residence': 'Nơi thường trú', 'expiry': 'Ngày hết hạn',
}

COLORS = {
    'id_number':'#FF5733','name':'#33A1FF','dob':'#28A745','gender':'#FFC107',
    'nationality':'#9C27B0','origin':'#FF9800','residence':'#00BCD4','expiry':'#F44336'
}

COUNTRIES = [
    "Việt Nam", "Trung Quốc", "Hàn Quốc", "Nhật Bản", "Mỹ", "Pháp", "Đức", "Anh", "Nga", 
    "Lào", "Campuchia", "Thái Lan", "Singapore", "Malaysia", "Indonesia", "Philippines",
    "Đài Loan", "Ấn Độ", "Úc", "Canada"
]

def remove_accents(text):
    """Chuyển chuỗi tiếng Việt có dấu sang không dấu."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

# Build dict: không dấu -> có dấu, dùng để tra cứu nhanh
COUNTRIES_NO_ACCENT = {remove_accents(c).lower(): c for c in COUNTRIES}

def postprocess_text(field, text):
    text = text.strip()
    if not text:
        return ""
        
    if field == 'id_number':
        digits = re.sub(r'\D', '', text)
        if len(digits) >= 12:
            return digits[-12:]
        return digits
        
    elif field == 'name':
        text = re.sub(r'\d+', '', text)
        text = text.strip(" ,.-")
        return text.strip()
        
    elif field == 'nationality':
        text_no_accent = remove_accents(text).lower()
        # So sánh không dấu: kiểm tra từng tỪn quốc gia (không dấu) xem có xuất hiện trong chuỗi OCR không
        for country_no_accent, country_original in COUNTRIES_NO_ACCENT.items():
            if country_no_accent in text_no_accent:
                return country_original
        # Fallback: xóa số và trả về nguyên bản
        text = re.sub(r'\d+', '', text)
        text = text.strip(" ,.-")
        return text.strip()
        
    elif field == 'gender':
        text_lower = text.lower()
        if 'nam' in text_lower:
            return 'Nam'
        elif 'nữ' in text_lower or 'nu' in text_lower:
            return 'Nữ'
        text = re.sub(r'\d+', '', text)
        return text.strip(" ,.-")
        
    elif field in ['dob', 'expiry']:
        if field == 'expiry':
            text_lower = text.lower()
            if any(k in text_lower for k in ['vô', 'không', 'thời', 'hạn']):
                return 'Không thời hạn'
                
        parts = re.split(r'[/|-]', text)
        if len(parts) == 3:
            day = re.sub(r'\D', '', parts[0])
            month = re.sub(r'\D', '', parts[1])
            year = re.sub(r'\D', '', parts[2])
            
            if len(day) > 2: day = day[-2:]
            if len(month) > 2: month = month[-2:]
            if len(year) > 4: year = year[-4:]
            
            return f"{day}/{month}/{year}"
        return text
        
    elif field in ['origin', 'residence']:
        text = re.sub(r'^\d+\s+', '', text)
        text = re.sub(r'\s+\d+$', '', text)
        text = text.strip(" ,.-")
        return text.strip()
        
    return text

def crop_region(image_np, box, padding=4):
    x1, y1, x2, y2 = map(int, box)
    h, w = image_np.shape[:2]
    return image_np[max(0,y1-padding):min(h,y2+padding),
                    max(0,x1-padding):min(w,x2+padding)]

def ocr_crop_vietocr(image_np):
    pil_img = Image.fromarray(image_np)
    return vietocr_predictor.predict(pil_img).strip()

def fix_orientation(card_img):
    h, w = card_img.shape[:2]
    detections = qr_detector.detect(image=card_img)
    
    # Không tìm thấy QR thì giữ nguyên ảnh
    if len(detections) == 0:
        return card_img
        
    detection = detections[0]
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    
    qr_center_x = (x1 + x2) / 2
    qr_center_y = (y1 + y2) / 2
    
    # QR nằm góc trên bên phải -> Đúng chiều
    if qr_center_x > w/2 and qr_center_y < h/2:
        return card_img
    # QR nằm góc dưới bên trái -> Ngược 180 độ
    elif qr_center_x < w/2 and qr_center_y > h/2:
        return cv2.rotate(card_img, cv2.ROTATE_180)
    # QR nằm góc trên bên trái -> Ảnh bị xoay trái, cần xoay phải 90 độ (CW)
    elif qr_center_x < w/2 and qr_center_y < h/2:
        return cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)
    # QR nằm góc dưới bên phải -> Ảnh bị xoay phải, cần xoay trái 90 độ (CCW)
    else:
        return cv2.rotate(card_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def run_pipeline(img_rgb):
    conf_yolo = 0.3
    if img_rgb is None:
        return None, "", "", "", "", "", "", "", ""

    result_json = {k: "" for k in FIELD_LABELS.keys()}
    field_boxes = {}

    # === BƯỚC 1: YOLO detect thẻ trên ảnh gốc ===
    results = yolo_model(img_rgb, conf=conf_yolo, verbose=False)[0]

    card_box = None
    for box in results.boxes:
        if CLASS_NAMES.get(int(box.cls[0])) == 'card':
            card_box = box.xyxy[0].cpu().numpy()
            break

    # === BƯỚC 2: Crop thẻ → QR trên crop → xoay chuẩn ===
    if card_box is not None:
        # Crop thẻ với padding lớn hơn để không bị cắt mép QR
        card_crop = crop_region(img_rgb, card_box, padding=20)

        # Xoay dựa trên QR của riêng vùng thẻ (không phải ảnh gốc)
        aligned_card = fix_orientation(card_crop)

        # === BƯỚC 3: YOLO lần 2 trên ảnh thẻ đã xoay chuẩn ===
        results2 = yolo_model(aligned_card, conf=conf_yolo, verbose=False)[0]
        for box2 in results2.boxes:
            cls_name2 = CLASS_NAMES.get(int(box2.cls[0]), "unknown")
            xyxy2 = box2.xyxy[0].cpu().numpy()
            if cls_name2 in result_json:
                field_boxes.setdefault(cls_name2, []).append(xyxy2)

        # Ảnh hiển thị là ảnh thẻ đã được căn chỉnh
        annotated = aligned_card.copy()
        ocr_source = aligned_card

    else:
        # === FALLBACK: Không detect được thẻ → thử QR trên ảnh gốc rồi YOLO lại ===
        aligned_full = fix_orientation(img_rgb)
        results_fb = yolo_model(aligned_full, conf=conf_yolo, verbose=False)[0]
        for box in results_fb.boxes:
            cls_name = CLASS_NAMES.get(int(box.cls[0]), "unknown")
            xyxy = box.xyxy[0].cpu().numpy()
            if cls_name in result_json:
                field_boxes.setdefault(cls_name, []).append(xyxy)

        annotated = aligned_full.copy()
        ocr_source = aligned_full

    # === Kiểm tra: Không nhận diện được bất kỳ trường nào ===
    if not field_boxes:
        gr.Warning(
            "⚠️ Không nhận diện được thẻ CCCD trong ảnh. "
            "Vui lòng kiểm tra lại ảnh và thử tải lên ảnh mặt trước của thẻ Căn cước Công dân."
        )
        return annotated, "", "", "", "", "", "", "", ""

    # === BƯỚC 4: OCR từng trường ===
    for field, boxes_list in field_boxes.items():
        color_rgb = tuple(int(COLORS.get(field, '#FFFFFF')[i:i+2], 16) for i in (1, 3, 5))
        boxes_sorted = sorted(boxes_list, key=lambda b: b[1])

        line_texts = []
        for xyxy in boxes_sorted:
            cropped = crop_region(ocr_source, xyxy, padding=4)
            text = ocr_crop_vietocr(cropped)
            if text:
                line_texts.append(text)

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 2)
            cv2.putText(annotated, field.upper(), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_rgb, 1)

        result_json[field] = postprocess_text(field, " ".join(line_texts))

    # Cảnh báo nếu trích xuất được ít hơn 3 trường quan trọng
    key_fields = ['id_number', 'name', 'dob']
    found_key = sum(1 for f in key_fields if result_json.get(f))
    if found_key < 2:
        gr.Warning(
            "⚠️ Nhận diện được thẻ nhưng trích xuất thông tin chưa đầy đủ. "
            "Hãy thử ảnh rõ hơn, đảm bảo đủ ánh sáng và thẻ không bị che khuất."
        )

    return (
        annotated,
        result_json['id_number'],
        result_json['name'],
        result_json['dob'],
        result_json['gender'],
        result_json['nationality'],
        result_json['origin'],
        result_json['residence'],
        result_json['expiry']
    )

# --- Gradio UI ---
custom_css = """
* {
    font-family: Arial, Helvetica, sans-serif !important;
}
"""

with gr.Blocks(title="Trích xuất thông tin CCCD (Có Xoay Ảnh)", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🪪 Nhận dạng và Trích xuất thông tin CCCD")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload ảnh CCCD")
            process_btn = gr.Button("Trích xuất thông tin", variant="primary")
            output_image = gr.Image(type="numpy", label="Ảnh kết quả (Đã Căn Chỉnh)")

        with gr.Column(scale=1):
            gr.Markdown("### Kết quả trích xuất")
            out_id = gr.Textbox(label="Số CCCD")
            out_name = gr.Textbox(label="Họ và tên")
            out_dob = gr.Textbox(label="Ngày sinh")
            out_gender = gr.Textbox(label="Giới tính")
            out_nationality = gr.Textbox(label="Quốc tịch")
            out_origin = gr.Textbox(label="Quê quán")
            out_residence = gr.Textbox(label="Nơi thường trú")
            out_expiry = gr.Textbox(label="Ngày hết hạn")

    # Xử lý khi nhấn nút
    process_btn.click(
        fn=run_pipeline,
        inputs=[input_image],
        outputs=[output_image, out_id, out_name, out_dob, out_gender,
                 out_nationality, out_origin, out_residence, out_expiry]
    )

    # Tự động xử lý ngay khi upload ảnh xong
    input_image.change(
        fn=run_pipeline,
        inputs=[input_image],
        outputs=[output_image, out_id, out_name, out_dob, out_gender,
                 out_nationality, out_origin, out_residence, out_expiry]
    )

if __name__ == "__main__":
    demo.launch()
