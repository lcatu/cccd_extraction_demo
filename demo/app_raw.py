import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
from qrdet import QRDetector

# Sửa lỗi 'ANTIALIAS' không tương thích với Pillow >= 10.0.0
if not hasattr(Image, 'ANTIALIAS'):
    setattr(Image, 'ANTIALIAS', getattr(Image, 'Resampling', Image).LANCZOS)

from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHT   = os.path.join(BASE_DIR, 'models', 'yolo_best.pt')
VIETOCR_WEIGHT = os.path.join(BASE_DIR, 'models', 'vietocr_best.pth')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

yolo_model = YOLO(YOLO_WEIGHT)

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = VIETOCR_WEIGHT
config['cnn']['pretrained'] = False
config['device'] = device
vietocr_predictor = Predictor(config)

qr_detector = QRDetector()

CLASS_NAMES = {
    0: 'card', 1: 'dob', 2: 'expiry', 3: 'gender', 4: 'id_number',
    5: 'name', 6: 'nationality', 7: 'origin', 8: 'residence',
}

FIELD_LABELS = {
    'id_number': 'Số CCCD',    'name': 'Họ và tên',
    'dob': 'Ngày sinh',        'gender': 'Giới tính',
    'nationality': 'Quốc tịch','origin': 'Quê quán',
    'residence': 'Nơi thường trú', 'expiry': 'Ngày hết hạn',
}

COLORS = {
    'id_number':'#FF5733','name':'#33A1FF','dob':'#28A745','gender':'#FFC107',
    'nationality':'#9C27B0','origin':'#FF9800','residence':'#00BCD4','expiry':'#F44336'
}


def crop_region(image_np, box, padding=4):
    x1, y1, x2, y2 = map(int, box)
    h, w = image_np.shape[:2]
    return image_np[max(0, y1-padding):min(h, y2+padding),
                    max(0, x1-padding):min(w, x2+padding)]


def fix_orientation(card_img):
    h, w = card_img.shape[:2]
    detections = qr_detector.detect(image=card_img)
    if len(detections) == 0:
        return card_img
    detection = detections[0]
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    qr_center_x = (x1 + x2) / 2
    qr_center_y = (y1 + y2) / 2

    if qr_center_x > w/2 and qr_center_y < h/2:
        return card_img
    elif qr_center_x < w/2 and qr_center_y > h/2:
        return cv2.rotate(card_img, cv2.ROTATE_180)
    elif qr_center_x < w/2 and qr_center_y < h/2:
        return cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return cv2.rotate(card_img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def run_pipeline_raw(img_rgb):
    """Pipeline KHÔNG có hậu xử lý — trả ra kết quả OCR thô."""
    conf_yolo = 0.3
    if img_rgb is None:
        return None, "", "", "", "", "", "", "", ""

    result_json = {k: "" for k in FIELD_LABELS.keys()}
    field_boxes = {}

    # Bước 1: YOLO lần 1 phát hiện thẻ
    results = yolo_model(img_rgb, conf=conf_yolo, verbose=False)[0]
    card_box = None
    for box in results.boxes:
        if CLASS_NAMES.get(int(box.cls[0])) == 'card':
            card_box = box.xyxy[0].cpu().numpy()
            break

    # Bước 2: Crop + căn chỉnh xoay
    if card_box is not None:
        card_crop = crop_region(img_rgb, card_box, padding=20)
        aligned_card = fix_orientation(card_crop)

        # Bước 3: YOLO lần 2 phát hiện các trường thông tin
        results2 = yolo_model(aligned_card, conf=conf_yolo, verbose=False)[0]
        for box2 in results2.boxes:
            cls_name2 = CLASS_NAMES.get(int(box2.cls[0]), "unknown")
            if cls_name2 in result_json:
                field_boxes.setdefault(cls_name2, []).append(box2.xyxy[0].cpu().numpy())

        annotated  = aligned_card.copy()
        ocr_source = aligned_card
    else:
        aligned_full = fix_orientation(img_rgb)
        results_fb = yolo_model(aligned_full, conf=conf_yolo, verbose=False)[0]
        for box in results_fb.boxes:
            cls_name = CLASS_NAMES.get(int(box.cls[0]), "unknown")
            if cls_name in result_json:
                field_boxes.setdefault(cls_name, []).append(box.xyxy[0].cpu().numpy())

        annotated  = aligned_full.copy()
        ocr_source = aligned_full

    if not field_boxes:
        gr.Warning("⚠️ Không nhận diện được thẻ CCCD trong ảnh.")
        return annotated, "", "", "", "", "", "", "", ""

    # Bước 4: OCR — KHÔNG hậu xử lý, trả ra chuỗi thô
    for field, boxes_list in field_boxes.items():
        hex_col  = COLORS.get(field, '#FFFFFF').lstrip('#')
        color_rgb = tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))
        boxes_sorted = sorted(boxes_list, key=lambda b: b[1])

        line_texts = []
        for xyxy in boxes_sorted:
            cropped = crop_region(ocr_source, xyxy, padding=4)
            pil_img = Image.fromarray(cropped)
            text = vietocr_predictor.predict(pil_img).strip()
            if text:
                line_texts.append(text)

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 2)
            cv2.putText(annotated, field.upper(), (x1, max(y1-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_rgb, 1)

        # ===== KHÔNG gọi postprocess_text — giữ nguyên văn bản thô =====
        result_json[field] = " ".join(line_texts)

    return (
        annotated,
        result_json['id_number'],
        result_json['name'],
        result_json['dob'],
        result_json['gender'],
        result_json['nationality'],
        result_json['origin'],
        result_json['residence'],
        result_json['expiry'],
    )


# --- Gradio UI ---
custom_css = """
* { font-family: Arial, Helvetica, sans-serif !important; }
"""

with gr.Blocks(title="CCCD Extraction — Kết quả thô (Không hậu xử lý)",
               theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🪪 Nhận dạng CCCD — Kết quả thô (Không hậu xử lý)")
    gr.Markdown("> ⚠️ Đây là phiên bản **KHÔNG có hậu xử lý** dùng để so sánh với phiên bản chính thức.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image  = gr.Image(type="numpy", label="Upload ảnh CCCD")
            process_btn  = gr.Button("Trích xuất thông tin (thô)", variant="primary")
            output_image = gr.Image(type="numpy", label="Ảnh kết quả (Đã Căn Chỉnh)")

        with gr.Column(scale=1):
            gr.Markdown("### Kết quả OCR thô (chưa làm sạch)")
            out_id          = gr.Textbox(label="Số CCCD")
            out_name        = gr.Textbox(label="Họ và tên")
            out_dob         = gr.Textbox(label="Ngày sinh")
            out_gender      = gr.Textbox(label="Giới tính")
            out_nationality = gr.Textbox(label="Quốc tịch")
            out_origin      = gr.Textbox(label="Quê quán")
            out_residence   = gr.Textbox(label="Nơi thường trú")
            out_expiry      = gr.Textbox(label="Ngày hết hạn")

    outputs = [output_image, out_id, out_name, out_dob, out_gender,
               out_nationality, out_origin, out_residence, out_expiry]

    process_btn.click(fn=run_pipeline_raw, inputs=[input_image], outputs=outputs)
    input_image.change(fn=run_pipeline_raw, inputs=[input_image], outputs=outputs)

if __name__ == "__main__":
    demo.launch()
