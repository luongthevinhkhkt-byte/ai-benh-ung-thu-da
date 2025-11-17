# app.py
import os
import io
import base64
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("SECRET_KEY", "dev_key_123")

# Tắt session filesystem để tiết kiệm RAM
# app.config["SESSION_TYPE"] = "filesystem"  # BỎ DÒNG NÀY

# ============================
# MAPPING CHUẨN THEO ImageFolder (alphabet order)
# ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# 0 1 2 3 4 5 6
# ============================
CLASSES = [
    "akiec - Tổn thương tiền ung thư da (Actinic Keratoses / Intraepithelial Carcinoma)",
    "bcc - Ung thư biểu mô tế bào đáy (Basal Cell Carcinoma)",
    "bkl - Tổn thương lành tính dạng dày sừng (Benign Keratosis-like Lesions)",
    "df - U xơ da lành tính (Dermatofibroma)",
    "mel - U hắc tố ác tính (Melanoma - Ung thư da hắc tố)",
    "nv - Nốt ruồi / Nốt sắc tố lành tính (Melanocytic Nevi)",
    "vasc - Tổn thương mạch máu (Vascular Lesions)"
]

# === TRANSFORM NHẸ HƠN: Resize trực tiếp về 224x224 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === GEMINI CONFIG ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === RESNET18 MODEL LOADER ===
@torch.no_grad()
def load_resnet18_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model_path = "Mo_Hinh/mo_hinh_RESNET18.pth"
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[MODEL] ResNet18 loaded: {model_path}")
        except Exception as e:
            print(f"[ERROR] Load model failed: {e}")
    else:
        print(f"[WARNING] Model not found: {model_path} – Using ImageNet weights!")
    
    return model

# Load model một lần
resnet_model = load_resnet18_model()

def predict_image_pil(img_pil):
    img = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = resnet_model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        print(f"[PREDICT] Class: {CLASSES[idx].split()[0]} | Confidence: {conf:.4f}")
    return idx, probs, conf

# === TREATMENT PLAN ===
def get_treatment_plan(key):
    key = key.lower()
    plans = {
        'akiec': (
            "<strong>Khái niệm:</strong> Tổn thương tiền ung thư da, mảng thô ráp, có vảy, hay ở vùng nắng.<br>"
            "<strong>Nguyên nhân:</strong> Tia UV lâu dài.<br>"
            "<strong>Biện pháp:</strong> Tránh nắng 10h–16h, kem chống nắng SPF 50+, mũ rộng vành."
        ),
        'bcc': (
            "<strong>Khái niệm:</strong> Ung thư da phổ biến, nốt bóng, dễ chảy máu.<br>"
            "<strong>Nguyên nhân:</strong> Tia UV tích lũy, da sáng.<br>"
            "<strong>Biện pháp:</strong> Kem chống nắng hàng ngày, khám da định kỳ, tránh giường tắm nắng."
        ),
        'bkl': (
            "<strong>Khái niệm:</strong> Lành tính, u sừng bã nhờn, màu nâu, sần.<br>"
            "<strong>Nguyên nhân:</strong> Tuổi tác, di truyền.<br>"
            "<strong>Biện pháp:</strong> Theo dõi nếu thay đổi hình dạng/màu."
        ),
        'df': (
            "<strong>Khái niệm:</strong> Nốt chắc, nâu/hồng, lõm khi ấn (dimple sign).<br>"
            "<strong>Nguyên nhân:</strong> Chấn thương nhẹ (côn trùng, cạo lông).<br>"
            "<strong>Biện pháp:</strong> Tránh tổn thương da, mặc quần dài."
        ),
        'mel': (
            "<strong>Khái niệm:</strong> Ung thư da nguy hiểm nhất, dễ di căn.<br>"
            "<strong>Nguyên nhân:</strong> Tia UV mạnh, nhiều nốt ruồi, da trắng.<br>"
            "<strong>Biện pháp:</strong> Tránh nắng, SPF 50+, kiểm tra ABCDE, khám da liễu định kỳ."
        ),
        'nv': (
            "<strong>Khái niệm:</strong> Nốt ruồi lành tính, phổ biến.<br>"
            "<strong>Nguyên nhân:</strong> Di truyền, nội tiết, nắng nhẹ.<br>"
            "<strong>Biện pháp:</strong> Kem chống nắng, theo dõi ABCDE, khám nếu thay đổi."
        ),
        'vasc': (
            "<strong>Khái niệm:</strong> Tổn thương mạch máu, chấm đỏ, dễ chảy máu.<br>"
            "<strong>Nguyên nhân:</strong> Bẩm sinh, mang thai, chấn thương.<br>"
            "<strong>Biện pháp:</strong> Tránh va chạm, theo dõi ở trẻ (thường tự khỏi)."
        )
    }
    return plans.get(key, "Tham khảo bác sĩ da liễu.")

# === CHAT HISTORY (last 4) ===
def add_to_history(role, text):
    hist = session.get("chat_history", [])
    prefix = "Bạn: " if role == "user" else "Trợ lý: "
    hist.append(f"{prefix}{text}")
    session["chat_history"] = hist[-4:]
    session.modified = True

def get_recent_context():
    return "\n".join(session.get("chat_history", [])[-4:])

# === GEMINI REPLY (KHÔNG DÙNG RAG) ===
def generate_reply(query, recent=""):
    disease_code = ""
    for line in recent.split("\n"):
        if "Bạn:" in line:
            line_up = line.upper()
            for code in ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]:
                if code in line_up:
                    disease_code = code
                    break
            if disease_code: break

    prompt = f"""
        Bạn là bác sĩ da liễu chuyên nghiệp. Tư vấn ngắn gọn, rõ ràng, dựa trên kiến thức y khoa chuẩn (AAD, WHO, Mayo Clinic).
        Chỉ trả lời về 7 bệnh da trong HAM10000:
        - AKIEC, BCC, BKL, DF, MEL, NV, VASC

        **Lịch sử (4 lượt gần nhất):**
        {recent or '[Không có]'}

        **Câu hỏi người dùng:** {query}
        **Mã bệnh (nếu có):** {disease_code or '[Chưa xác định]'}

        **QUY TẮC:**
        1. Chỉ thông tin tham khảo – không thay bác sĩ.
        2. MEL, BCC, AKIEC → khuyên khám NGAY.
        3. Không kê đơn, bôi thuốc, phẫu thuật tại nhà.
        4. Format: Chào → Chẩn đoán → Hành động → Lời khuyên → Kết thúc.
        5. Luôn nhắc: Đây là tham khảo, cần gặp bác sĩ.
        6. Không đề cập tác giả nếu không hỏi.

        **TRẢ LỜI NGẮN GỌN, RÕ RÀNG, KHÔNG DÀI DÒNG.**
        """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")  # Dùng flash nhanh + nhẹ
        res = model.generate_content(prompt, generation_config={"max_output_tokens": 300})
        response = (res.text or "").strip()
        if "tham khảo" not in response.lower():
            response += "\n\n**Lưu ý: Đây chỉ là thông tin tham khảo. Vui lòng gặp bác sĩ da liễu để được chẩn đoán chính xác.**"
        return response
    except Exception as e:
        print(f"Gemini error: {e}")
        return "Lỗi hệ thống. Vui lòng thử lại sau."

# === ROUTES ===
@app.route("/")
def index():
    session.clear()  # Reset session
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        msg = data.get("message", "").strip()
        if not msg:
            return jsonify({"response": "Hãy nhập câu hỏi!"}), 200
        
        add_to_history("user", msg)
        recent = get_recent_context()
        reply = generate_reply(msg, recent)
        add_to_history("assistant", reply)
        return jsonify({"response": reply}), 200
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "Lỗi xử lý. Vui lòng thử lại."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh được gửi"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Không chọn file"}), 400
    try:
        img = Image.open(file.stream).convert("RGB")
        idx, probs, conf = predict_image_pil(img)
        label = CLASSES[idx]
        key_short = label.split()[0].lower()

        # Resize ảnh nhỏ để giảm RAM khi encode base64
        img_small = img.resize((300, 300), Image.LANCZOS)
        buf = io.BytesIO()
        img_small.save(buf, "JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        treatment = get_treatment_plan(key_short)
        return jsonify({
            "label": label,
            "image_base64": b64,
            "confident": round(conf, 4),
            "treatment": treatment
        }), 200
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({"error": "Lỗi xử lý ảnh"}), 500

@app.route("/reset", methods=["POST"])
def reset_session():
    session.clear()
    return jsonify({"status": "reset"}), 200

@app.route("/get_history")
def get_history():
    history = session.get("chat_history", [])
    formatted = []
    for line in history:
        if line.startswith("Bạn: "):
            formatted.append({"role": "user", "content": line[5:]})
        elif line.startswith("Trợ lý: "):
            formatted.append({"role": "assistant", "content": line[8:]})
    return jsonify({"history": formatted})

# === RUN ===
if __name__ == "__main__":
    print("Starting server... http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)  # TẮT DEBUG
