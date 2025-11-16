# app.py
import os
import io
import base64
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ResNet18
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("SECRET_KEY", "dev_key_123")
app.config["SESSION_TYPE"] = "filesystem"

# ============================
# MAPPING CHUẨN THEO ImageFolder (alphabet order)
# ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
#    0       1      2     3     4     5     6
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

# === TRANSFORM CHUẨN NHƯ LÚC TRAIN (ResNet18 dùng input 224x224) ===
transform = transforms.Compose([
    transforms.Resize((600, 450)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === GEMINI CONFIG ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === RESNET18 MODEL LOADER ===
@torch.no_grad()
def load_resnet18_model():
    # Load ResNet18 với trọng số ImageNet
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Thay lớp fully connected để phân loại 7 lớp
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    # Đường dẫn mô hình đã fine-tune
    model_path = "Mo_Hinh/mo_hinh_RESNET18.pth"
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[MODEL] ResNet18 loaded: {model_path}")
        except Exception as e:
            print(f"[ERROR] Load ResNet18 model failed: {e}")
    else:
        print(f"[WARNING] Model not found: {model_path} – Using ImageNet weights only!")
    
    return model

# Khởi tạo mô hình
resnet_model = load_resnet18_model()

def predict_image_pil(img_pil):
    img = transform(img_pil).unsqueeze(0)  # [1, 3, 224, 224]
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
            "<strong>Khái niệm:</strong> Tổn thương tiền ung thư da, xuất hiện dưới dạng mảng da thô ráp, có vảy, thường ở vùng tiếp xúc nhiều ánh nắng.<br>"
            "<strong>Nguyên nhân thường gặp:</strong> Tiếp xúc lâu dài với tia cực tím (UV) từ ánh nắng mặt trời hoặc máy tắm nắng nhân tạo.<br>"
            "<strong>Biện pháp hạn chế:</strong> Tránh ra nắng vào khung giờ 10h–16h, sử dụng kem chống nắng SPF 50+ hàng ngày, đội mũ rộng vành và mặc áo dài tay.<br>"
        ),
        'bcc': (
            "<strong>Khái niệm:</strong> Loại ung thư da phổ biến nhất, phát triển từ tế bào đáy, thường biểu hiện bằng nốt sần bóng, chảy máu nhẹ, không lan xa.<br>"
            "<strong>Nguyên nhân thường gặp:</strong> Tiếp xúc tia UV tích lũy qua nhiều năm, đặc biệt ở người da sáng, hệ miễn dịch suy yếu.<br>"
            "<strong>Biện pháp hạn chế:</strong> Sử dụng kem chống nắng phổ rộng SPF 50+ mỗi ngày, kiểm tra da định kỳ 6–12 tháng, tuyệt đối không sử dụng giường tắm nắng.<br>"
        ),
        'bkl': (
            "<strong>Khái niệm:</strong> Tổn thương lành tính, thường là u sừng bã nhờn (seborrheic keratosis), màu nâu, dính như sáp, bề mặt sần sùi.<br>"
            "<strong>Nguyên nhân thường gặp:</strong> Liên quan đến tuổi tác (thường gặp sau 30 tuổi), yếu tố di truyền, không rõ nguyên nhân chính xác.<br>"
            "<strong>Biện pháp hạn chế:</strong> Không thể ngăn ngừa hoàn toàn; khuyến khích theo dõi định kỳ nếu tổn thương thay đổi hình dạng hoặc màu sắc.<br>"
        ),
        'df': (
            "<strong>Khái niệm:</strong> Nốt da lành tính, chắc, màu nâu hoặc hồng, thường xuất hiện ở cẳng chân, có thể lõm khi ấn (dấu hiệu dimple).<br>"
            "<strong>Nguyên nhân thường gặp:</strong> Phản ứng da với chấn thương nhẹ như côn trùng cắn, gai đâm, hoặc cạo lông.<br>"
            "<strong>Biện pháp hạn chế:</strong> Hạn chế chấn thương da, mặc quần dài khi đi vùng cỏ cây, xử lý vết côn trùng cắn kịp thời.<br>"
        ),
        'mel': (
            "<strong>Khái niệm:</strong> Ung thư da ác tính nguy hiểm nhất, phát triển từ tế bào hắc tố, có khả năng di căn xa nếu không phát hiện sớm.<br>"
            "<strong>Nguyên nhân thường gặp:</strong> Tiếp xúc tia UV mạnh (nắng gắt, bỏng nắng), nhiều nốt ruồi bất thường, da trắng, tiền sử gia đình.<br>"
            "<strong>Biện pháp hạn chế:</strong> Tránh nắng trực tiếp, dùng kem chống nắng SPF 50+ hàng ngày, tự kiểm tra da theo quy tắc ABCDE mỗi tháng, khám da liễu định kỳ.<br>"
        ),
        'nv': (
            "<strong>Khái niệm:</strong> Nốt ruồi lành tính (nốt sắc tố), phổ biến ở mọi lứa tuổi, hình thành từ cụm tế bào hắc tố.<br>"
            "<strong>Nguyên nhân thường gặp:</strong> Yếu tố di truyền, tiếp xúc ánh nắng nhẹ, thay đổi nội tiết (mang thai, dậy thì).<br>"
            "<strong>Biện pháp hạn chế:</strong> Sử dụng kem chống nắng, tránh phơi nắng kéo dài, theo dõi nốt ruồi theo quy tắc ABCDE, khám ngay nếu có thay đổi.<br>"
        ),
        'vasc': (
            "<strong>Khái niệm:</strong> Tổn thương mạch máu lành tính, thường là chấm đỏ, u máu nổi, dễ chảy máu khi va chạm.<br>"
            "<strong>Nguyên nhân thường gặp:</strong> Bẩm sinh (trẻ sơ sinh), mang thai, chấn thương, hoặc dùng một số thuốc (như estrogen).<br>"
            "<strong>Biện pháp hạn chế:</strong> Tránh va chạm mạnh, theo dõi ở trẻ em (thường tự thoái triển), bảo vệ da khỏi chấn thương.<br>"
        )
    }
    return plans.get(key, "Tham khảo bác sĩ da liễu.")

# === RAG SYSTEM ===
def load_and_split_pdfs(folder="docs", chunk_size=1000, overlap=150):
    chunks = []
    if not os.path.exists(folder):
        print(f"Folder {folder} không tồn tại!")
        return chunks
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".pdf"): 
            continue
        path = os.path.join(folder, fname)
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            if not text: 
                continue
            step = chunk_size - overlap
            for i in range(0, len(text), step):
                piece = text[i:i+chunk_size].strip()
                if piece:
                    chunks.append({"text": f"[Nguồn: {fname}] {piece}", "source": fname})
        except Exception as e:
            print(f"PDF error {fname}: {e}")
    return chunks

def embed_texts_with_genai(texts):
    embeddings = []
    for t in texts:
        for _ in range(3):
            try:
                res = genai.embed_content(model="models/embedding-004", content=t)
                emb = res.get("embedding")
                if emb is not None:
                    embeddings.append(np.array(emb, dtype=float))
                    break
            except Exception as e:
                print(f"Embed error: {e}")
                import time; time.sleep(1)
        else:
            embeddings.append(np.zeros(768))
    return np.vstack(embeddings) if embeddings else np.zeros((0, 0))

class RAGSystem:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self._build_index()
    def _build_index(self):
        print("Building RAG index...")
        self.chunks = load_and_split_pdfs()
        if self.chunks:
            self.embeddings = embed_texts_with_genai([c["text"] for c in self.chunks])
            print(f"RAG ready: {len(self.chunks)} chunks")
        else:
            print("No PDF found for RAG.")
    def retrieve(self, query, top_k=3):
        if not self.chunks or self.embeddings is None:
            return ""
        q_emb = embed_texts_with_genai([query])[0].reshape(1, -1)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return "\n\n---\n\n".join(self.chunks[i]["text"] for i in top_idx)

rag = RAGSystem()

# === CHAT HISTORY (last 4) ===
def add_to_history(role, text):
    hist = session.get("chat_history", [])
    prefix = "Bạn: " if role == "user" else "Trợ lý: "
    hist.append(f"{prefix}{text}")
    session["chat_history"] = hist[-4:]
    session.modified = True

def get_recent_context():
    return "\n".join(session.get("chat_history", [])[-4:])

# === GEMINI REPLY ===
def generate_reply(query, rag_context="", recent=""):
    disease_code = ""
    for line in recent.split("\n"):
        if "Bạn:" in line and any(code in line.upper() for code in ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]):
            for code in ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]:
                if code in line.upper():
                    disease_code = code
                    break
            break

    prompt = f"""
        Bạn là một bác sĩ chuyên gia về da liễu, dựa trên kiến thức tài liệu uy tín trên các nền tảng lớn như AAD, WHO, Mayo Clinic và các nghiên cứu y khoa mới nhất.
        Nhiệm vụ của bạn là tư vấn người dùng các vấn đề liên quan đến các loại bệnh ung thư da và tổn thương da dựa trên dữ liệu HAM10000 gồm 7 loại bệnh:
        - AKIEC: Tổn thương tiền ung thư da (Actinic Keratoses / Intraepithelial Carcinoma)
        - BCC: Ung thư biểu mô tế bào đáy (Basal Cell Carcinoma)
        - BKL: Tổn thương lành tính dạng dày sừng (Benign Keratosis-like Lesions)
        - DF: U xơ da lành tính (Dermatofibroma)
        - MEL: U hắc tố ác tính (Melanoma - Ung thư da hắc tố)
        - NV: Nốt ruồi / Nốt sắc tố lành tính (Melanocytic Nevi)
        - VASC: Tổn thương mạch máu (Vascular Lesions)
        
        **Tài liệu RAG (nếu có):**
        {rag_context or '[Không có tài liệu phù hợp]'}

        **Lịch sử gần đây (4 lượt):**
        {recent or '[Không có]'}
        **QUY TẮC BẮT BUỘC:**
        1. **Chỉ đưa thông tin tham khảo** – không thay thế khám bác sĩ.
        2. **Trường hợp nghi ngờ ung thư (MEL, BCC, AKIEC)** → BẮT BUỘC khuyên **nên khám ngay + đưa ra các gợi ý chăm sóc sức khỏe kịp thời**.
        3. **Dựa vào RAG nếu có**, nếu không → dùng kiến thức y khoa uy tín (AAD, WHO, Mayo Clinic).
        4. **Không tự ý kê đơn, bôi thuốc, phẫu thuật tại nhà**.
        5. **Format rõ ràng**: Chào → Chẩn đoán → Hành động → Lời khuyên → Kết thúc.
        6. Xưng xử **như bác sĩ da liễu chuyên nghiệp**.
        7. Luôn nhắc người dùng: **Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.**
        8. Tác giả tạo ra dự án này: Học sinh: Nguyễn Thành Đạt, Nguyễn Gia Bảo, Giáo viên hướng dẫn: Lê Thị Hai, Đơn vị: Trường THCS Lương Thế Vinh. Công nghệ sử dụng: Finetuned ResNet18, Gemini 2.5, RAG với Google Embedding-004, không cần trả lời nếu không cần thiết và người dùng không hỏi tác giả là ai.
        ---

        **Câu hỏi người dùng:** {query}

        **Mã bệnh phát hiện (nếu có):** {disease_code or '[Chưa xác định]'}
        ---
        **TRẢ LỜI THEO FORMAT SAU (ví dụ MELANOMA):**
        Chào bạn,
        Bạn được nghi ngờ **MELANOMA** (ung thư hắc tố) – **rất nghiêm trọng**.

        **Hành động khẩn cấp:**
        • Đặt lịch khám **bác sĩ da liễu** ngay trong 1 tuần
        • Yêu cầu **sinh thiết** để xác định giai đoạn
        • Không tự ý bôi thuốc, nặn, hoặc phơi nắng

        **Lộ trình điều trị tham khảo (chỉ khi có chẩn đoán chính thức):**
        • Giai đoạn sớm: phẫu thuật cắt rộng
        • Giai đoạn di căn: hóa trị, miễn dịch

        **Lời khuyên:**
        • Tránh nắng 10h–16h, dùng kem chống nắng SPF 50+
        • Theo dõi ABCDE: bất đối xứng, bờ, màu, đường kính, tiến triển

        **Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.**

        Chúc bạn mau khỏe bệnh!

        ---

        Bây giờ, hãy trả lời theo đúng format trên, dựa vào RAG nếu có, nếu không thì dùng kiến thức khoa học chuẩn.
        """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        res = model.generate_content(prompt)
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
    session["chat_history"] = []
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        msg = data.get("message", "").strip()
        if not msg:
            return jsonify({"response": "Hãy nhập câu hỏi!"}), 200

        add_to_history("user", msg)
        rag_ctx = rag.retrieve(msg, top_k=3)
        recent = get_recent_context()
        reply = generate_reply(msg, rag_ctx, recent)
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

        # Convert ảnh sang base64 để hiển thị
        buf = io.BytesIO()
        img.save(buf, "JPEG")
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
    session.pop("chat_history", None)
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
    app.run(host="0.0.0.0", port=5000, debug=True)
