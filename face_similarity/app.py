import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import cv2
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
import torch

st.title("HUMAN FACE DETECTION")

# Chỉ đọc database một lần và cache lại
@st.cache_resource
def read_face_db():
    with open("face_db.json", "r") as f:
        face_db = json.load(f)

    # Chuyển lại từ list -> numpy array
    face_db = {
        k: [np.array(vec) for vec in v]
        for k, v in face_db.items()
    }
    return face_db

# Hàm so sánh ảnh mới với DB
def query_database(vec_new, db, threshold=0.8):
    
    best_score = -1
    best_name = "Unknown"

    for name, vec_list in db.items():
        for vec in vec_list:
            score = cosine_similarity([vec_new], [vec])[0][0]
            if score > best_score:
                best_score = score
                best_name = name

    if best_score >= threshold:
        return best_name, best_score
    else:
        return "Unknown", best_score
    

def get_crop_image_with_yolo(model, image):
    results = model.predict(image) 
    result = results[0]

    # Nếu không có box nào được dự đoán
    if len(result.boxes) == 0:
        return None, []

    # Tìm box có confidence cao nhất
    best_box = max(result.boxes, key=lambda box: float(box.conf[0]))

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    conf = float(best_box.conf[0])
    cls = int(best_box.cls[0])
    label = f"{model.names[cls]} {conf:.2f}"

    # Vẽ bounding box và nhãn
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Crop ảnh theo bounding box tốt nhất
    image_crop = image[y1:y2, x1:x2]
    image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)

    # Chuyển ảnh gốc sang RGB để hiển thị
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb, image_crop

def save_data_to_database(face_db):
    serializable_face_db = {
    k: [arr.tolist() for arr in v] 
    for k, v in face_db.items()
    }
    with open("face_db.json", "w") as f:
        json.dump(serializable_face_db, f, indent=2)
    

def get_embedding_from_numpy(facenet_model, image_np: np.ndarray) -> np.ndarray:
    """
    Nhận ảnh numpy RGB, trả về vector 512 chiều từ Facenet.
    """
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    if image_np is None:
        return None

    # Nếu ảnh là BGR (OpenCV), chuyển sang RGB
    if image_np.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_np

    # Resize ảnh về 160x160
    image_resized = cv2.resize(image_rgb, (160, 160))

    # Normalize và chuyển về tensor
    image_tensor = transforms.ToTensor()(image_resized)
    image_tensor = transforms.Normalize([0.5], [0.5])(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)  # B x C x H x W

    # Dự đoán
    with torch.no_grad():
        embedding = facenet_model(image_tensor)
    return embedding.squeeze().numpy() 



@st.cache_resource
def load_detection_model(model_path):
    return YOLO("best.pt")

@st.cache_resource
def load_embedding_model(model_pretrained):
    facenet_model = InceptionResnetV1(pretrained=model_pretrained).eval()
    return facenet_model

# Load database
face_db = read_face_db()
detection_model = load_detection_model("best.pt")
embedding_model = load_embedding_model("vggface2")



# Khởi tạo trạng thái hiển thị nếu chưa có
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# Nút bấm "Thêm người mới"
if st.button("➕ Thêm người mới"):
    st.session_state.show_form = True

# Chỉ hiển thị form khi nút được bấm
if st.session_state.show_form:
    st.subheader("📝 Nhập thông tin người mới")

    # Upload nhiều ảnh
    uploaded_files = st.file_uploader("Chọn một hoặc nhiều ảnh", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Nhập tên
    name_input = st.text_input("✍️ Nhập tên của bạn")

    # Nút xử lý
    if st.button("🚀 Xử lý ảnh"):
        if uploaded_files and name_input:
            st.success(f"Tên bạn nhập: {name_input}")
            new_data_human = []
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                #st.image(image, caption=f"Ảnh {idx+1}", use_column_width=True)
                st.info(f"Ảnh {idx+1} đã được nạp thành công!")
                image_np = np.array(image)
                if image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                print(image_np.shape)
                _, query_image = get_crop_image_with_yolo(detection_model, image_np)
                query_embedding = get_embedding_from_numpy(embedding_model, query_image)
                new_data_human.append(query_embedding)
                
            if name_input in face_db:
                face_db[name_input].extend(new_data_human)
            else:
                face_db[name_input] = new_data_human
            save_data_to_database(face_db)

        else:
            st.warning("Vui lòng chọn ít nhất một ảnh và nhập tên.")


# Giao diện upload ảnh
uploaded_file = st.file_uploader("Chọn một ảnh để upload", type=["jpg", "jpeg", "png"])

# Xử lý và hiển thị ảnh
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn đã upload", use_column_width=True)

    image = np.array(image)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Trích khuôn mặt và embedding
    _, query_image = get_crop_image_with_yolo(detection_model, image)
    query_embedding = get_embedding_from_numpy(embedding_model, query_image)

    # So khớp với database
    name, score = query_database(query_embedding, face_db)

    # Hiển thị kết quả lên giao diện
    st.write(f"**Tên dự đoán:** {name}")
    st.write(f"**Độ tương đồng:** {score:.4f}")

