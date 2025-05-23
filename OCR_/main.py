# OCR/main.py
import cv2
import pytesseract
# Đảm bảo file rotate_func1.py nằm cùng cấp với main.py
import rotate_func1
import os
import sys
import re
import torch

# --- Thiết lập đường dẫn và import modules từ NLP-model ---

# Đường dẫn đến thư mục chứa file main.py (ví dụ: your_workspace/OCR/)
THIS_DIR = os.path.dirname(__file__)

# Đường dẫn đến thư mục gốc của workspace (ví dụ: your_workspace/)
# Bằng cách đi lên một cấp từ THIS_DIR
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

# Đường dẫn đến thư mục NLP-model (ví dụ: your_workspace/NLP-model/)
# Bằng cách đi từ PROJECT_DIR vào thư thư mục NLP-model
NLP_MODEL_DIR = os.path.join(PROJECT_DIR, "NLP-model")

# Thêm đường dẫn đến thư mục NLP-model vào sys.path.
# Điều này cho phép Python tìm thấy các file model.py và utils.py
# khi chúng ta thực hiện import model và utils.
# Kiểm tra để tránh thêm nhiều lần nếu script được chạy nhiều lần trong cùng process
if NLP_MODEL_DIR not in sys.path:
    sys.path.insert(0, NLP_MODEL_DIR) # Sử dụng insert(0, ...) để ưu tiên thư mục project

# Bây giờ có thể import các class và hàm từ NLP-model
# (Vì NLP_MODEL_DIR đã có trong sys.path)
from model import BiLSTM_CRF
from utils import load_pickle, smart_tokenize, extract_entities

# ----------------------------------------------------------

# 1. Nạp các tài nguyên cho mô hình NLP (tải 1 lần khi script bắt đầu)
try:
    # Sử dụng NLP_MODEL_DIR để tạo đường dẫn đầy đủ đến các file .pkl
    word2idx_path = os.path.join(NLP_MODEL_DIR, "word2idx.pkl")
    label2idx_path = os.path.join(NLP_MODEL_DIR, "label2idx.pkl")
    # Dựa trên train.py, checkpoint được lưu ở PROJECT_DIR, không phải NLP_MODEL_DIR
    model_state_dict_path = os.path.join(PROJECT_DIR, "bilstm_crf_best.pt") # Đường dẫn đúng từ train.py

    print(f"Đang nạp word2idx từ: {word2idx_path}")
    word2idx = load_pickle(word2idx_path)
    print(f"Đang nạp label2idx từ: {label2idx_path}")
    label2idx = load_pickle(label2idx_path)
    idx2label = {idx: lab for lab, idx in label2idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nlp_model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tagset_size=len(label2idx),
        padding_idx=word2idx['<PAD>']
    ).to(device)

    print(f"Đang nạp trọng số mô hình từ: {model_state_dict_path}")
    state_dict = torch.load(model_state_dict_path, map_location=device)

    # --- SỬA LỖI: Ánh xạ tên tham số TỪ CŨ (trong state_dict) SANG MỚI (mô hình tiêu chuẩn mong đợi) ---
    # Mô hình sử dụng Standard torchcrf.CRF (sau khi sửa model.py),
    # nó mong đợi tên tham số mới: start_transitions, end_transitions, transitions.
    # state_dict từ file có vẻ chứa tên tham số cũ: start_trans, end_trans, trans_matrix.
    new_state_dict = {}
    for key, value in state_dict.items():
        # Kiểm tra tên cũ (từ state_dict), ánh xạ sang tên mới (mô hình mong đợi)
        if key == "crf.start_trans":
            new_key = "crf.start_transitions"
        elif key == "crf.end_trans":
            new_key = "crf.end_transitions"
        elif key == "crf.trans_matrix":
            new_key = "crf.transitions"
        # Xử lý trường hợp tên key không có tiền tố "crf."
        elif key == "start_trans":
             new_key = "start_transitions"
        elif key == "end_trans":
             new_key = "end_transitions"
        elif key == "trans_matrix":
             new_key = "transitions"
        else:
            # Giữ nguyên các tên tham số khác
            new_key = key

        # Kiểm tra để đảm bảo new_key khớp với tên tham số trong mô hình hiện tại
        # Có thể dùng print(nlp_model.state_dict().keys()) để debug nếu cần
        if new_key in nlp_model.state_dict():
             new_state_dict[new_key] = value
        else:
             print(f"Cảnh báo: Bỏ qua key không khớp trong state_dict: {key} (ánh xạ thành {new_key})")


    # Nạp state_dict đã được ánh xạ tên vào mô hình
    # Strict=True (mặc định) yêu cầu tất cả các key phải khớp.
    # Nếu có tham số nào trong mô hình không có trong new_state_dict hoặc ngược lại, sẽ báo lỗi.
    try:
        # Nạp chỉ các key có trong new_state_dict, bỏ qua các key bị thiếu trong state_dict gốc
        # Nếu mô hình có thêm tham số mới, load_state_dict(..., strict=False) sẽ cho phép bỏ qua chúng.
        # Tuy nhiên, strict=True là tốt nhất để đảm bảo mọi thứ khớp.
        # Nếu vẫn có lỗi, hãy in ra model.state_dict().keys() và state_dict.keys() để so sánh
        # nlp_model.load_state_dict(new_state_dict, strict=True) # Thử strict=True
        # Nếu strict=True báo lỗi Missing Key khác ngoài CRF, thử strict=False
        nlp_model.load_state_dict(new_state_dict, strict=False) # Sử dụng strict=False để bỏ qua các key bị thiếu nếu có (ví dụ optimizer params)

    except RuntimeError as e:
         print(f"Lỗi nghiêm trọng khi nạp state_dict sau khi ánh xạ tên: {e}")
         print("Vui lòng kiểm tra sự khác biệt giữa key trong checkpoint và key trong mô hình.")
         # Gợi ý debug: print(nlp_model.state_dict().keys()) và print(state_dict.keys())
         sys.exit(1)
    # --- KẾT THÚC ĐOẠN CODE ĐÃ SỬA ---


    nlp_model.eval() # Chuyển mô hình sang chế độ đánh giá/dự đoán
    print("Đã nạp mô hình NLP thành công.")

except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file cần thiết cho mô hình NLP: {e}")
    # Thông báo lỗi phù hợp với đường dẫn checkpoint và .pkl
    if "bilstm_crf_best.pt" in str(e):
         print(f"Hãy đảm bảo file bilstm_crf_best.pt nằm trong thư mục: {PROJECT_DIR}")
    elif "pkl" in str(e):
        print(f"Hãy đảm bảo các file .pkl (word2idx.pkl, label2idx.pkl) nằm trong thư mục: {NLP_MODEL_DIR}")
    else:
         print(f"Không tìm thấy file: {e}")
    sys.exit(1) # Thoát chương trình nếu không nạp được tài nguyên
except Exception as e: # Bắt các lỗi khác có thể xảy ra trong quá trình nạp
    print(f"Lỗi không xác định khi nạp mô hình NLP: {e}")
    sys.exit(1)


# Hàm mã hóa câu (inputs và mask) - cần thiết cho mô hình NLP
# Hàm này được sửa để không giới hạn max_len mặc định, khớp với cách dùng trong vòng lặp predict.
# Bỏ tham số max_len và logic padding cố định
def encode(tokens, vocab):
    """
    Mã hóa tokens thành IDs và mask cho model.
    Sử dụng độ dài tokens gốc.
    """
    # Chuyển token -> ID. Sử dụng .get() với UNK cho các từ không có trong vocab
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    # Mask: 1 cho token thực, 0 cho padding. Mask có cùng độ dài với IDs.
    mask = [1] * len(ids)

    # Chuyển sang Tensor và đẩy thiết bị (CPU/GPU)
    # batch_size ở đây là 1 vì xử lý từng câu riêng lẻ
    # Unsqueeze(0) thêm chiều batch
    return (
        torch.LongTensor([ids]).to(device),
        torch.BoolTensor([mask]).to(device)
    )


# --- Bắt đầu quy trình xử lý ảnh OCR ---

# read image
# Đường dẫn ảnh ví dụ - CẦN SỬA LẠI ĐỂ LINH HOẠT HƠN
# Nên sử dụng đường dẫn tương đối từ PROJECT_DIR hoặc đọc từ tham số dòng lệnh
img_path = r"C:\Users\Precision\Desktop\AI-test\Pic\test_pic.jpg" 
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy file ảnh tại đường dẫn: {img_path}")
    sys.exit(1)
img = cv2.imread(img_path)
if img is None:
    print(f"Lỗi: Không thể đọc file ảnh: {img_path}")
    sys.exit(1)

# rotate (sửa nghiêng)
# rotate_func1 được import trực tiếp vì nó cùng cấp với main.py
# Đảm bảo file rotate_func1.py nằm cùng thư mục với main.py
# và hàm deskew_image hoạt động đúng
try:
    fixed = rotate_func1.deskew_image(img, angle_threshold=1)
except Exception as e:
    print(f"Cảnh báo: Lỗi khi sửa nghiêng ảnh: {e}")
    print("Tiếp tục xử lý ảnh gốc...")
    fixed = img # Sử dụng ảnh gốc nếu sửa nghiêng gặp lỗi


# RGB to Gray
gray = cv2.cvtColor(fixed, cv2.COLOR_RGB2GRAY)

# threshold (nhị phân hóa ảnh)
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 21)

# OCR setup và chạy
# Đường dẫn Tesseract OCR - CẦN SỬA LẠI ĐỂ LINH HOẠT HƠN
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <-- SỬA ĐƯỜNG DẪN NÀY
if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
     print(f"Lỗi: Không tìm thấy Tesseract OCR tại đường dẫn: {pytesseract.pytesseract.tesseract_cmd}")
     print("Hãy cài đặt Tesseract hoặc cập nhật đường dẫn trong code.")
     sys.exit(1)

custom_config = r'--oem 3 --psm 6' # Cấu hình cho Tesseract
# Chạy OCR trên ảnh đã qua xử lý adaptive_thresh
try:
    text = pytesseract.image_to_string(adaptive_thresh,lang="vie+eng", config=custom_config)
    if not text or not text.strip(): # Kiểm tra cả text rỗng và chỉ chứa khoảng trắng
        print("OCR không phát hiện được văn bản hoặc văn bản rỗng.")
        print("\n===== Danh sách thuốc phát hiện từ văn bản OCR =====")
        print("Không tìm thấy tên thuốc nào.")
        print("=="*25)
        sys.exit(0) # Thoát nếu không có văn bản từ OCR
except Exception as e:
    print(f"Lỗi khi chạy Tesseract OCR: {e}")
    sys.exit(1) # Thoát nếu OCR gặp lỗi nghiêm trọng

# --- Tích hợp xử lý NLP trên văn bản từ OCR ---

print("\nĐang xử lý NLP để trích xuất tên thuốc...")

# Chuyển văn bản từ OCR (một chuỗi dài) thành danh sách các câu, mỗi câu là danh sách token
# Tách văn bản thành các dòng dựa trên ký tự xuống dòng
ocr_lines = text.strip().splitlines()
sentences = []
for line in ocr_lines:
    # Sử dụng smart_tokenize từ utils.py để tách token từng dòng.
    # Chuyển về chữ thường để khớp với vocab đã train.
    tokens = smart_tokenize(line.strip().lower())
    if tokens: # Chỉ thêm vào danh sách nếu dòng không rỗng sau khi token hóa
        sentences.append(tokens)

if not sentences:
    print("Không có câu nào được tạo từ văn bản OCR sau khi token hóa.")
    print("\n===== Danh sách thuốc phát hiện từ văn bản OCR =====")
    print("Không tìm thấy tên thuốc nào.")
    print("=="*25)
    sys.exit(0) # Thoát nếu không có câu nào để xử lý NLP

# 5. Dự đoán nhãn cho từng câu và thu thập thực thể THUỐC
found_drugs = []
# Sử dụng torch.no_grad() để tắt tính gradient trong quá trình dự đoán, tiết kiệm bộ nhớ và tăng tốc
with torch.no_grad():
    for tokens in sentences:
        if not tokens: # Bỏ qua nếu token list rỗng
            continue

        # Mã hóa câu thành Tensor ID và Mask (encode đã được sửa)
        X, M = encode(tokens, word2idx)

        # Đưa qua mô hình NLP để dự đoán nhãn
        try:
            # Gọi mô hình để dự đoán - model.py đã có logic xử lý inference (crf.decode)
            # model(X, mask=M) trả về list[list[int]] từ crf.decode khi tags=None
            pred_ids = nlp_model(X, mask=M)

            # crf.decode trả về list[list[int]] có shape (batch_size, sequence_length)
            # Vì batch_size = 1, ta lấy phần tử đầu tiên của list ngoài cùng
            if isinstance(pred_ids, list) and len(pred_ids) > 0:
                 pred_ids_list = pred_ids[0] # pred_ids_list là list[int]
            else:
                 print(f"Cảnh báo: Dự đoán trả về định dạng không mong muốn cho câu '{' '.join(tokens)}'. Bỏ qua.")
                 continue

        except Exception as e:
            print(f"Cảnh báo: Lỗi khi dự đoán câu '{' '.join(tokens)}': {e}")
            print("Bỏ qua câu này và tiếp tục...")
            continue

        # Chuyển các ID nhãn dự đoán về lại chuỗi nhãn (chỉ lấy phần độ dài gốc)
        # Đảm bảo độ dài pred_ids_list và tokens khớp.
        # encode không pad cố định, nên chúng nên khớp nhau.
        if len(pred_ids_list) != len(tokens):
             print(f"Cảnh báo: Độ dài nhãn dự đoán ({len(pred_ids_list)}) không khớp độ dài token ({len(tokens)}) cho câu '{' '.join(tokens)}'. Bỏ qua.")
             continue

        # Ánh xạ các ID dự đoán sang chuỗi nhãn
        pred_labels = [idx2label[i] for i in pred_ids_list] # Sử dụng pred_ids_list


        # Gom các token và nhãn thành các thực thể dựa trên định dạng BIO
        entities = extract_entities(tokens, pred_labels)

        # Lấy danh sách các thực thể loại 'DRUG'
        drugs = entities.get('DRUG', [])

        # Lọc danh sách tên thuốc theo các tiêu chí
        # Dựa trên predict1.py, lọc sau khi gom entitiy
        filtered_drugs = []
        if drugs: # Kiểm tra trước khi lọc để tránh lỗi nếu drugs rỗng
             filtered_drugs = [d for d in drugs if isinstance(d, str) and re.search(r"[A-Za-z]", d) and len(d) > 2] # Thêm kiểm tra isinstance(d, str)
             filtered_drugs = [d for d in filtered_drugs if isinstance(d, str) and not re.fullmatch(r"[\d\.\,\-\/\\]+", d)] # Thêm kiểm tra isinstance(d, str)


        # Thêm các tên thuốc mới tìm được vào danh sách cuối cùng (loại bỏ trùng lặp)
        for d in filtered_drugs: # Duyệt qua danh sách đã lọc
            clean_d = d.lower() # Chuyển về chữ thường để so sánh và lưu trữ
            if clean_d not in found_drugs:
                found_drugs.append(clean_d)

# 6. In kết quả cuối cùng
print("\n===== Danh sách thuốc phát hiện từ văn bản OCR =====")
if found_drugs:
    # In kết quả đã được lowercased
    for drug in found_drugs:
        print(drug)
else:
    print("Không tìm thấy tên thuốc nào.")
print("=="*25)