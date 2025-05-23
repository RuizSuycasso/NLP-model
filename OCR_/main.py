# OCR/main.py
import cv2
import pytesseract
# Đảm bảo file rotate_func1.py nằm cùng cấp với main.py
# (Hoặc điều chỉnh sys.path nếu cấu trúc thư mục khác)
from . import rotate_func1
import os
import sys
import re
import torch
import time # Thêm thư viện time để đo thời gian xử lý (tùy chọn, hữu ích cho debug)


# --- Thiết lập đường dẫn và import modules từ NLP-model ---

# Đường dẫn đến thư mục chứa file main.py (ví dụ: your_workspace/OCR/)
THIS_DIR = os.path.dirname(__file__)

# Đường dẫn đến thư mục gốc của workspace (ví dụ: your_workspace/)
# Bằng cách đi lên một cấp từ THIS_DIR
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

# Đường dẫn đến thư mục NLP-model (ví dụ: your_workspace/NLP-model/)
NLP_MODEL_DIR = os.path.join(PROJECT_DIR, "NLP-model")

# Thêm đường dẫn đến thư mục NLP-model vào sys.path.
# Điều này cho phép Python tìm thấy các file model.py và utils.py
# khi chúng ta thực hiện import model và utils.
# Kiểm tra để tránh thêm nhiều lần nếu script được chạy nhiều lần trong cùng process
if NLP_MODEL_DIR not in sys.path:
    sys.path.insert(0, NLP_MODEL_DIR) # Sử dụng insert(0, ...) để ưu tiên thư mục project

# Bây giờ có thể import các class và hàm từ NLP-model
# (Vì NLP_MODEL_DIR đã có trong sys.path)
try:
    from model import BiLSTM_CRF
    from utils import load_pickle, smart_tokenize, extract_entities
except ImportError as e:
    print(f"Lỗi: Không thể import các module từ NLP-model. Hãy đảm bảo thư mục '{NLP_MODEL_DIR}' tồn tại và chứa các file model.py, utils.py.")
    print(f"Lỗi chi tiết: {e}")
    sys.exit(1)


# 1. Nạp các tài nguyên cho mô hình NLP (tải 1 lần khi module được import)
# Đặt khối nạp này ngoài hàm để nó chỉ chạy một lần khi main.py được import bởi app.py
try:
    print("--- Bắt đầu nạp tài nguyên NLP ---")
    start_time_load = time.time()

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
    print(f"Sử dụng thiết bị: {device}")

    nlp_model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tagset_size=len(label2idx),
        padding_idx=word2idx['<PAD>']
    ).to(device)

    print(f"Đang nạp trọng số mô hình từ: {model_state_dict_path}")
    state_dict = torch.load(model_state_dict_path, map_location=device)

    # --- SỬA LỖI: Ánh xạ tên tham số TỪ CŨ SANG MỚI ---
    new_state_dict = {}
    # print("Các key trong state_dict ban đầu:") # Debugging keys
    # print(state_dict.keys())
    # print("Các key trong mô hình hiện tại:") # Debugging keys
    # print(nlp_model.state_dict().keys())
    for key, value in state_dict.items():
        if key == "crf.start_trans": new_key = "crf.start_transitions"
        elif key == "crf.end_trans": new_key = "crf.end_transitions"
        elif key == "crf.trans_matrix": new_key = "crf.transitions"
        elif key == "start_trans": new_key = "start_transitions"
        elif key == "end_trans": new_key = "end_transitions"
        elif key == "trans_matrix": new_key = "transitions"
        else: new_key = key

        if new_key in nlp_model.state_dict():
             new_state_dict[new_key] = value
        # else:
        #      print(f"Cảnh báo: Bỏ qua key không khớp trong state_dict: {key} (ánh xạ thành {new_key})") # Cảnh báo này hữu ích khi debug

    try:
        # Sử dụng strict=False để bỏ qua các key bị thiếu trong checkpoint (ví dụ optimizer state)
        # và các key trong mô hình không có trong checkpoint (sau ánh xạ).
        nlp_model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
         print(f"Lỗi nghiêm trọng khi nạp state_dict sau khi ánh xạ tên: {e}")
         print("Vui lòng kiểm tra sự khác biệt giữa key trong checkpoint và key trong mô hình.")
         sys.exit(1)

    nlp_model.eval() # Chuyển mô hình sang chế độ đánh giá/dự đoán
    end_time_load = time.time()
    print(f"Đã nạp mô hình NLP thành công sau {end_time_load - start_time_load:.2f} giây.")
    print("--- Kết thúc nạp tài nguyên NLP ---")


except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file cần thiết cho mô hình NLP: {e}")
    if "bilstm_crf_best.pt" in str(e):
         print(f"Hãy đảm bảo file bilstm_crf_best.pt nằm trong thư mục: {PROJECT_DIR}")
    elif "pkl" in str(e):
        print(f"Hãy đảm bảo các file .pkl (word2idx.pkl, label2idx.pkl) nằm trong thư mục: {NLP_MODEL_DIR}")
    else:
         print(f"Không tìm thấy file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi không xác định khi nạp mô hình NLP: {e}")
    sys.exit(1)


# --- Thiết lập Tesseract ---
# Bỏ dòng thiết lập đường dẫn cứng nhắc trên Windows.
# pyterseract sẽ tự động tìm tesseract executable trong PATH của hệ thống.
# Trên Render, Tesseract sẽ được cài đặt vào PATH thông qua file apt-packages.
# tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <-- ĐÃ BỎ
# pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path # <-- ĐÃ BỎ

# Kiểm tra xem Tesseract có thể được tìm thấy không
# Thử gọi phiên bản Tesseract để kiểm tra
try:
    pytesseract.get_tesseract_version()
    print("Đã tìm thấy Tesseract OCR trong PATH của hệ thống.")
except pytesseract.TesseractNotFoundError:
    print("Lỗi: Không tìm thấy Tesseract OCR executable trong PATH của hệ thống.")
    print("Hãy đảm bảo Tesseract đã được cài đặt và đường dẫn của nó được thêm vào biến môi trường PATH.")
    print("Trên Render, hãy tạo file 'apt-packages' ở thư mục gốc với nội dung 'tesseract-ocr' (và các ngôn ngữ nếu cần).")
    sys.exit(1) # Thoát nếu không tìm thấy Tesseract lúc khởi động


# Hàm mã hóa câu (inputs và mask) - cần thiết cho mô hình NLP
def encode(tokens, vocab):
    """
    Mã hóa tokens thành IDs và mask cho model.
    Sử dụng độ dài tokens gốc.
    """
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    mask = [1] * len(ids)
    return (
        torch.LongTensor([ids]).to(device),
        torch.BoolTensor([mask]).to(device)
    )

# --- Định nghĩa hàm process_image để app.py gọi ---
def process_image(image_path):
    """
    Processes an image through OCR and NLP to extract drug names.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        list: A list of lowercased, filtered drug names found in the image,
              or an empty list if no drugs are found or processing fails.
    Raises:
        FileNotFoundError: If the image file does not exist.
        IOError: If the image file cannot be read.
        RuntimeError: If Tesseract OCR fails or NLP processing encounters a critical error.
    """
    print(f"--- Bắt đầu xử lý ảnh: {image_path} ---")
    start_time_process = time.time()

    # read image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại đường dẫn: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Không thể đọc file ảnh: {image_path}")
    print("Đã đọc ảnh.")

    # rotate (sửa nghiêng)
    try:
        start_time_deskew = time.time()
        fixed = rotate_func1.deskew_image(img, angle_threshold=1)
        end_time_deskew = time.time()
        print(f"Đã sửa nghiêng ảnh sau {end_time_deskew - start_time_deskew:.2f} giây.")
    except Exception as e:
        print(f"Cảnh báo: Lỗi khi sửa nghiêng ảnh '{image_path}': {e}")
        print("Tiếp tục xử lý ảnh gốc...")
        fixed = img


    # RGB to Gray and threshold
    start_time_preprocess = time.time()
    gray = cv2.cvtColor(fixed, cv2.COLOR_RGB2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 21)
    end_time_preprocess = time.time()
    print(f"Đã tiền xử lý ảnh (grayscale, adaptive threshold) sau {end_time_preprocess - start_time_preprocess:.2f} giây.")

    # Run OCR
    custom_config = r'--oem 3 --psm 6'
    try:
        print("Đang chạy Tesseract OCR...")
        start_time_ocr = time.time()
        # pyterseract sẽ tự tìm tesseract trong PATH
        text = pytesseract.image_to_string(adaptive_thresh, lang="vie+eng", config=custom_config)
        end_time_ocr = time.time()
        print(f"Đã hoàn thành Tesseract OCR sau {end_time_ocr - start_time_ocr:.2f} giây.")

        if not text or not text.strip():
            print(f"OCR không phát hiện được văn bản từ ảnh '{image_path}' hoặc văn bản rỗng.")
            return []
        print(f"Văn bản OCR:\n---\n{text.strip()}\n---") # In văn bản OCR đã strip

    except pytesseract.TesseractNotFoundError:
         # Lỗi này đã được kiểm tra lúc khởi động, nhưng bắt lại ở đây cũng an toàn
         raise RuntimeError("Tesseract executable not found in PATH.")
    except Exception as e:
        # Bắt các lỗi khác từ Tesseract (ví dụ: ngôn ngữ không tồn tại)
        raise RuntimeError(f"Lỗi khi chạy Tesseract OCR cho ảnh '{image_path}': {e}") from e

    # Tokenize and structure sentences
    start_time_tokenize = time.time()
    ocr_lines = text.strip().splitlines()
    sentences = []
    for line in ocr_lines:
        tokens = smart_tokenize(line.strip().lower())
        if tokens:
            sentences.append(tokens)
    end_time_tokenize = time.time()
    print(f"Đã tách thành {len(sentences)} câu sau khi token hóa trong {end_time_tokenize - start_time_tokenize:.2f} giây.")

    if not sentences:
        print(f"Không có câu nào được tạo từ văn bản OCR của ảnh '{image_path}' sau khi token hóa.")
        return []

    # Predict NLP and extract entities
    found_drugs = []
    start_time_nlp = time.time()
    with torch.no_grad():
        print("Đang chạy mô hình NLP...")
        for i, tokens in enumerate(sentences):
            if not tokens:
                continue

            # Mã hóa câu thành Tensor ID và Mask
            X, M = encode(tokens, word2idx)

            try:
                # Đưa qua mô hình NLP để dự đoán nhãn
                # model(X, mask=M) trả về list[list[int]] từ crf.decode
                pred_ids = nlp_model(X, mask=M)

                if isinstance(pred_ids, list) and len(pred_ids) > 0:
                     pred_ids_list = pred_ids[0] # Lấy kết quả cho batch size 1
                else:
                     print(f"Cảnh báo: Dự đoán trả về định dạng không mong muốn cho câu {i+1} ('{' '.join(tokens)}'). Bỏ qua.")
                     continue

                # Đảm bảo độ dài dự đoán và token khớp
                if len(pred_ids_list) != len(tokens):
                     print(f"Cảnh báo: Độ dài nhãn dự đoán ({len(pred_ids_list)}) không khớp độ dài token ({len(tokens)}) cho câu {i+1}. Bỏ qua.")
                     continue

                # Ánh xạ các ID dự đoán sang chuỗi nhãn
                pred_labels = [idx2label[i_pred] for i_pred in pred_ids_list]
                # print(f"Câu {i+1}: Tokens={tokens}, Labels={pred_labels}") # Debugging labels

                # Gom các token và nhãn thành các thực thể
                entities = extract_entities(tokens, pred_labels)
                drugs = entities.get('DRUG', [])
                # print(f"Câu {i+1}: Entities={entities}") # Debugging entities

                # Lọc danh sách tên thuốc theo các tiêu chí
                filtered_drugs = []
                if drugs:
                     filtered_drugs = [d for d in drugs if isinstance(d, str) and re.search(r"[A-Za-z]", d) and len(d) > 2]
                     filtered_drugs = [d for d in filtered_drugs if isinstance(d, str) and not re.fullmatch(r"[\d\.\,\-\/\\]+", d)]

                # Thêm các tên thuốc mới tìm được vào danh sách cuối cùng (loại bỏ trùng lặp, chuyển về chữ thường)
                for d in filtered_drugs:
                    clean_d = d.lower()
                    if clean_d not in found_drugs:
                        found_drugs.append(clean_d)

            except Exception as e:
                # Không thoát, chỉ in cảnh báo và bỏ qua câu lỗi
                print(f"Cảnh báo: Lỗi khi dự đoán câu {i+1} ('{' '.join(tokens)}') từ ảnh '{image_path}': {e}")
                print("Bỏ qua câu này và tiếp tục...")
                continue # Tiếp tục xử lý các câu khác

    end_time_nlp = time.time()
    print(f"Hoàn thành xử lý NLP trong {end_time_nlp - start_time_nlp:.2f} giây.")
    print(f"Tìm thấy {len(found_drugs)} tên thuốc: {found_drugs}") # Debugging found drugs

    end_time_process = time.time()
    print(f"--- Kết thúc xử lý ảnh {image_path} sau {end_time_process - start_time_process:.2f} giây ---")

    return found_drugs

# --- Phần chạy test độc lập (chỉ chạy khi file main.py được thực thi trực tiếp) ---
# Đã được comment lại để file hoạt động như một module khi được import bởi app.py
# Bạn có thể uncomment phần này để test main.py riêng lẻ nếu cần
# if __name__ == '__main__':
#     print("Chạy chế độ test độc lập cho main.py...")
#     # Đường dẫn ảnh test - Cần điều chỉnh cho phù hợp với cấu trúc thư mục project của bạn
#     # Ví dụ: Ảnh nằm trong thư mục Pic ở cùng cấp với OCR/
#     # img_path_for_test = os.path.join(PROJECT_DIR, "Pic", "test_pic.jpg")
#     # Hoặc đường dẫn tuyệt đối cho test cục bộ:
#     img_path_for_test = r"C:\Users\Precision\Desktop\AI-test\Pic\test_pic.jpg" # <-- SỬA ĐƯỜNG DẪN NÀY CHO TEST

#     if not os.path.exists(img_path_for_test):
#          print(f"Lỗi: Không tìm thấy file ảnh test tại đường dẫn: {img_path_for_test}")
#          print("Vui lòng cập nhật đường dẫn ảnh test trong khối if __name__ == '__main__':")
#          sys.exit(1)

#     try:
#         print(f"Đang xử lý ảnh test: {img_path_for_test}")
#         results = process_image(img_path_for_test)
#         print("\n===== Danh sách thuốc phát hiện từ ảnh test =====")
#         if results:
#             # In kết quả đã được lowercased
#             for drug in results:
#                 print(drug)
#         else:
#             print("Không tìm thấy tên thuốc nào trong ảnh test.")
#         print("=="*25)
#     except Exception as e:
#          print(f"Lỗi nghiêm trọng khi xử lý ảnh test: {e}")
#          # In traceback đầy đủ để debug
#          import traceback
#          traceback.print_exc()
#          sys.exit(1)