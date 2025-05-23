from flask import Flask, request, jsonify
import os
import uuid
import cv2
# Đảm bảo file main.py được import và có hàm process_image
# Nếu main.py ở trong thư mục OCR, import sẽ là:
# from OCR.main import process_image # Tùy cấu trúc thư mục project gốc

# --- Thiết lập đường dẫn và import process_image từ main.py ---
# Đây là cách tốt hơn để import hàm process_image từ main.py trong thư mục OCR
# Đảm bảo app.py và thư mục OCR nằm cùng cấp trong project gốc
# Ví dụ: your_workspace/app.py và your_workspace/OCR/main.py
import sys

# Đường dẫn đến thư mục chứa app.py (ví dụ: your_workspace/)
APP_DIR = os.path.dirname(__file__) # Đây sẽ là PROJECT_DIR nếu app.py nằm ở gốc

# Nếu app.py nằm ở gốc, OCR/main.py thì cần thêm đường dẫn gốc vào sys.path
# và import từ OCR.main
# Cách đơn giản nhất nếu app.py và OCR cùng cấp thư mục:
# import OCR.main as ocr_processor
# process_image = ocr_processor.process_image # Gọi hàm từ module đã import

# Tuy nhiên, nếu app.py nằm ở gốc và OCR là thư mục con, cách import ban đầu
# from main import process_image sẽ không hoạt động.
# Dựa trên cấu trúc trước đó (app.py có vẻ nằm ngoài thư mục OCR, cùng cấp với NLP-model),
# và main.py nằm trong thư mục OCR, app.py cần biết cách tìm OCR.
# Nếu app.py nằm ở gốc (PROJECT_DIR), thì cần thêm PROJECT_DIR vào sys.path (đã có trong main.py)
# và import từ OCR.main
# Cách import khi app.py ở gốc và main.py ở OCR/:
# import sys
# import os
# PROJECT_DIR_FOR_APP = os.path.dirname(__file__) # Nếu app.py ở gốc project
# if PROJECT_DIR_FOR_APP not in sys.path:
#     sys.path.insert(0, PROJECT_DIR_FOR_APP) # Thêm gốc project vào sys.path
# try:
#     from OCR.main import process_image
# except ImportError as e:
#     print(f"Lỗi: Không thể import process_image từ OCR.main. Hãy đảm bảo cấu trúc thư mục đúng.")
#     print(f"Lỗi chi tiết: {e}")
#     sys.exit(1)

# Giả định đơn giản nhất là app.py và OCR/ cùng cấp, và bạn đang chạy từ thư mục gốc project.
# Lúc này sys.path đã chứa thư mục gốc.
# Import trực tiếp hàm từ OCR.main
try:
     from OCR.main import process_image
except ImportError:
     # Nếu import trên không hoạt động, có thể app.py nằm trong thư mục OCR cùng main.py
     # và cấu trúc thư mục gốc/OCR/main.py là không đúng
     # Hoặc sys.path chưa được thiết lập đúng khi chạy app.py
     # Cách khắc phục: Thiết lập lại sys.path trong app.py tương tự main.py
     THIS_DIR_APP = os.path.dirname(__file__)
     PROJECT_DIR_APP = os.path.abspath(os.path.join(THIS_DIR_APP, os.pardir))
     if PROJECT_DIR_APP not in sys.path:
         sys.path.insert(0, PROJECT_DIR_APP)
     try:
         # Thử lại import sau khi sửa sys.path
         from OCR.main import process_image
     except ImportError as e:
         print(f"Lỗi nghiêm trọng: Không thể import process_image từ OCR.main ngay cả sau khi sửa sys.path.")
         print(f"Hãy kiểm tra lại cấu trúc thư mục và việc cài đặt các dependencies.")
         print(f"Lỗi chi tiết: {e}")
         sys.exit(1)

# ---------------------------------------------------------------------


app = Flask(__name__)

# Thư mục lưu ảnh tạm thời từ người dùng gửi lên
# Đặt thư mục này ngoài OCR/ nếu bạn muốn nó độc lập với code source
# Ví dụ: your_workspace/uploads/
UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/ocr', methods=['POST'])
def ocr():
    # SỬA: Lấy file ảnh theo tên 'file' để khớp với Android
    if 'file' not in request.files:
        return jsonify({"error": "No 'file' part in the request"}), 400

    file = request.files['file'] # <--- Sửa từ 'image' thành 'file'
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = None # Khởi tạo biến image_path
    try:
        # Lưu ảnh tạm thời
        ext = os.path.splitext(file.filename)[-1]
        # Đảm bảo phần mở rộng không rỗng và hợp lệ
        if not ext:
             ext = '.jpg' # Mặc định nếu không có đuôi file
        elif ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
             # Cảnh báo hoặc trả lỗi nếu đuôi file không được hỗ trợ
             print(f"Cảnh báo: Định dạng file '{ext}' có thể không được hỗ trợ tốt.")
             # Có thể chọn trả về lỗi ở đây:
             # return jsonify({"error": f"Unsupported file format: {ext}"}), 415


        filename = f"{uuid.uuid4().hex}{ext}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        print(f"Đã lưu ảnh tạm thời tại: {image_path}") # Debugging

        # Gọi hàm xử lý OCR từ main.py
        # process_image cần nhận image_path
        result = process_image(image_path)

        # Trả kết quả JSON
        return jsonify({"result": result})

    except FileNotFoundError as e:
        print(f"Lỗi File Not Found: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    except IOError as e:
        print(f"Lỗi IO: {e}")
        return jsonify({"error": f"Error reading or writing file: {str(e)}"}), 500
    except RuntimeError as e:
         print(f"Lỗi Runtime (OCR/NLP): {e}")
         return jsonify({"error": f"Processing error: {str(e)}"}), 500
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý request: {e}") # Log lỗi chi tiết trên server
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        # Quan trọng: Xóa ảnh tạm thời sau khi xử lý xong (hoặc gặp lỗi)
        if image_path and os.path.exists(image_path):
             try:
                 os.remove(image_path)
                 print(f"Đã xóa ảnh tạm thời: {image_path}") # Debugging
             except Exception as e:
                 print(f"Cảnh báo: Không thể xóa ảnh tạm thời '{image_path}': {e}")


if __name__ == '__main__':
    # Khi deploy trên Render với Gunicorn, phần này không chạy.
    # Nó chỉ chạy khi bạn chạy file app.py trực tiếp để test.
    # Đặt debug=False khi deploy production.
    app.run(debug=True, port=os.environ.get('PORT', 5000)) # Sử dụng cổng từ biến môi trường cho Render