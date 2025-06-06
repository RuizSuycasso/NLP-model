from flask import Flask, request, jsonify
import os
import uuid
import cv2
import sys
import traceback

# Import process_image từ main.py trong package OCR_
from OCR_.main import process_image


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Thư mục upload ảnh tạm thời: {UPLOAD_FOLDER}")


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint cho Render"""
    return jsonify({"status": "OK", "message": "OCR Service is running"})


@app.route('/ocr', methods=['POST'])
def ocr():
    print("Nhận request /ocr POST")

    if 'file' not in request.files:
        print("Lỗi: Không có phần 'file' trong request.")
        return jsonify({"error": "No 'file' part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        print("Lỗi: Tên file trống.")
        return jsonify({"error": "No selected file"}), 400

    image_path = None
    try:
        ext = os.path.splitext(file.filename)[-1]
        if not ext:
             ext = '.jpg'
        elif ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
             print(f"Cảnh báo: Định dạng file '{ext}' có thể không được hỗ trợ tốt. Tiếp tục...")

        filename = f"{uuid.uuid4().hex}{ext.lower()}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        file.save(image_path)
        print(f"Đã lưu ảnh tạm thời tại: {image_path}")

        print("Đang gọi process_image...")
        result = process_image(image_path)
        print("process_image đã hoàn thành.")

        print(f"Trả về kết quả: {result}")
        return jsonify({"result": result})

    except FileNotFoundError as e:
        print(f"Lỗi File Not Found (xử lý request): {e}")
        return jsonify({"error": f"File error: {str(e)}"}), 500
    except IOError as e:
        print(f"Lỗi IO (xử lý request): {e}")
        return jsonify({"error": f"Error reading or writing file: {str(e)}"}), 500
    except RuntimeError as e:
         print(f"Lỗi Runtime (OCR/NLP) (xử lý request): {e}")
         return jsonify({"error": f"Processing error: {str(e)}"}), 500
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý request: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        if image_path and os.path.exists(image_path):
             try:
                 os.remove(image_path)
                 print(f"Đã xóa ảnh tạm thời: {image_path}")
             except Exception as e:
                 print(f"Cảnh báo: Không thể xóa ảnh tạm thời '{image_path}': {e}")
        else:
            print(f"Không có ảnh tạm thời để xóa hoặc đường dẫn không hợp lệ: {image_path}")


if __name__ == '__main__':
    # Chỉ chạy debug khi test local
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Khởi động Flask app trên port {port}, debug={debug_mode}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
else:
    # Khi chạy với gunicorn trên production
    port = int(os.environ.get('PORT', 5000))
    print(f"Flask app sẵn sàng cho gunicorn trên port {port}")