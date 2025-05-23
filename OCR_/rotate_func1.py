import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

def preprocess_image(image):
    """Tăng cường chất lượng ảnh trước khi xử lý"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng CLAHE để cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Nhị phân hóa ảnh
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return binary

def get_skew_angle_hough(image):
    """Phát hiện góc nghiêng bằng phương pháp HoughLines"""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Tính góc nghiêng (đơn vị độ)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Lọc bỏ các góc outlier (quá khác biệt so với median)
    median_angle = np.median(angles)
    filtered_angles = [a for a in angles if abs(a - median_angle) < 10]
    
    # Trả về góc trung bình sau khi lọc
    return np.mean(filtered_angles) if filtered_angles else 0.0

def get_skew_angle_tesseract(image):
    """Phát hiện góc nghiêng bằng Tesseract OCR"""
    try:
        # Tesseract hoạt động tốt hơn với ảnh đen trắng
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Sử dụng OSD (Orientation and Script Detection)
        data = pytesseract.image_to_osd(binary, config="--psm 0")
        
        # Trích xuất góc từ kết quả
        angle = float(data.split("\n")[1].split(":")[1])
        return angle
    except Exception as e:
        print(f"Tesseract error: {e}")
        return 0.0

def rotate_image(image, angle):
    """Xoay ảnh theo góc xác định"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Tính ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Thực hiện xoay ảnh
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def deskew_image(image, angle_threshold=0.5):
    """Chỉnh sửa độ nghiêng của ảnh"""
    # Tiền xử lý ảnh
    processed = preprocess_image(image)
    
    # Phát hiện góc bằng cả hai phương pháp
    angle_hough = get_skew_angle_hough(processed)
    angle_tess = get_skew_angle_tesseract(processed)
    
    # print(f"[DEBUG] Hough angle: {angle_hough:.2f}°, Tesseract angle: {angle_tess:.2f}°")
    
    # Quy tắc chọn góc:
    # 1. Ưu tiên Tesseract nếu góc Hough quá lớn (có thể nhiễu)
    # 2. Ngược lại chọn góc có độ lớn hơn
    if abs(angle_hough) > 10:  # Nếu góc Hough > 10° có thể không đáng tin
        final_angle = angle_tess
    else:
        final_angle = angle_hough if abs(angle_hough) > abs(angle_tess) else angle_tess
    
    # Chỉ xoay nếu góc đủ lớn
    if abs(final_angle) > angle_threshold:
        # print(f"Rotating by {final_angle:.2f}°")
        return rotate_image(image, final_angle)
    
    # print("No significant skew detected")
    return image

