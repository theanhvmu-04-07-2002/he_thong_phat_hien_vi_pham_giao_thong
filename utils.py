import numpy as np
import cv2

def draw_text(image, text, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=2):
    return cv2.putText(image, text, org, fontFace, fontScale, color, thickness)

# Function to convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
def xyxy2xywh(x):
    y = x.clone()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

# Function to detect if a frame is red
def is_red(light_frame, threshold=0.008, tich_luy_hien_tai=0, tich_luy=3):
    # tích lũy hiện tại: số lần liên tiếp phát hiện đèn đỏ trong các frame trước
    # tích lũy: giá trị tích lũy cần thiết để xác nhận đèn đỏ
    hsv = cv2.cvtColor(light_frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    red_ratio = np.sum(red_mask) / red_mask.size

    if red_ratio > threshold:
        return True, tich_luy_hien_tai + 1, None
    else:
        return False, tich_luy_hien_tai, None

def save_violation_data(track_id, label, image_path, timestamp):
    conn = sqlite3.connect('violations.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS violations
                      (track_id TEXT, label TEXT, image_path TEXT, timestamp TEXT)''')
    cursor.execute("INSERT INTO violations VALUES (?, ?, ?, ?)", (track_id, label, image_path, timestamp))
    conn.commit()
    conn.close()

# Helper function to save image of the entire frame with only the violator's bounding box
def save_frame_snapshot(frame, bbox, image_path):
    # Copy the frame to not affect the display
    frame_copy = np.copy(frame)
    x1, y1, x2, y2 = bbox
    # Draw bounding box for the violator
    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for violator
    cv2.imwrite(image_path, frame_copy)