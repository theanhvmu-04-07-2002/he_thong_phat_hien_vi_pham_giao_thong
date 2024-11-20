from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import cv2
import sqlite3
from ultralytics import YOLO
from urllib.parse import parse_qs
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor 
from deep_sort_realtime.deepsort_tracker import DeepSort 
import numpy as np
from utils import *
import json
import math
import traceback

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/violations", StaticFiles(directory="violations"), name="violations")
templates = Jinja2Templates(directory="templates")

VIDEO_BASE_PATH = './data/video'
MODEL_PATH_VEHICLE = './model/best.pt'
MODEL_PATH_HELMET = './model/best_helmet_end.pt'
VIOLATION_FOLDER = './violations'
os.makedirs(VIOLATION_FOLDER, exist_ok=True)
active_websocket = None

# Deep Sort trackers
track_vehicle = DeepSort(max_age=60)
track_helmet = DeepSort(max_age=60)
track_light = DeepSort(max_age=60)

# Load YOLO models
model_vehicle = YOLO(MODEL_PATH_VEHICLE)
model_helmet = YOLO(MODEL_PATH_HELMET)

# Initialize SQLite database
DATABASE_PATH = './violations.db'
conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    image_path TEXT,
                    video_path TEXT,
                    category nvarchar,
                    track_id varchar
                  )''')
conn.commit()

executor = ThreadPoolExecutor()

# def save_violation_to_db(timestamp, image_path, video_path, category):
#     video_name = os.path.basename(video_path)
#     cursor.execute('INSERT INTO violations (timestamp, image_path, video_path, category, track_id) VALUES (?, ?, ?, ?, ?)',
#                    (timestamp, image_path, video_name, category, track_id))
#     conn.commit()
def save_violation_to_db(timestamp, image_path, video_path, category, track_id):
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            video_name = os.path.basename(video_path)
            cursor.execute('INSERT INTO violations (timestamp, image_path, video_path, category, track_id) VALUES (?, ?, ?, ?, ?)',
                           (timestamp, image_path, video_name, category, track_id))
            conn.commit()
            print(f"Lưu dữ liệu Thành công: {timestamp}, {image_path}, {video_name}, {category}, {track_id}")
    except sqlite3.Error as e:
        print(f"Lỗi lưu dữ liệu save_violation_to_db: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/tables", response_class=HTMLResponse)
async def get_tables(request: Request):
    return templates.TemplateResponse("tables.html", {"request": request})

@app.get("/camera", response_class=HTMLResponse)
async def get_camera(request: Request):
    video_files = os.listdir(VIDEO_BASE_PATH)
    return templates.TemplateResponse("camera.html", {"request": request, "video_files": video_files, "enumerate": enumerate})

def get_all_violations():
    cursor.execute("SELECT * FROM violations")
    violations = cursor.fetchall()
    print(violations)  # In ra để kiểm tra xem có dữ liệu hay không
    return violations
@app.get("/api/violations")
async def read_violations():
    violations = get_all_violations()
    return JSONResponse(content={"data": violations})
@app.get("/api/statistics")
async def get_statistics():
    cursor.execute('SELECT COUNT(*) FROM violations WHERE category = "Lane Violation"')
    lane_violations = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM violations WHERE category = "No Helmet"')
    no_helmet_violations = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM violations WHERE category = "Light Violation"')
    light_violations = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM violations')
    total_violations = cursor.fetchone()[0]

    return JSONResponse(content={
        "lane_violations": lane_violations,
        "no_helmet_violations": no_helmet_violations,
        "light_violations": light_violations,
        "total_violations": total_violations
    })
@app.get("/api/violations_daily")
async def get_violations_daily():
    cursor.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM violations
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """)
    violations_daily = cursor.fetchall()
    data = [{"date": row[0], "count": row[1]} for row in violations_daily]
    return JSONResponse(content={"data": data})

@app.websocket("/ws/{video_name}")
async def websocket_endpoint(websocket: WebSocket, video_name: str, roi_x1: int = 100, roi_y1: int = 100, roi_x2: int = 1150, roi_y2: int = 750):
    global active_websocket
    if active_websocket:
        await active_websocket.close()

    active_websocket = websocket
    await websocket.accept()
    video_path = os.path.join(VIDEO_BASE_PATH, video_name)

    # Parse query parameters
    query_params = parse_qs(websocket.url.query)
    detection_type = query_params.get('detectionType', ['none'])[0].lower()
    left_labels = json.loads(query_params.get('leftLabels', ['[]'])[0])
    right_labels = json.loads(query_params.get('rightLabels', ['[]'])[0])

    print(f"Connected to {video_name}")
    print(f"Detection Type: {detection_type}")
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop to the region of interest
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            original_frame = frame.copy()
            
            if detection_type == 'vehicle' or detection_type == 'helmet':
                detect_vehicle = []
                results_vehicle = model_vehicle.predict(source=roi, conf=0.6, iou=0.65)[0]

                for box in results_vehicle.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    detect_vehicle.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

                current_track_vehicle = track_vehicle.update_tracks(detect_vehicle, frame=roi)

                for i, track in enumerate(current_track_vehicle):
                    if not (track.is_confirmed() and track.det_conf):
                        continue

                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = list(map(int, ltrb))
                    track_id = track.track_id
                    label = track.det_class
                    confidence = track.det_conf
                    track_id = track.track_id

                    if confidence > 0.65:
                        text = f"{model_vehicle.names[int(label)]}, id: {track_id}, conf: {round(confidence, 2)}"
                        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(roi, text, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if detection_type == 'helmet':
                    for i, track in enumerate(current_track_vehicle):
                        if not (track.is_confirmed() and track.det_conf):
                            continue

                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = list(map(int, ltrb))
                        crop_img = roi[y1:y2, x1:x2]
                        if crop_img.size != 0:

                            results_helmet = model_helmet.predict(source=crop_img, imgsz=320, conf=0.45, iou=0.45)[0]
                            
                            for helmet_box in results_helmet.boxes:
                                hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0].tolist())
                                hlabel = helmet_box.cls[0]
                                hconfidence = helmet_box.conf[0]
                                htext = f"{model_helmet.names[int(hlabel)]}: {hconfidence:.2f}"
                                cv2.rectangle(roi, (x1 + hx1, y1 + hy1), (x1 + hx2, y1 + hy2), (0, 0, 255) if model_helmet.names[int(hlabel)] == "Without Helmet" else (0, 255, 0), 1)
                                cv2.putText(roi, htext, (x1 + hx1, y1 + hy1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255) if model_helmet.names[int(hlabel)] == "Without Helmet" else (255, 0, 0), 1, cv2.LINE_AA)

                                if hconfidence > 0.65 and model_helmet.names[int(hlabel)] == "Without Helmet":
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    violation_image_path = os.path.join(VIOLATION_FOLDER, f"khong_doi_mu_{timestamp}.jpg")
                                    cv2.imwrite(violation_image_path, crop_img)

                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    luu = executor.submit(save_violation_to_db, current_time, violation_image_path, video_path, "Không đội mũ", "track_id")
                                    try:
                                        # Đợi kết quả và kiểm tra thành công hay thất bại
                                        luu.result()
                                        print("Lưu thông tin thành công.")
                                    except Exception as e:
                                        print(f"Lỗi khi lưu thông tin: {str(e)}")

            elif detection_type == 'lane':
                
                roi_x1  = int(query_params.get('roi_x1', [0])[0])
                roi_y1  = int(query_params.get('roi_y1', [0])[0])
                roi_x2  = int(query_params.get('roi_x2', [0])[0])
                roi_y2  = int(query_params.get('roi_y2', [0])[0])
                
                left_labels = json.loads(query_params.get('leftLabels', ['[]'])[0]) 
                right_labels = json.loads(query_params.get('rightLabels', ['[]'])[0])
                
                midpoint_x = roi_x1 + (roi_x2 - roi_x1) // 2  

                cv2.line(frame, (midpoint_x, roi_y1), (midpoint_x, roi_y2), (0, 0, 255), 2) 

                results_vehicle = model_vehicle.predict(source=roi, imgsz=320, conf=0.3, iou=0.4)[0] 
                for box in results_vehicle.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
                    vehicle_center_x = (x1 + x2) // 2 
                    vehicle_label = model_vehicle.names[int(box.cls.item())] 

                    if vehicle_label in left_labels:
                        lane = "Left"
                    elif vehicle_label in right_labels:
                        lane = "Right"
                    else:
                        continue  
                    
                    if (lane == "Left" and vehicle_center_x > midpoint_x) or (lane == "Right" and vehicle_center_x < midpoint_x):
                        alert_message = f"Violation!!! Lane of vehicles"
                        violation_detected = True
                        color = (0, 0, 255)  
                    else:
                        continue  
                    
                    cv2.rectangle(frame, (roi_x1 + x1, roi_y1 + y1), (roi_x1 + x2, roi_y1 + y2), color, 2)
                    cv2.putText(frame, f"{vehicle_label} - {alert_message}", (roi_x1 + x1, roi_y1 + y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)

                    if violation_detected:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        violation_image_path = os.path.join(VIOLATION_FOLDER, f"di_sai_lan_{timestamp}.jpg")
                        frame_with_bounding_box = frame.copy()
                        cv2.rectangle(frame_with_bounding_box, (roi_x1 + x1, roi_y1 + y1), (roi_x1 + x2, roi_y1 + y2), color, 2)
                        cv2.putText(frame_with_bounding_box, f"{vehicle_label} - {alert_message}", (roi_x1 + x1, roi_y1 + y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)
                        cv2.imwrite(violation_image_path, frame_with_bounding_box)

                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        luu = executor.submit(save_violation_to_db, current_time, violation_image_path, video_path, "Lỗi sai làn", "0")
                        try:
                            luu.result()
                            print ("Lưu dữ liệu thành công")
                        except Exception as e:
                            print(f"Lỗi khi lưu thông tin: {str(e)}")

            
            if detection_type == 'light':
                trang_thai_den = 0 
                den_phuong_tien = {} 
                trang_thai_xe = {} 
                saverd_images = {}
                #re_phai = None
                
                roi_x1  = int(query_params.get('roi_x1', [0])[0])
                roi_y1  = int(query_params.get('roi_y1', [0])[0])
                roi_x2  = int(query_params.get('roi_x2', [0])[0])
                roi_y2  = int(query_params.get('roi_y2', [0])[0])
                
                # lấy radio
                right_turn_allowed = query_params.get('rightTurnAllowed', ['not_allowed'])[0]
                # den do
                light_roi_x1 = int(query_params.get('lightRoiX1', [0])[0])
                light_roi_y1 = int(query_params.get('lightRoiY1', [0])[0])
                light_roi_x2 = int(query_params.get('lightRoiX2', [0])[0])
                light_roi_y2 = int(query_params.get('lightRoiY2', [0])[0])
                y_line = int(query_params.get('yLine', [450])[0])
                y_line_buffer = 200
                # re phai  
                right_roi_x1  = int(query_params.get('rightX1', [0])[0])
                right_roi_y1  = int(query_params.get('rightY1', [0])[0])
                right_roi_x2  = int(query_params.get('rightX2', [0])[0])
                right_roi_y2  = int(query_params.get('rightY2', [0])[0])
                right_roi_buffer = 100
                # print(right_roi_y1)
                # y_line_buffer = 200 
                color_right_line = (0,255,255)
                cv2.putText(frame, "A", (right_roi_x1, right_roi_y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color_right_line, 2)
                cv2.putText(frame, "B", (right_roi_x2, right_roi_y2), cv2.FONT_HERSHEY_SIMPLEX, 1, color_right_line, 2)
                cv2.line(frame, (right_roi_x1, right_roi_y1), (right_roi_x2, right_roi_y2), color_right_line, 2)
                
                # bootom lane
                right_roi_x2 = int(query_params.get('lightRoiX2', [0])[0])
                right_roi_y2 = int(query_params.get('lightRoiY2', [0])[0])
                
                light_frame = frame[light_roi_y1:light_roi_y2, light_roi_x1:light_roi_x2, :]
                red, trang_thai_den, _ = is_red(light_frame, tich_luy_hien_tai=trang_thai_den)
                text = 'RED' if red else 'GREEN'
                light_color = (0, 0, 255) if red else (0, 255, 0)
                
                cv2.putText(frame, f"Light: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)
                
                cv2.rectangle(frame, (light_roi_x1, light_roi_y1), (light_roi_x2, light_roi_y2), (0, 255, 255), 2) # vẽ roi
                cv2.line(frame, (0, y_line), (frame.shape[1], y_line), (0, 255, 255), 2) # vẽ y line màu vàng
                result = model_vehicle.predict(roi, conf=0.35, verbose=False) 
                if len(result): 
                    result = result[0] 
                    names = result.names
                    detect = [] 
                    track_image_count = {}
                    for box in result.boxes: 
                        x1, y1, x2, y2 = list(map(int, box.xyxy[0])) 
                        conf = box.conf.item() 
                        cls = int(box.cls.item()) 
                        detect.append([[x1, y1, x2 - x1, y2 - y1], conf, cls]) 
                        
                    def xoa_data_vi_pham(track_id, xoa_du_lieu):
                        try:
                            with sqlite3.connect(DATABASE_PATH) as conn:
                                cursor = conn.cursor()
                                if xoa_du_lieu == 'co_xoa':
                                    # xóa ảnh
                                    cursor.execute('SELECT image_path FROM violations WHERE track_id = ?', (track_id,))
                                    rows = cursor.fetchall()
                                    for row in rows:
                                        image_path = row[0]
                                        if os.path.exists(image_path):
                                            os.remove(image_path)

                                    # xóa csdl
                                    cursor.execute('DELETE FROM violations WHERE track_id = ?', (track_id,))
                                    conn.commit()
                                    print(f"Xóa dữ liệu thành công track_id {track_id} ")
                        except sqlite3.Error as e:
                            print(f"Xoá dữ liệu thất bại xoa_data_vi_pham: {e}")
                    def luu_anh_vi_pham(frame, frame_with_box, track, label, track_id, roi_x1, roi_y1, output_folder):
                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = list(map(int, ltrb))
                        
                        color = (0, 0, 255)  
                        cv2.rectangle(frame_with_box, (roi_x1 + x1, roi_y1 + y1), (roi_x1 + x2, roi_y1 + y2), color, 2)
                        

                        label_with_id = f"{label} (ID: {track_id})"
                        cv2.rectangle(frame_with_box, (roi_x1 + x1 - 1, roi_y1 + y1 - 20), (roi_x1 + x1 + len(label_with_id) * 12, roi_y1 + y1), color, -1)
                        cv2.putText(frame_with_box, label_with_id, (roi_x1 + x1 + 5, roi_y1 + y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        original_image_path = os.path.join(output_folder, f"original_frame_{timestamp}.jpg")
                        # cv2.imwrite(original_image_path, frame)
                        # print(f"Lưu ảnh gốc thành công: {original_image_path}")
                        
                        image_path_with_box = os.path.join(output_folder, f"vuot_den_do_{timestamp}.jpg")
                        cv2.imwrite(image_path_with_box, frame_with_box)
                        print(f"Lưu ảnh với box thành công: {image_path_with_box}")

                        return original_image_path, image_path_with_box
                    #----------------------------------------------------
                    
                    # trang_thai_xe = {} 
                    labels = {}
                    tracks = track_light.update_tracks(detect, frame=roi)
                    for i, track in enumerate(tracks):
                        if track.is_confirmed() and track.det_conf:
                            ltrb = track.to_ltrb()
                            x1, y1, x2, y2 = list(map(int, ltrb))
                            yc = roi_y1 + (y1 + (y2 - y1) // 2)  
                            xc = roi_x1 + (x1 + (x2 - x1) // 2) 
                            track_id = track.track_id
                            # current_state = trang_thai_xe.get(track_id, 0)  
                            
                            if track_id not in trang_thai_xe:
                                trang_thai_xe[track_id] = 0
                            
                            label = labels.get(track_id, None)
                            xoa_du_lieu = None
                            
                            if yc > (y_line + y_line_buffer): 
                                trang_thai_xe[track_id] = 2  
                                label = "k vuot"
                                labels[track_id] = label
    
                            if (y_line - y_line_buffer) <= yc <= y_line and red:  
                                trang_thai_xe[track_id] = 1
                                label = "vuot"
                                labels[track_id] = label
                            
                            # print(f" trước khi vào vùng: Track ID: {track_id}, Label: {label}, Trang Thai Xe: {trang_thai_xe[track_id]}")
                            if right_roi_x1 <= xc <= right_roi_x1 + right_roi_buffer and red:  
                                print(f" Track ID: {track_id}, trước khi vào vùng:, Label: {label}, Trang Thai Xe: {trang_thai_xe[track_id]}")
                                if  trang_thai_xe[track_id]==0:
                                    xoa_du_lieu = "co_xoa"
                                    label = "k vuot"
                                    trang_thai_xe[track_id] = 2
                                    labels[track_id] = label
                                    print(f" Track ID: {track_id}, sau khi vào vùng, Label: {label}, Trang Thai Xe: {trang_thai_xe[track_id]}")
                            else:
                                # if current_state == 0:
                                den_phuong_tien[track_id] = red
                                
                            # updated_state = trang_thai_xe.get(track_id, 0)

                            if label:
                                frame_with_box = original_frame.copy()
                                color = (0, 0, 255) if label == 'vuot' else (0, 255, 0)
                                label_with_id = f"{label} (ID: {track_id} {trang_thai_xe[track_id]})"
                                color = (0, 0, 255) if label == 'vuot' else (0, 255, 0)
                                cv2.rectangle(roi, (x1, y1), (x2, y2), color, 2)
                                cv2.rectangle(roi, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), color, -1) 
                                cv2.putText(roi, label_with_id, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # hiện nhãn
                                
                                if label == 'vuot':
                                    
                                    original_image_path, violation_image_path = luu_anh_vi_pham(original_frame, frame_with_box, track, label, track_id, roi_x1, roi_y1, VIOLATION_FOLDER)
                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    executor.submit(save_violation_to_db, current_time, violation_image_path, video_path, "Lỗi vượt đèn đỏ", track_id)

                                # if xoa_du_lieu == 'co_xoa':
                                #     if right_turn_allowed == 'duoc_re_phai':
                                #         executor.submit(xoa_data_vi_pham, track_id, xoa_du_lieu)
                
                #-------------------------------------------------------------------
           
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)

            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        active_websocket = None
        await websocket.close()
                                
