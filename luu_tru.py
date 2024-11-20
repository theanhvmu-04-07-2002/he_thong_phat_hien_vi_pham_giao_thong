                      
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
                    #----------------------------------------------------
                    
                    # trang_thai_xe = {} 
                    labels = {}
                    # track_states = {}
                    
                    tracks = track_light.update_tracks(detect, frame=roi)
                    for i, track in enumerate(tracks):
                        if track.is_confirmed() and track.det_conf:
                            ltrb = track.to_ltrb()
                            x1, y1, x2, y2 = list(map(int, ltrb))
                            yc = roi_y1 + (y1 + (y2 - y1) // 2)  
                            xc = roi_x1 + (x1 + (x2 - x1) // 2) 
                            track_id = track.track_id
                            
                            if track_id not in trang_thai_xe:
                                trang_thai_xe[track_id] = 0   
                            current_state = trang_thai_xe.get(track_id, 0)
                            
                            label = labels.get(track_id, None)
                            xoa_du_lieu = None
                        
                            if yc > (y_line + y_line_buffer): 
                                trang_thai_xe[track_id] = 2  
                                label = "k vuot"
                                labels[track_id] = label
    
                            elif (y_line - y_line_buffer) <= yc <= y_line and red:  
                                if(current_state == 0):
                                    trang_thai_xe[track_id] = 1
                                    label = "vuot"
                                    labels[track_id] = label
                                    print(f" Đang vượt đèn đỏ: Track ID: {track_id}, Label: {label}, Trang Thai Xe: {trang_thai_xe[track_id]}")
                            
                            print(f" Trước khi vào vùng: Track ID: {track_id}, Label: {label}, Trang Thai Xe: {trang_thai_xe[track_id]}")
                            if right_roi_x1 <= xc <= right_roi_x1 + right_roi_buffer and red:  
                                print(f" Track ID: {track_id}, vào vùng:, Label: {label}, Trang Thai Xe: {trang_thai_xe[track_id]}")
                                if  trang_thai_xe[track_id] == 1:
                                    xoa_du_lieu = "co_xoa"
                                    label = "k vuot"
                                    trang_thai_xe[track_id] = 2
                                    labels[track_id] = label
                                    print(f" Track ID: {track_id}, sau khi vào vùng, Label: {label}, Trang Thai Xe: {trang_thai_xe[track_id]}")

                            # updated_state = trang_thai_xe.get(track_id, 0)
                            if track_id not in labels:
                                den_phuong_tien[track_id] = red
                                label = None
                            
                            
                            if label:
                                frame_with_box = original_frame.copy()
                                color = (0, 0, 255) if label == 'vuot' else (0, 255, 0)
                                label_with_id = f"{label} (ID: {track_id} {trang_thai_xe[track_id]})"
                                color = (0, 0, 255) if label == 'vuot' else (0, 255, 0)
                                cv2.rectangle(roi, (x1, y1), (x2, y2), color, 2)
                                cv2.rectangle(roi, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), color, -1) 
                                cv2.putText(roi, label_with_id, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # hiện nhãn
                                
                                # if label == 'vuot':
                                    
                                #     original_image_path, violation_image_path = luu_anh_vi_pham(original_frame, frame_with_box, track, label, track_id, roi_x1, roi_y1, VIOLATION_FOLDER)
                                #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                #     executor.submit(save_violation_to_db, current_time, violation_image_path, video_path, "Lỗi vượt đèn đỏ", track_id)

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
                                
