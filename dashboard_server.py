from flask import Flask, Response, render_template, jsonify
import cv2
from ultralytics import YOLO
import time
import threading
import torch
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = 'model.pt'  # Pastikan file ini benar sesuai yang Anda pakai
model = YOLO(MODEL_PATH)

# Inisialisasi dua kamera: 0 (Ruang 1), 1 (Ruang 2)
CAMERA_IDS = ['0', '1']

# Objek kamera (akan di-reopen bila gagal)
camera_objects = {}

def open_camera(index: int):
    cam = cv2.VideoCapture(index)
    # contoh resolusi yang lebih rendah: 640x480 atau 320x240
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # optional: cek ukuran yang didapat kembali
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Opened camera {index} -> {w}x{h}')
    return cam

for _cid in CAMERA_IDS:
    camera_objects[_cid] = open_camera(int(_cid))

# Struktur frame terbaru per kamera
latest_frames = {
    cam_id: {
        'frame': None,
        'last_read_time': 0.0,
        'lock': threading.Lock(),
        'error_count': 0
    } for cam_id in CAMERA_IDS
}

# Struktur state per kamera
states = {
    cam_id: {
        'last_detection': {
            'status': 'AMAN',
            'status_message': 'AMAN',
            'time': None,
            'confidence_percentage': 0.0,
            'objects_detected': []
        },
        'spark_history': [],  # percikan
        'fire_history': []    # api
    } for cam_id in CAMERA_IDS
}

HISTORY_MAX_LEN = 3
JPEG_QUALITY = 80  # antara 60-85 biasanya seimbang
REOPEN_DELAY = 0.5
STREAM_LOOP_SLEEP = 0.001
INFER_TARGET_FPS = 5  # batas maksimum frekuensi inference per kamera
MIN_INFER_INTERVAL = 1.0 / INFER_TARGET_FPS

# Konfigurasi penyimpanan frame deteksi (annotated) ke folder assets
SAVE_DETECTIONS = True
DETECTION_SAVE_DIR = os.path.join('assets')
MIN_SAVE_INTERVAL = 2.0  # detik antar simpan tipe sama per kamera

if SAVE_DETECTIONS:
    os.makedirs(DETECTION_SAVE_DIR, exist_ok=True)

# Cache hasil inference per kamera agar tidak selalu inference setiap frame stream
infer_cache = {
    cam_id: {
        'annotated_frame': None,
        'last_infer_time': 0.0,
        'info': None
    } for cam_id in CAMERA_IDS
}

# Waktu simpan terakhir per kamera per tipe ('fire','spark')
last_saved = {
    cam_id: {
        'fire': 0.0,
        'spark': 0.0
    } for cam_id in CAMERA_IDS
}

# Flag menandakan apakah untuk deteksi yang sedang aktif kita sudah menyimpan gambar
saved_for_active_detection = {
    cam_id: {
        'fire': False,
        'spark': False
    } for cam_id in CAMERA_IDS
}

# Menyimpan frame terakhir yang masih mengandung api untuk disimpan ketika deteksi berakhir
last_frame_with_fire = { cam_id: None for cam_id in CAMERA_IDS }

def capture_loop(cam_id: str):
    """Thread pembacaan kamera: selalu update latest_frames[cam_id]['frame']"""
    global camera_objects
    cam = camera_objects[cam_id]
    while True:
        ok, frame = cam.read()
        if not ok:
            latest_frames[cam_id]['error_count'] += 1
            # Coba reopen
            try:
                cam.release()
            except Exception:
                pass
            time.sleep(REOPEN_DELAY)
            camera_objects[cam_id] = open_camera(int(cam_id))
            cam = camera_objects[cam_id]
            continue
        with latest_frames[cam_id]['lock']:
            latest_frames[cam_id]['frame'] = frame
            latest_frames[cam_id]['last_read_time'] = time.time()
            latest_frames[cam_id]['error_count'] = 0

# Start capture threads
for cam_id in CAMERA_IDS:
    t = threading.Thread(target=capture_loop, args=(cam_id,), daemon=True)
    t.start()

def process_frame(frame):
    """Jalankan deteksi pada frame dan kembalikan info + anotasi."""
    with torch.no_grad():
        results = model(frame, verbose=False)
    fire_conf = 0.0
    spark_conf = 0.0
    fire_boxes = []
    spark_boxes = []
    detected_objects = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            if confidence < 0.3:
                continue
            detected_objects.append(class_name)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            lower = class_name.lower()
            if lower in ['api', 'fire']:
                if confidence > fire_conf:
                    fire_conf = confidence
                fire_boxes.append((x1, y1, x2, y2, confidence))
            elif lower in ['percikan', 'spark']:
                if confidence > spark_conf:
                    spark_conf = confidence
                spark_boxes.append((x1, y1, x2, y2, confidence))

    # Tentukan status
    if fire_conf >= 0.4:
        status = 'BAHAYA'
        status_message = 'BAHAYA - API TERDETEKSI'
        conf_used = fire_conf
    elif spark_conf >= 0.4:
        status = 'TERDETEKSI BAHAYA'
        status_message = 'TERDETEKSI BAHAYA - PERCIKAN TERDETEKSI'
        conf_used = spark_conf
    else:
        status = 'AMAN'
        status_message = 'AMAN'
        conf_used = 0.0

    # Gambar box
    for (x1, y1, x2, y2, conf) in fire_boxes:
        color = (0, 0, 255)
        label = f'fire {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for (x1, y1, x2, y2, conf) in spark_boxes:
        color = (0, 165, 255)
        label = f'spark {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Overlay status
    status_color = (0, 0, 255) if status == 'BAHAYA' else (0, 165, 255) if status == 'TERDETEKSI BAHAYA' else (0, 255, 0)
    display_text = status_message
    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
    cv2.rectangle(frame, (10, 10), (10 + text_size[0] + 20, 10 + text_size[1] + 20), (0, 0, 0), -1)
    cv2.putText(frame, display_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
    if conf_used > 0:
        cv2.putText(frame, f"{conf_used*100:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    return frame, {
        'status': status,
        'status_message': status_message,
        'confidence_percentage': conf_used * 100,
        'objects_detected': detected_objects,
        'fire_conf': fire_conf * 100,
        'spark_conf': spark_conf * 100
    }

def gen_frames(cam_id: str):
    if cam_id not in latest_frames:
        return
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    while True:
        now = time.time()
        # Ambil frame terbaru
        with latest_frames[cam_id]['lock']:
            base_frame = None if latest_frames[cam_id]['frame'] is None else latest_frames[cam_id]['frame'].copy()

        if base_frame is None:
            # Placeholder sederhana (hitam) agar stream tetap hidup
            placeholder = np.zeros((240, 320, 3), dtype='uint8')
            cv2.putText(placeholder, 'NO FRAME', (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buf = cv2.imencode('.jpg', placeholder, encode_param)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.05)
            continue

        cache = infer_cache[cam_id]
        need_infer = (now - cache['last_infer_time']) >= MIN_INFER_INTERVAL or cache['annotated_frame'] is None

        if need_infer:
            frame, info = process_frame(base_frame)
            cache['annotated_frame'] = frame
            cache['info'] = info
            cache['last_infer_time'] = now

            # Update state hanya saat ada inference baru
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            st = states[cam_id]
            st['last_detection'].update({
                'status': info['status'],
                'status_message': info['status_message'],
                'time': current_time,
                'confidence_percentage': info['confidence_percentage'],
                'objects_detected': info['objects_detected']
            })

            # Histories
            if info['fire_conf'] >= 40:
                st['fire_history'].append({
                    'time': current_time,
                    'confidence': info['fire_conf'],
                    'status': 'BAHAYA'
                })
                if len(st['fire_history']) > HISTORY_MAX_LEN:
                    st['fire_history'].pop(0)
            if info['spark_conf'] >= 40:
                st['spark_history'].append({
                    'time': current_time,
                    'confidence': info['spark_conf'],
                    'status': 'TERDETEKSI BAHAYA'
                })
                if len(st['spark_history']) > HISTORY_MAX_LEN:
                    st['spark_history'].pop(0)

            # Simpan frame jika ada deteksi (threshold sama dengan logika status: 40%)
            if SAVE_DETECTIONS:
                # Fire: simpan hanya sekali untuk periode deteksi yang berkelanjutan
                if info['fire_conf'] >= 40:
                    # update last seen frame that still contains fire
                    try:
                        last_frame_with_fire[cam_id] = frame.copy()
                    except Exception:
                        last_frame_with_fire[cam_id] = frame

                    # Simpan frame pertama kali api terdeteksi
                    if not saved_for_active_detection[cam_id]['fire']:
                        ts = time.strftime('%Y%m%d_%H%M%S')
                        fname = f"{cam_id}_fire_start_{int(info['fire_conf'])}_{ts}.jpg"
                        cv2.imwrite(os.path.join(DETECTION_SAVE_DIR, fname), last_frame_with_fire[cam_id])
                        last_saved[cam_id]['fire'] = now
                        saved_for_active_detection[cam_id]['fire'] = True
                else:
                    # Jika sebelumnya api aktif (telah disimpan start), saat deteksi hilang simpan frame terakhir yang masih berisi api
                    if saved_for_active_detection[cam_id]['fire']:
                        if last_frame_with_fire.get(cam_id) is not None:
                            ts = time.strftime('%Y%m%d_%H%M%S')
                            # gunakan confidence 0 karena saat terakhir terdeteksi kita menyimpan nama file dengan _end
                            fname = f"{cam_id}_fire_end_{ts}.jpg"
                            try:
                                cv2.imwrite(os.path.join(DETECTION_SAVE_DIR, fname), last_frame_with_fire[cam_id])
                            except Exception:
                                # fallback: simpan current frame if copying failed
                                cv2.imwrite(os.path.join(DETECTION_SAVE_DIR, fname), frame)
                            last_saved[cam_id]['fire'] = now
                        # reset state sehingga pada deteksi berikutnya akan menyimpan start lagi
                        saved_for_active_detection[cam_id]['fire'] = False
                        last_frame_with_fire[cam_id] = None

                # Spark: juga simpan hanya sekali per periode deteksi
                if info['spark_conf'] >= 40:
                    # Tetap gunakan MIN_SAVE_INTERVAL sebagai safety untuk mencegah spam bila diperlukan
                    if (not saved_for_active_detection[cam_id]['spark']) and ((now - last_saved[cam_id]['spark']) >= MIN_SAVE_INTERVAL):
                        ts = time.strftime('%Y%m%d_%H%M%S')
                        fname = f"{cam_id}_spark_{int(info['spark_conf'])}_{ts}.jpg"
                        cv2.imwrite(os.path.join(DETECTION_SAVE_DIR, fname), frame)
                        last_saved[cam_id]['spark'] = now
                        saved_for_active_detection[cam_id]['spark'] = True
                else:
                    saved_for_active_detection[cam_id]['spark'] = False
        else:
            frame = cache['annotated_frame']

        # Encode frame (annotated lama atau baru)
        _, buf = cv2.imencode('.jpg', frame, encode_param)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        time.sleep(STREAM_LOOP_SLEEP)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    if cam_id not in CAMERA_IDS:
        return "Camera not found", 404
    return Response(gen_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/notification/<cam_id>')
def notification(cam_id):
    if cam_id not in CAMERA_IDS:
        return jsonify({'error': 'Camera not found'}), 404
    return jsonify(states[cam_id]['last_detection'])

@app.route('/api/spark-history/<cam_id>')
def get_spark_history(cam_id):
    if cam_id not in CAMERA_IDS:
        return jsonify({'error': 'Camera not found'}), 404
    return jsonify(states[cam_id]['spark_history'])

@app.route('/api/fire-history/<cam_id>')
def get_fire_history(cam_id):
    if cam_id not in CAMERA_IDS:
        return jsonify({'error': 'Camera not found'}), 404
    return jsonify(states[cam_id]['fire_history'])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
