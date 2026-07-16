import cv2, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ultralytics import YOLO

MODEL_PATH = '/home/j30/yolov8n.engine'
CAM = '/dev/video0'
BIND = '0.0.0.0'
PORT = 8001

model = YOLO(MODEL_PATH)

lock = threading.Lock()
latest_jpeg = None
latest_stats = {'fps': 0.0, 'infer_ms': 0.0, 'n_dets': 0}

def capture_loop():
    global latest_jpeg, latest_stats
    cap = cv2.VideoCapture(CAM)
    t_prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        t0 = time.time()
        results = model(frame, verbose=False, conf=0.55)
        infer_ms = (time.time() - t0) * 1000

        annotated = results[0].plot(line_width=3, font_size=18, conf=False)
        ok2, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok2:
            continue

        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now

        with lock:
            latest_jpeg = buf.tobytes()
            latest_stats = {
                'fps': round(fps, 1),
                'infer_ms': round(infer_ms, 1),
                'n_dets': len(results[0].boxes),
            }

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpg = latest_jpeg
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == '/stats':
            import json
            with lock:
                body = json.dumps(latest_stats).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            html = b'''<!DOCTYPE html><html><head><title>Live YOLO - Jetson Orin Nano</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center;padding:20px}
img{max-width:90vw;border:2px solid #444;border-radius:8px}
#stats{color:#8f8;font-size:16px;margin-top:10px}</style></head>
<body><h2>Jetson Orin Nano - Live YOLOv8n (TensorRT FP16)</h2>
<img src="/stream">
<div id="stats">loading...</div>
<script>
async function poll(){
  try{
    const r = await fetch('/stats?t='+Date.now());
    const d = await r.json();
    document.getElementById('stats').textContent =
      `${d.fps} FPS | inference ${d.infer_ms}ms | ${d.n_dets} objects detected`;
  }catch(e){}
}
setInterval(poll, 500); poll();
</script></body></html>'''
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)

if __name__ == '__main__':
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print(f'serving on http://{BIND}:{PORT}/', flush=True)
    server = ThreadingHTTPServer((BIND, PORT), Handler)
    server.serve_forever()
