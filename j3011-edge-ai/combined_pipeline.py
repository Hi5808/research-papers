import cv2, time, threading, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from ultralytics import YOLO
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image

BIND = '0.0.0.0'
PORT = 8002
CAM = '/dev/video0'

print('loading YOLO TensorRT engine...', flush=True)
yolo = YOLO('/home/j30/yolov8n.engine')

print('loading Qwen2-VL-2B-Instruct...', flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)
vlm_model = AutoModelForImageTextToText.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    quantization_config=bnb_config,
    device_map='cuda:0',
    low_cpu_mem_usage=True,
)
vlm_processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
vlm_messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Briefly describe what you see in one sentence.'}]}]
vlm_text_template = vlm_processor.apply_chat_template(vlm_messages, tokenize=False, add_generation_prompt=True)
print('both models loaded', flush=True)

lock = threading.Lock()
latest_jpeg = None
latest_frame_rgb = None
detect_stats = {'fps': 0.0, 'infer_ms': 0.0, 'n_dets': 0}
caption_state = {'text': 'loading...', 'ts': '', 'infer_s': 0.0}

def detect_loop():
    global latest_jpeg, latest_frame_rgb, detect_stats
    cap = cv2.VideoCapture(CAM)
    t_prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        t0 = time.time()
        results = yolo(frame, verbose=False, conf=0.55)
        infer_ms = (time.time() - t0) * 1000

        annotated = results[0].plot(line_width=3, font_size=18, conf=False)
        ok2, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now

        with lock:
            if ok2:
                latest_jpeg = buf.tobytes()
            latest_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detect_stats = {'fps': round(fps, 1), 'infer_ms': round(infer_ms, 1), 'n_dets': len(results[0].boxes)}

def caption_loop():
    global caption_state
    while True:
        with lock:
            frame_rgb = latest_frame_rgb.copy() if latest_frame_rgb is not None else None
        if frame_rgb is None:
            time.sleep(0.5)
            continue

        img = Image.fromarray(frame_rgb)
        inputs = vlm_processor(text=[vlm_text_template], images=[img], return_tensors='pt').to('cuda')
        t0 = time.time()
        with torch.no_grad():
            out = vlm_model.generate(**inputs, max_new_tokens=50, do_sample=False)
        dt = time.time() - t0
        result = vlm_processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()

        with lock:
            caption_state = {'text': result, 'ts': time.strftime('%H:%M:%S'), 'infer_s': round(dt, 1)}
        print(f'[caption {time.strftime("%H:%M:%S")}] ({dt:.1f}s) {result}', flush=True)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
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
            with lock:
                body = json.dumps({**detect_stats, 'caption': caption_state}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            html = b'''<!DOCTYPE html><html><head><title>Live YOLO + VLM - Jetson Orin Nano</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center;padding:20px}
img{max-width:90vw;border:2px solid #444;border-radius:8px}
#detstats{color:#8f8;font-size:16px;margin-top:10px}
#caption{font-size:20px;max-width:700px;margin:15px auto;color:#fff}
#capmeta{color:#888;font-size:13px}</style></head>
<body><h2>Jetson Orin Nano - Combined YOLO + VLM Pipeline</h2>
<img src="/stream">
<div id="detstats">loading...</div>
<div id="caption">loading...</div>
<div id="capmeta"></div>
<script>
async function poll(){
  try{
    const r = await fetch('/stats?t='+Date.now());
    const d = await r.json();
    document.getElementById('detstats').textContent =
      `YOLO: ${d.fps} FPS | ${d.infer_ms}ms | ${d.n_dets} objects`;
    document.getElementById('caption').textContent = d.caption.text;
    document.getElementById('capmeta').textContent =
      `VLM caption @ ${d.caption.ts} (${d.caption.infer_s}s)`;
  }catch(e){}
}
setInterval(poll, 1000); poll();
</script></body></html>'''
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)

if __name__ == '__main__':
    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=caption_loop, daemon=True).start()
    print(f'serving on http://{BIND}:{PORT}/', flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
