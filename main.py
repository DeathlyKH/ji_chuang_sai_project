import cv2                                                                                                                                                                                          
import numpy as np
import time
import threading
import re
import random
import sys
import os
import platform

# ============== 平台检测 ==============
IS_ARM = platform.machine().startswith(('arm', 'aarch'))
IS_WINDOWS = platform.system() == 'Windows'

# 抑制OpenCV MSMF后端报错（仅Windows）
if IS_WINDOWS:
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# ============== 摄像头初始化（跨平台兼容）=============
def init_camera():
    """初始化摄像头 - 支持本地摄像头或网络摄像头（从电脑接收）"""
    print("[INFO] 正在初始化摄像头...")
    
    # ========== 模式选择 ==========
    # 开发板没有摄像头？改成从电脑网络接收！
    # 1. 先尝试本地摄像头
    # 2. 如果失败，自动切换到网络摄像头模式
    
    USE_NETWORK_CAMERA = True  # ← 改成 True 使用电脑摄像头
    PC_IP = "192.168.31.178"   # ← 改成你电脑的IP地址！
    PC_PORT = 8080
    
    if USE_NETWORK_CAMERA:
        print(f"[INFO] 网络摄像头模式：连接 {PC_IP}:{PC_PORT}")
        return init_network_camera(PC_IP, PC_PORT)
    
    # 本地摄像头模式（原来的逻辑）
    return init_local_camera()


def init_local_camera():
    """本地USB摄像头初始化"""
    # 方法1：最简单的方式
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, test = cap.read()
        if ret and test is not None and test.size > 0:
            print(f"[OK] 本地摄像头打开成功: {test.shape[1]}x{test.shape[0]}")
            if IS_ARM:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            return cap
        cap.release()
    
    # 方法2：Windows DSHOW
    if IS_WINDOWS:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None and test.size > 0:
                print(f"[OK] 摄像头(DSHOW)打开成功")
                return cap
            cap.release()
    
    # 方法3：Linux V4L2
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, test = cap.read()
        if ret and test is not None and test.size > 0:
            print(f"[OK] 摄像头(V4L2)打开成功")
            return cap
        cap.release()
    
    print("[ERR] 本地摄像头无法打开！")
    return create_dummy_camera()


def init_network_camera(pc_ip, pc_port):
    """网络摄像头 - 从电脑接收视频流"""
    import socket
    import struct
    
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(10)
        print(f"[INFO] 正在连接电脑 {pc_ip}:{pc_port}...")
        client.connect((pc_ip, pc_port))
        client.settimeout(None)
        print("[OK] 已连接到电脑摄像头！")
        
        class NetworkCamera:
            def __init__(self, sock):
                self.sock = sock
                self.buffer = b''
                self.payload_size = struct.calcsize('!I')  # 4字节长度头
            
            def read(self):
                try:
                    # 接收4字节长度头
                    while len(self.buffer) < self.payload_size:
                        data = self.sock.recv(4096)
                        if not data:
                            return False, None
                        self.buffer += data
                    
                    msg_size = struct.unpack('!I', self.buffer[:self.payload_size])[0]
                    self.buffer = self.buffer[self.payload_size:]
                    
                    # 接收图像数据
                    while len(self.buffer) < msg_size:
                        data = self.sock.recv(4096)
                        if not data:
                            return False, None
                        self.buffer += data
                    
                    frame_data = self.buffer[:msg_size]
                    self.buffer = self.buffer[msg_size:]
                    
                    # 解码JPEG
                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
                    return False, None
                    
                except Exception as e:
                    print(f"[WARN] 网络摄像头读取失败: {e}")
                    return False, None
            
            def release(self):
                try:
                    self.sock.close()
                except:
                    pass
        
        return NetworkCamera(client)
        
    except Exception as e:
        print(f"[ERR] 网络摄像头连接失败: {e}")
        print("[提示] 请确认：1)电脑已运行camera_server.py 2)IP地址正确 3)防火墙未拦截8080端口")
        return create_dummy_camera()


def create_dummy_camera():
    """虚拟摄像头（显示提示信息）"""
    class DummyCapture:
        def __init__(self):
            self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(self.frame, "Camera Not Found!", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(self.frame, "Check connection", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        def read(self):
            return True, self.frame.copy()
        def release(self):
            pass
    return DummyCapture()

# ============== 中文绘制 ==============
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] 未安装 Pillow，中文显示可能异常。运行: pip install Pillow")

def _get_font(size=18):
    if not PIL_AVAILABLE:
        return None
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                continue
    return ImageFont.load_default()

def _draw_chinese_on_frame(frame, lines_with_pos_color, font_size=17):
    """一次性绘制多行中文，避免逐行覆盖
    lines_with_pos_color: [(text, (x,y), (B,G,R)), ...]
    """
    if not PIL_AVAILABLE:
        for text, pos, color in lines_with_pos_color:
            safe = text.encode('ascii', 'ignore').decode() or "..."
            cv2.putText(frame, safe, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return
    font = _get_font(font_size)
    if font is None:
        return
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, pos, color in lines_with_pos_color:
        r, g, b = color[2], color[1], color[0]
        draw.text(pos, text, font=font, fill=(r, g, b))
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============== 尝试导入输入层模块 ==============
GESTURE_AVAILABLE = False
EXPRESSION_AVAILABLE = False
SPEECH_AVAILABLE = False
import_errors = {"gesture": "", "expression": "", "speech": ""}

try:
    from step2_test_gesture import PoseDetector, GestureClassifier
    GESTURE_AVAILABLE = True
except Exception as e:
    import_errors["gesture"] = str(e)
    print(f"[WARN] 手势模块导入失败: {e}")

try:
    from step3_expression import FaceDetector, ExpressionClassifier
    EXPRESSION_AVAILABLE = True
except Exception as e:
    import_errors["expression"] = str(e)
    print(f"[WARN] 表情模块导入失败: {e}")

SpeechModule = None
SPEECH_AVAILABLE = False
# 语音导入延迟到初始化线程中，避免numba/llvmlite阻塞主程序

# ============== 导入输出层 ==============
from output_engine import BearCharacter, create_bear_pair


# ============== 多模态融合引擎 ==============
class FusionEngine:
    """手势/表情/语音 → 熊大熊二动作映射"""

    # 语音命令 → (action, 目标熊)
    SPEECH_COMMAND_MAP = {
        "hello_bear": ("greeting", "熊大"),
        "take_photo": ("photo", "熊大"),
        "goodbye": ("goodbye", "熊二"),
    }

    def __init__(self, xiongda, xionger):
        self.xiongda = xiongda
        self.xionger = xionger
        self.last_trigger = 0
        self.cooldown = 2.5

    def _can_trigger(self):
        now = time.time()
        if now - self.last_trigger < self.cooldown:
            return False
        self.last_trigger = now
        return True

    def on_gesture(self, gesture_name):
        if not self._can_trigger():
            return
        print(f"[融合] 手势: {gesture_name}")
        if gesture_name == "wave":
            self.xiongda.trigger_response("wave")
        elif gesture_name == "cheer":
            self.xiongda.trigger_response("cheer")
        elif gesture_name == "heart":
            self.xionger.trigger_response("heart")

    def on_expression(self, expr_name):
        if not self._can_trigger():
            return
        print(f"[融合] 表情: {expr_name}")
        if expr_name == "happy":
            (self.xiongda if random.random() > 0.5 else self.xionger).trigger_response("happy")
        elif expr_name == "surprise":
            (self.xiongda if random.random() > 0.5 else self.xionger).trigger_response("surprised")

    def on_speech_command(self, cmd):
        """处理语音命令（step4返回的是命令字符串）"""
        if not cmd:
            return
        print(f"[融合] 语音命令: {cmd}")
        mapping = self.SPEECH_COMMAND_MAP.get(cmd)
        if mapping:
            action, target = mapping
            bear = self.xiongda if target == "熊大" else self.xionger
            bear.trigger_response(action)
            self.last_trigger = time.time()

    def on_keyboard(self, key):
        key_map = {
            '1': ('greeting', self.xiongda),
            '2': ('photo', self.xiongda),
            '3': ('goodbye', self.xionger),
            '4': ('wave', self.xiongda),
            '5': ('cheer', self.xiongda),
            '6': ('heart', self.xionger),
            '7': ('happy', self.xionger),
            '8': ('surprised', self.xiongda),
        }
        if key in key_map:
            action, bear = key_map[key]
            print(f"[键盘] 模拟触发: {action}")
            bear.trigger_response(action)


# ============== 背景绘制 ==============
def draw_forest_background(frame):
    h, w = frame.shape[:2]
    for y in range(h // 2):
        ratio = y / (h // 2)
        color = (int(140 + 30 * ratio), int(200 + 10 * ratio), int(230 - 20 * ratio))
        cv2.line(frame, (0, y), (w, y), color, 1)
    for y in range(h // 2, h):
        ratio = (y - h // 2) / (h // 2)
        color = (int(45 + 25 * ratio), int(110 + 35 * ratio), int(45 + 15 * ratio))
        cv2.line(frame, (0, y), (w, y), color, 1)
    cv2.circle(frame, (w - 100, 80), 40, (60, 220, 255), -1)
    cv2.circle(frame, (w - 100, 80), 40, (80, 200, 240), 2)
    for cx, cy in [(150, 80), (400, 120), (w - 250, 100)]:
        cv2.ellipse(frame, (cx, cy), (40, 25), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (cx - 25, cy + 10), (25, 20), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (cx + 25, cy + 10), (25, 20), 0, 0, 360, (255, 255, 255), -1)
    for tx in [80, w - 80, w // 2]:
        tree_pts = np.array([[tx, h // 2 + 20], [tx - 30, h - 50], [tx + 30, h - 50]], np.int32)
        cv2.fillPoly(frame, [tree_pts], (35, 95, 35))


# ============== 主程序 ==============
WINDOW_NAME = "Bear Interactive"


def main():
    WINDOW_W, WINDOW_H = 1280, 720
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    # ========== 摄像头初始化（跨平台兼容）==========
    cap = init_camera()

    # ========== 初始化输入模块 ==========

    # ========== 初始化输入模块 ==========
    pose_detector = None
    gesture_classifier = None
    gesture_init_err = ""
    if GESTURE_AVAILABLE:
        try:
            pose_detector = PoseDetector('yolov8n-pose.onnx')
            gesture_classifier = GestureClassifier(history_size=15)
            print("[OK] 手势模块已加载")
        except Exception as e:
            gesture_init_err = str(e)
            print(f"[ERR] 手势初始化失败: {e}")
    else:
        gesture_init_err = import_errors.get("gesture", "未找到")[:40]
        print(f"[WARN] 手势不可用: {gesture_init_err}")

    face_detector = None
    expression_classifier = None
    expr_init_err = ""
    if EXPRESSION_AVAILABLE:
        try:
            face_detector = FaceDetector()
            # 尝试加载ONNX模型，找不到则进入模拟模式
            expr_model_path = "models/expression_mobilenetv3.onnx"
            if not os.path.exists(expr_model_path):
                # 尝试其他可能路径
                for alt in ["expression_mobilenetv3.onnx", "expression_model.onnx"]:
                    if os.path.exists(alt):
                        expr_model_path = alt
                        break
            expression_classifier = ExpressionClassifier(model_path=expr_model_path if os.path.exists(expr_model_path) else None)
            print("[OK] 表情模块已加载")
        except Exception as e:
            expr_init_err = str(e)
            print(f"[ERR] 表情初始化失败: {e}")
    else:
        expr_init_err = import_errors.get("expression", "未找到")[:40]
        print(f"[WARN] 表情不可用: {expr_init_err}")

    speech_module = None
    speech_init_err = ""
    speech_result = [None]
    
    def _init_speech():
        """后台线程：导入+初始化语音（避免numba/llvmlite阻塞主程序）"""
        try:
            # 先检测是否有真实麦克风
            import pyaudio
            pa = pyaudio.PyAudio()
            has_mic = False
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0 and 'mic' in info['name'].lower() or 'input' in info['name'].lower():
                    has_mic = True
                    break
            pa.terminate()
            
            if not has_mic:
                speech_result[0] = "ERROR:未检测到麦克风"
                print("[WARN] 未检测到麦克风，语音模块已禁用")
                return
                
            from step4_speech import SpeechModule as SM
            sm = SM(model_size="tiny")
            sm.start()
            speech_result[0] = sm
            print("[OK] 语音模块已加载并启动")
        except Exception as e:
            speech_result[0] = f"ERROR:{e}"
            print(f"[ERR] 语音初始化失败: {e}")
    
    speech_thread = threading.Thread(target=_init_speech)
    speech_thread.daemon = True
    speech_thread.start()
    speech_thread.join(timeout=8)  # 给numba初始化留8秒
    
    if isinstance(speech_result[0], str) and speech_result[0].startswith("ERROR:"):
        speech_init_err = speech_result[0][6:]
        print(f"[ERR] 语音初始化失败: {speech_init_err}")
    elif speech_result[0] is not None:
        speech_module = speech_result[0]
    else:
        speech_init_err = "语音后台加载中，手势/表情已就绪"
        print(f"[WARN] {speech_init_err}")

    # ========== 输出层 ==========
    xiongda, xionger = create_bear_pair(stage_w=640, stage_h=720)
    fusion = FusionEngine(xiongda, xionger)

    status_data = {
        "gesture": "手势: 模块未加载" if not GESTURE_AVAILABLE else "手势: 等待检测...",
        "expression": "表情: 模块未加载" if not EXPRESSION_AVAILABLE else "表情: 等待检测...",
        "speech": "语音: 模块未加载" if not SPEECH_AVAILABLE else "语音: 等待输入...",
        "fps": "FPS: --",
    }

    frame_count = 0
    t0 = time.time()
    running = True
    grab_fail_count = 0
    gesture_err_count = 0
    expr_err_count = 0
    speech_err_count = 0
    input_frame_counter = 0  # 跳帧计数器
    last_expr_result = "等待检测..."  # 保留上次表情结果

    while running:
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            break

        ret, cam_frame = cap.read()
        if not ret:
            grab_fail_count += 1
            if grab_fail_count > 10:
                # 尝试重新打开摄像头（自动重连）
                cap.release()
                time.sleep(0.2)
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret2, cam_frame = cap.read()
                if ret2:
                    grab_fail_count = 0
                    print("[OK] 摄像头已恢复")
                else:
                    cam_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(cam_frame, "Camera Offline - Retrying", (120, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        else:
            grab_fail_count = max(0, grab_fail_count - 1)

        # 跳帧：ARM每5帧做一次推理，PC每3帧
        input_frame_counter += 1
        skip_frames = 5 if IS_ARM else 3
        do_inference = (input_frame_counter % skip_frames == 0)

        # ========== 手势处理（每3帧）==========
        if do_inference and pose_detector and gesture_classifier and ret:
            try:
                blob, scale, pad_offset = pose_detector.preprocess(cam_frame)
                pose_detector.net.setInput(blob)
                outputs = pose_detector.net.forward()
                persons = pose_detector.postprocess(outputs, cam_frame.shape[1], cam_frame.shape[0], scale, pad_offset)

                if persons:
                    target_person = max(persons, key=lambda x: x['conf'])
                    gesture, g_conf = gesture_classifier.classify(target_person['keypoints'])
                    if gesture and gesture != "none":
                        status_data["gesture"] = f"手势: {gesture} ({g_conf:.2f})"
                        fusion.on_gesture(gesture)
                    else:
                        status_data["gesture"] = f"手势: 检测中 ({len(persons)}人)"
                else:
                    status_data["gesture"] = "手势: 未检测到人"
                gesture_err_count = 0
            except Exception as e:
                gesture_err_count += 1
                err_short = str(e)[:35]
                status_data["gesture"] = f"手势错误#{gesture_err_count}: {err_short}"
                if gesture_err_count <= 3:
                    import traceback
                    traceback.print_exc()
        elif gesture_init_err:
            status_data["gesture"] = f"手势: 初始化失败"
        elif not GESTURE_AVAILABLE:
            status_data["gesture"] = "手势: 模块未导入"

    # ========== 表情处理（每3帧，增强+远距检测）==========
        faces_for_drawing = []  # 用于在画面上画框
        if do_inference and face_detector and expression_classifier and ret:
            try:
                faces = face_detector.detect(cam_frame)
                # 放宽参数重试1
                if len(faces) == 0:
                    try:
                        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
                        faces = face_detector.detector.detectMultiScale(
                            gray, scaleFactor=1.15, minNeighbors=4,
                            minSize=(40, 40), maxSize=(500, 500)
                        )
                    except Exception:
                        pass
                # 放宽参数重试2
                if len(faces) == 0:
                    try:
                        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
                        faces = face_detector.detector.detectMultiScale(
                            gray, scaleFactor=1.2, minNeighbors=3,
                            minSize=(30, 30), maxSize=(600, 600)
                        )
                    except Exception:
                        pass
                # 再试一次：直方图均衡化+最宽松
                if len(faces) == 0:
                    try:
                        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        faces = face_detector.detector.detectMultiScale(
                            gray, scaleFactor=1.2, minNeighbors=2,
                            minSize=(30, 30), maxSize=(600, 600)
                        )
                    except Exception:
                        pass

                faces_for_drawing = faces
                if len(faces) > 0:
                    areas = [fw * fh for (x, y, fw, fh) in faces]
                    largest_idx = np.argmax(areas)
                    fx, fy, fw, fh = faces[largest_idx]

                    h, w = cam_frame.shape[:2]
                    margin = int(0.2 * fh)
                    y1 = max(0, fy - margin)
                    y2 = min(h, fy + fh + margin)
                    x1 = max(0, fx - margin)
                    x2 = min(w, fx + fw + margin)
                    face_crop = cam_frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        expr, conf = expression_classifier.predict(face_crop)
                        status_data["expression"] = f"表情: {expr} ({conf:.2f})"
                        last_expr_result = f"{expr} ({conf:.2f})"
                        fusion.on_expression(expr)
                    else:
                        status_data["expression"] = f"表情: {last_expr_result} | 裁剪失败"
                else:
                    status_data["expression"] = f"表情: {last_expr_result} | 未检测到人"
                expr_err_count = 0
            except Exception as e:
                expr_err_count += 1
                err_short = str(e)[:35]
                status_data["expression"] = f"表情错误#{expr_err_count}: {err_short}"
                if expr_err_count <= 3:
                    import traceback
                    traceback.print_exc()
        elif expr_init_err:
            status_data["expression"] = f"表情: 初始化失败"
        elif not EXPRESSION_AVAILABLE:
            status_data["expression"] = "表情: 模块未导入"

        # ========== 语音处理（每帧检查，get_command很轻量）==========
        # 超时线程后来加载成功了吗？如果是，启用它
        try:
            if speech_module is None and speech_result is not None and len(speech_result) > 0:
                sr = speech_result[0]
                if sr is not None and not isinstance(sr, str):
                    speech_module = sr
                    speech_init_err = ""  # 清除错误
                    print("[OK] 语音模块后台加载完成，已启用")
        except Exception:
            pass
        
        if speech_module:
            try:
                cmd = speech_module.get_command()
                # 关键词白名单：只有这些指令才响应，其他全部丢弃（防止乱识别）
                if cmd:
                    keyword_map = {"hello_bear": ("你好",), "take_photo": ("拍照", "茄子"), "goodbye": ("再见", "拜拜")}
                    matched = False
                    for mapped_cmd, keywords in keyword_map.items():
                        if any(k in cmd for k in keywords):
                            status_data["speech"] = f"语音: {keywords[0]}"
                            fusion.on_speech_command(mapped_cmd)
                            matched = True
                            break
                    if not matched and "以下是普通话" not in cmd and len(cmd) < 10:
                        # 短文本但不是关键词，显示过滤
                        status_data["speech"] = f"语音: (忽略:'{cmd}')"
                    elif not matched:
                        status_data["speech"] = "语音: (过滤噪声)"
                elif do_inference:
                    status_data["speech"] = "语音: 对着麦克风说 你好/茄子/拜拜"
                speech_err_count = 0
            except Exception as e:
                speech_err_count += 1
                err_short = str(e)[:35]
                status_data["speech"] = f"语音错误#{speech_err_count}: {err_short}"
                if speech_err_count <= 3:
                    import traceback
                    traceback.print_exc()
        elif speech_init_err:
            status_data["speech"] = f"语音: {speech_init_err}"
        else:
            status_data["speech"] = "语音: 模块未导入"

        # ========== 渲染 ==========
        canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

        # 左侧面板
        left_panel = canvas[:, :640].copy()
        if ret:
            # 在摄像头画面上画人脸检测框（让用户知道什么时候检测到了）
            if len(faces_for_drawing) > 0:
                for (fx, fy, fw, fh) in faces_for_drawing:
                    cv2.rectangle(cam_frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                    cv2.putText(cam_frame, "face", (fx, fy-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            left_panel[:480, :] = cv2.resize(cam_frame, (640, 480))
        else:
            left_panel[:480, :] = (30, 30, 30)

        left_panel[480:, :] = (25, 25, 25)

        status_lines = [
            ("=== 输入层状态 ===", (20, 505), (0, 200, 255)),
            (status_data["gesture"], (20, 545), (200, 200, 200)),
            (status_data["expression"], (20, 585), (200, 200, 200)),
            (status_data["speech"], (20, 625), (200, 200, 200)),
            (status_data["fps"] + " | 按Q/ESC退出 | 1-8测试", (20, 685), (120, 220, 120)),
        ]
        _draw_chinese_on_frame(left_panel, status_lines)
        canvas[:, :640] = left_panel

        # 右侧舞台
        right_panel = canvas[:, 640:].copy()
        draw_forest_background(right_panel)
        xiongda.draw(right_panel)
        xionger.draw(right_panel)
        canvas[:, 640:] = right_panel

        # FPS
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            t0 = time.time()
            status_data["fps"] = f"FPS: {fps:.1f}"

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            running = False
            break
        elif 49 <= key <= 56:
            fusion.on_keyboard(chr(key))

    # 清理
    running = False
    cap.release()
    if speech_module:
        try:
            speech_module.stop()
        except:
            pass
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)
    print("[OK] 系统已关闭")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[OK] 用户中断")
        cv2.destroyAllWindows()
        sys.exit(0)
