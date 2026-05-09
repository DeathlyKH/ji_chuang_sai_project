import cv2
import numpy as np
import threading
import time
import random
import math
import os
import platform

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow not installed")

# ============== 字体 ==============
_FONTS = {}

def _get_font(size=18):
    if not PIL_AVAILABLE:
        return None
    key = f"s{size}"
    if key in _FONTS:
        return _FONTS[key]
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                _FONTS[key] = ImageFont.truetype(p, size)
                return _FONTS[key]
            except:
                pass
    _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]

# ============== TTS ==============
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class TTSEngine:
    def __init__(self):
        self.engine = None
        self.speaking = False
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 190)
                self.engine.setProperty('volume', 0.9)
                for v in self.engine.getProperty('voices'):
                    if 'chinese' in v.name.lower() or 'zh' in v.id.lower():
                        self.engine.setProperty('voice', v.id)
                        break
            except:
                self.engine = None

    def speak(self, text):
        if not TTS_AVAILABLE or self.engine is None:
            return
        def _s():
            self.speaking = True
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass
            self.speaking = False
        threading.Thread(target=_s, daemon=True).start()


# ============== 粒子 ==============
class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, cx, cy, count=20, color=(255, 200, 0), life=1.0):
        for _ in range(count):
            a = random.uniform(0, 2 * math.pi)
            s = random.uniform(2, 8)
            self.particles.append({
                'x': cx, 'y': cy, 'vx': math.cos(a)*s, 'vy': math.sin(a)*s - 3,
                'life': life, 'max_life': life, 'size': random.randint(3,8),
                'color': color, 'gravity': 0.2
            })

    def update_and_draw(self, frame):
        new_p = []
        for p in self.particles:
            p['x'] += p['vx']; p['y'] += p['vy']; p['vy'] += p['gravity']
            p['life'] -= 0.025
            if p['life'] > 0:
                a = p['life'] / p['max_life']
                c = tuple(int(v*a) for v in p['color'])
                cv2.circle(frame, (int(p['x']),int(p['y'])), max(1,int(p['size']*a)), c, -1)
                new_p.append(p)
        self.particles = new_p


# ============== 气泡 ==============
class SpeechBubble:
    def __init__(self):
        self.text = ""; self.t0 = 0; self.dur = 3.0; self.alpha = 0.0

    def set_text(self, text, duration=3.5):
        self.text = text; self.t0 = time.time(); self.dur = duration; self.alpha = 1.0

    def draw(self, frame, cx, cy, stage_w):
        elapsed = time.time() - self.t0
        if elapsed > self.dur:
            self.alpha = max(0, self.alpha - 0.06)
            if self.alpha <= 0: return
        if not self.text: return
        lines = [self.text[i:i+8] for i in range(0, len(self.text), 8)]
        line_h, pad = 26, 12
        bw = 190
        bh = max(45, len(lines)*line_h + pad*2)
        bx = max(8, min(cx - bw//2, stage_w - bw - 8))
        by = max(8, cy - bh - 35)
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx,by), (bx+bw,by+bh), (255,255,255), -1)
        cv2.rectangle(overlay, (bx,by), (bx+bw,by+bh), (160,160,160), 2)
        tri = np.array([[cx,by+bh],[cx-7,by+bh+9],[cx+7,by+bh+9]], np.int32)
        cv2.fillPoly(overlay, [tri], (255,255,255))
        cv2.fillPoly(overlay, [tri], (160,160,160))
        cv2.addWeighted(overlay, self.alpha*0.92, frame, 1-self.alpha*0.92, 0, frame)
        if PIL_AVAILABLE:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = _get_font(17)
            for i, line in enumerate(lines):
                y = by + pad + 18 + i*line_h
                draw.text((bx+pad, y-16), line, font=font, fill=(50,50,50))
            frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============== 精灵图加载（固定单图）=============
def load_sprite(path):
    """加载图片并自动去除纯色背景，返回RGBA数组"""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    # 采样四角颜色判断背景
    corners = np.array([img[0,0], img[0,w-1], img[h-1,0], img[h-1,w-1]], dtype=np.float32)
    bg_color = np.median(corners, axis=0)
    # 创建mask：差异>40的保留
    diff = np.abs(img.astype(np.float32) - bg_color)
    mask = (np.sum(diff, axis=2) > 40).astype(np.uint8) * 255
    # 形态学平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5,5), 1)
    # BGR→BGRA
    b, g, r = cv2.split(img)
    return cv2.merge([b, g, r, mask])


def overlay_rgba(bg, rgba, x, y):
    """将RGBA图像叠加到BGR背景上，带边缘羽化"""
    if rgba is None or bg is None:
        return
    h, w = rgba.shape[:2]
    H, W = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x+w), min(H, y+h)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2-x1), sy1 + (y2-y1)
    crop = rgba[sy1:sy2, sx1:sx2]
    alpha_raw = crop[:,:,3].astype(np.float32) / 255.0
    if alpha_raw.shape[0] > 5 and alpha_raw.shape[1] > 5:
        alpha_smooth = cv2.GaussianBlur(alpha_raw, (5, 5), 1.0)
    else:
        alpha_smooth = alpha_raw
    alpha = alpha_smooth[:,:,None]
    fg = crop[:,:,:3].astype(np.float32)
    bg_roi = bg[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * fg + (1.0 - alpha) * bg_roi
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)


# ============== 熊大熊二角色（固定单图版）=============
class BearCharacter:
    STATE_IDLE = "idle"
    STATE_GREETING = "greeting"
    STATE_HAPPY = "happy"
    STATE_SURPRISED = "surprised"
    STATE_PHOTO = "photo"
    STATE_SAD = "sad"
    STATE_WAVE = "wave"
    STATE_HEART = "heart"

    def __init__(self, name="熊大", x=320, y=360, size=180):
        self.name = name
        self.x = x
        self.y = y
        self.base_size = size
        self.state = self.STATE_IDLE
        self.state_time = time.time()
        self.partner = None

        # 只加载一张固定精灵图
        self.sprite = None
        self._load_sprite()

        # 子系统
        self.tts = TTSEngine()
        self.bubble = SpeechBubble()
        self.particles = ParticleSystem()

        # 动画参数
        self.breath_phase = random.random() * math.pi * 2
        self.flip = False
        self.jump_y = 0
        self.jump_vy = 0

        self._setup_responses()

    def _find_img_dir(self):
        for p in [os.path.join(os.path.dirname(__file__),"images"),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),"images"),
                  "images",
                  os.path.join(os.getcwd(),"images")]:
            if os.path.isdir(p):
                return p
        return None

    def _load_sprite(self):
        """只加载一张固定的 idle 图片"""
        img_dir = self._find_img_dir()
        if not img_dir:
            print(f"[WARN] {self.name}: 找不到images目录")
            return
        prefix = "xiongda" if self.name == "熊大" else "xionger"
        # 只加载 idle.png，所有状态都用这张图，通过动画区分
        path = os.path.join(img_dir, f"{prefix}_idle.png")
        self.sprite = load_sprite(path)
        if self.sprite is not None:
            print(f"[OK] {self.name} 加载精灵图: {self.sprite.shape[1]}x{self.sprite.shape[0]}")
        else:
            print(f"[WARN] {self.name}: 未找到 {path}")

    def _setup_responses(self):
        self.responses = {
            "greeting": {"text":["你好呀！我是熊大！","哈喽！很高兴见到你！","欢迎欢迎！"],
                        "tts":"你好呀，欢迎来到森林！"},
            "wave": {"text":["在挥手吗？我也跟你挥手！","嗨！嗨！看到你了！"],
                    "tts":"嗨！我看到你在挥手！"},
            "cheer": {"text":["耶！太棒了！","好厉害！为你欢呼！","棒棒哒！"],
                     "tts":"太棒了！为你欢呼！","particle":(0,200,255),"partner_react":"happy"},
            "heart": {"text":["比心！爱你哟！","收到你的爱心！","我也爱你！"],
                     "tts":"比心，我也爱你哟！","particle":(150,100,255),"partner_react":"happy"},
            "happy": {"text":["看起来很开心呢！","笑容真好看！","开心最重要！"],
                     "tts":"你看起来很开心，我也高兴！"},
            "surprised": {"text":["哇！吓我一跳！","好惊讶！","发生了什么？"],
                         "tts":"哇，吓我一跳！"},
            "photo": {"text":["茄子！咔嚓！","拍照啦！笑一个！","记录美好时刻！"],
                     "tts":"茄子！拍照啦！","particle":(200,255,255)},
            "goodbye": {"text":["再见！下次见！","拜拜！一路顺风！","我会想你的！"],
                       "tts":"再见，下次再来玩！"},
            "sad": {"text":["别难过，有我在呢。","开心一点嘛~","抱抱你！"],
                   "tts":"别难过，要开心哦。"}
        }

    def trigger_response(self, action, extra_text=None):
        now = time.time()
        if now - self.state_time < 2.0 and action != "greeting":
            return
        self.state_time = now
        cfg = self.responses.get(action, self.responses["greeting"])
        sm = {"greeting":self.STATE_GREETING,"wave":self.STATE_WAVE,
              "cheer":self.STATE_HAPPY,"heart":self.STATE_HEART,
              "happy":self.STATE_HAPPY,"surprised":self.STATE_SURPRISED,
              "photo":self.STATE_PHOTO,"goodbye":self.STATE_SAD,"sad":self.STATE_SAD}
        self.state = sm.get(action, self.STATE_IDLE)

        if "particle" in cfg:
            self.particles.emit(self.x, self.y-self.base_size//2, 25, cfg["particle"])
        text = extra_text or random.choice(cfg["text"])
        self.bubble.set_text(text, 3.5)
        if not self.tts.speaking:
            self.tts.speak(cfg.get("tts", text))
        if self.partner and "partner_react" in cfg:
            self.partner._react_to_partner(cfg["partner_react"])

    def _react_to_partner(self, react):
        self.state_time = time.time()
        if react == "happy":
            self.state = self.STATE_HAPPY
            self.jump_vy = -8

    def update(self):
        dt = 0.033
        self.breath_phase += dt * 2.5
        # 跳跃
        if self.jump_vy != 0 or self.jump_y != 0:
            self.jump_y += self.jump_vy
            self.jump_vy += 0.6
            if self.jump_y >= 0:
                self.jump_y = 0; self.jump_vy = 0
        # 转头
        if self.partner:
            dx = self.partner.x - self.x
            if self.state in [self.STATE_HAPPY, self.STATE_SURPRISED, self.STATE_GREETING]:
                self.flip = dx < 0
            else:
                self.flip = (abs(dx) < 150) and (dx < 0)
        # 自动恢复idle
        if self.state != self.STATE_IDLE and time.time() - self.state_time > 4.5:
            self.state = self.STATE_IDLE

    def draw(self, frame):
        self.update()
        h, w = frame.shape[:2]
        if self.sprite is None:
            cv2.putText(frame, f"[{self.name} no pic]", (self.x-50, self.y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
            return

        # 呼吸缩放
        breath = 1.0 + 0.02 * math.sin(self.breath_phase)
        # 状态额外缩放/跳跃
        extra = 1.0
        if self.state in [self.STATE_HAPPY, self.STATE_GREETING, self.STATE_HEART]:
            extra = 1.06 + 0.03 * math.sin(self.breath_phase * 3)
            jump = -abs(15 * math.sin(self.breath_phase * 5))
        elif self.state == self.STATE_SURPRISED:
            extra = 1.03
            jump = 0
        elif self.state == self.STATE_SAD:
            extra = 0.97
            jump = 0
        else:
            jump = self.jump_y

        total_scale = breath * extra
        sp_h, sp_w = self.sprite.shape[:2]
        new_w = int(self.base_size * total_scale)
        new_h = int(new_w * sp_h / sp_w)
        resized = cv2.resize(self.sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 翻转
        if self.flip:
            resized = cv2.flip(resized, 1)

        draw_y = self.y - new_h // 2 + int(jump)
        draw_x = self.x - new_w // 2

        # 阴影（草地色）
        shadow_y = self.y + self.base_size // 2 + 10
        overlay = frame.copy()
        cv2.ellipse(overlay, (self.x, shadow_y), (new_w//2+5, 12), 0, 0, 360, (35, 70, 35), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # 叠加
        overlay_rgba(frame, resized, draw_x, draw_y)

        # 名字
        self._draw_name(frame)
        # 气泡
        self.bubble.draw(frame, self.x, draw_y, w)
        # 粒子
        self.particles.update_and_draw(frame)

    def _draw_name(self, frame):
        if not PIL_AVAILABLE: return
        font = _get_font(16)
        if font is None: return
        label = self.name
        try:
            tw, th = font.getbbox(label)[2:4]
        except:
            tw, th = 40, 18
        lx = self.x - tw // 2
        ly = self.y + self.base_size // 2 + 25
        color = (50, 120, 40) if self.name == "熊大" else (60, 140, 50)
        overlay = frame.copy()
        cv2.rectangle(overlay, (lx-8, ly-3), (lx+tw+8, ly+th+5), color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((lx, ly-1), label, font=font, fill=(255,255,255))
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def create_bear_pair(stage_w=640, stage_h=720):
    xiongda = BearCharacter("熊大", x=int(stage_w*0.28), y=stage_h//2+80, size=180)
    xionger = BearCharacter("熊二", x=int(stage_w*0.72), y=stage_h//2+80, size=165)
    xiongda.partner = xionger
    xionger.partner = xiongda
    return xiongda, xionger
