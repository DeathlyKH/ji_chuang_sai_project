#!/usr/bin/env python3
"""
Step 3: 表情识别模块
- 人脸检测（OpenCV Haar 级联，无需额外模型）
- 表情分类（支持模拟模式 / ONNX 模型模式）
- 实时测试
"""

import cv2
import numpy as np
import os
import time      
from collections import deque


# ========== 配置 ==========
EXPRESSION_LABELS = ["happy", "surprise", "neutral", "others"]
LABEL_TO_IDX = {name: i for i, name in enumerate(EXPRESSION_LABELS)}

# 颜色映射（用于画框）
EXPR_COLORS = {
    "happy": (0, 255, 0),      # 绿色
    "surprise": (0, 255, 255),  # 黄色
    "neutral": (128, 128, 128), # 灰色
    "others": (0, 0, 255),      # 红色
    "none": (200, 200, 200)     # 浅灰
}


class FaceDetector:
    """
    人脸检测器（OpenCV Haar 级联）
    优点：无需下载额外模型，OpenCV 内置
    """
    
    def __init__(self):
        # OpenCV 自带的正面人脸检测器
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        if self.detector.empty():
            raise RuntimeError("无法加载 Haar 级联文件")
        
        print("[FaceDetector] Haar 级联加载成功")
    
    def detect(self, frame):
        """
        检测人脸
        返回: [(x, y, w, h), ...] 可能多个
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测参数
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,   # 缩放步长
            minNeighbors=5,    # 最小邻近矩形数
            minSize=(80, 80),  # 最小人脸尺寸
            maxSize=(400, 400) # 最大人脸尺寸
        )
        
        return faces  # numpy array, shape [N, 4]


class ExpressionClassifier:
    """
    表情分类器
    模式 A：加载 ONNX 模型（真实推理）
    模式 B：模拟模式（无模型时随机演示）
    """
    
    def __init__(self, model_path=None):
        self.onnx_net = None
        self.current_expr = "neutral"
        self.confidence = 0.0
        self.history = deque(maxlen=5)  # 时序平滑
        
        # 尝试加载 ONNX 模型
        if model_path and os.path.exists(model_path):
            self.onnx_net = cv2.dnn.readNetFromONNX(model_path)
            print(f"[ExpressionClassifier] ONNX 模型加载成功: {model_path}")
            self.mode = "onnx"
        else:
            print(f"[ExpressionClassifier] ONNX 模型未找到: {model_path}")
            print("  进入模拟模式（界面正常，结果是随机的）")
            self.mode = "simulate"
    
    def preprocess(self, face_img):
        """
        预处理人脸图像为模型输入
        face_img: BGR 图像，任意尺寸
        返回: [1, 3, 224, 224] blob
        """
        # Resize 到 224x224
        img = cv2.resize(face_img, (224, 224))
        
        # BGR → RGB, 归一化 [0,1], 减均值除标准差 (ImageNet)
        blob = cv2.dnn.blobFromImage(
            img, 
            scalefactor=1.0 / 255.0,
            size=(224, 224),
            mean=(0.485, 0.456, 0.406),
            swapRB=True,  # BGR→RGB
            crop=False
        )
        # blobFromImage 已经做了 scale 和 mean/std
        # 但我们需要手动除以 std
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        blob = blob / std
        
        return blob
    
    def predict(self, face_img):
        """
        预测表情
        返回: (label, confidence)
        """
        if self.mode == "onnx" and self.onnx_net is not None:
            # 真实 ONNX 推理
            blob = self.preprocess(face_img)
            self.onnx_net.setInput(blob)
            output = self.onnx_net.forward()[0]  # [4]
            
            # Softmax
            exp_out = np.exp(output - np.max(output))
            probs = exp_out / np.sum(exp_out)
            
            idx = np.argmax(probs)
            conf = float(probs[idx])
            label = EXPRESSION_LABELS[idx]
        
        else:
            # 模拟模式：根据人脸颜色简单启发 + 随机
            # 计算脸部平均亮度
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # 简单启发：亮一点 → happy，正常 → neutral
            if brightness > 130:
                label = "happy"
                conf = 0.7 + np.random.random() * 0.2
            elif brightness < 100:
                label = "surprise"
                conf = 0.6 + np.random.random() * 0.2
            else:
                label = "neutral"
                conf = 0.5 + np.random.random() * 0.2
        
        # 时序平滑
        self.history.append((label, conf))
        if len(self.history) >= 3:
            # 取最近5帧中 majority vote
            from collections import Counter
            labels = [h[0] for h in self.history]
            most_common, count = Counter(labels).most_common(1)[0]
            if count >= len(self.history) * 0.6:
                self.current_expr = most_common
                # 置信度取平均
                confs = [h[1] for h in self.history if h[0] == most_common]
                self.confidence = float(np.mean(confs))
        
        return self.current_expr, self.confidence
    
    def draw(self, frame, face_box, expr_label, expr_conf):
        """
        在画面上绘制人脸框和表情标签
        """
        x, y, w, h = face_box
        color = EXPR_COLORS.get(expr_label, (200, 200, 200))
        
        # 画人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # 标签文字
        text = f"{expr_label}: {expr_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # 文字背景
        cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
        cv2.putText(frame, text, (x, y - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame


def main():
    """主函数：实时表情识别测试"""
    print("=" * 60)
    print("表情识别测试程序")
    print("=" * 60)
    print("按 'q' 退出 | 's' 截图")
    print("=" * 60)
    
    # 初始化
    face_det = FaceDetector()
    expr_cls = ExpressionClassifier(model_path="models/expression_mobilenetv3.onnx")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    fps = 30
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        frame_count += 1
        
        # FPS 计算
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - start_time + 0.001)
            start_time = time.time()
        
        # ===== 1. 人脸检测 =====
        faces = face_det.detect(frame)
        
        # ===== 2. 表情识别（取最大人脸）=====
        if len(faces) > 0:
            # 取最大的人脸
            areas = [fw * fh for (x, y, fw, fh) in faces]
            largest_idx = np.argmax(areas)
            face = faces[largest_idx]
            fx, fy, fw, fh = face
            
            # 扩大裁剪区域（包含更多上下文）
            margin = int(0.2 * fh)
            y1 = max(0, fy - margin)
            y2 = min(h, fy + fh + margin)
            x1 = max(0, fx - margin)
            x2 = min(w, fx + fw + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                expr, conf = expr_cls.predict(face_crop)
                frame = expr_cls.draw(frame, face, expr, conf)
        
        # ===== 3. 信息面板 =====
        info = [
            f"Mode: {expr_cls.mode}",
            f"Expression: {expr_cls.current_expr}",
            f"Confidence: {expr_cls.confidence:.2f}",
            f"Faces: {len(faces)}",
            f"FPS: {fps:.1f}",
            "[q]uit [s]creenshot"
        ]
        for i, text in enumerate(info):
            cv2.putText(frame, text, (10, 30 + i * 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 136), 2)
        
        cv2.imshow("Expression Recognition - Day 2", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"expr_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[截图] 已保存: {fname}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")


if __name__ == "__main__":
    main()