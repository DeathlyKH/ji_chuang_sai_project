#!/usr/bin/env python3
"""
Step 2: 手势识别测试（Windows 本地，OpenCV DNN）
- YOLOv8-Pose 检测人体 + 17 个关键点
- 手势分类器：wave(挥手) / cheer(欢呼) / heart(比心)
"""

import cv2
import numpy as np
from collections import deque
import time

# ========== 配置参数 ==========
CONF_THRESHOLD = 0.5      # 人体检测置信度阈值
IOU_THRESHOLD = 0.45      # NMS 交并比阈值
MODEL_INPUT_SIZE = 640    # 模型输入尺寸

# COCO 17 关键点索引（固定标准）
NOSE = 0
L_EYE = 1; R_EYE = 2
L_EAR = 3; R_EAR = 4
L_SHOULDER = 5; R_SHOULDER = 6
L_ELBOW = 7; R_ELBOW = 8
L_WRIST = 9; R_WRIST = 10
L_HIP = 11; R_HIP = 12
L_KNEE = 13; R_KNEE = 14
L_ANKLE = 15; R_ANKLE = 16

# 骨架连接对（用于画线）
SKELETON = [
    (5, 7), (7, 9),       # 左臂：肩→肘→腕
    (6, 8), (8, 10),      # 右臂：肩→肘→腕
    (5, 6),               # 双肩
    (5, 11), (6, 12),     # 躯干
    (11, 12),             # 双髋
    (11, 13), (13, 15),   # 左腿
    (12, 14), (14, 16),   # 右腿
]


class PoseDetector:
    """
    YOLOv8-Pose 检测器（OpenCV DNN 后端）
    """
    
    def __init__(self, model_path='yolov8n-pose.onnx'):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[PoseDetector] 模型加载成功: {model_path}")
        
    def preprocess(self, frame):
        """
        LetterBox 预处理：保持长宽比缩放 + 灰色填充
        返回: (blob, scale, pad_offset)
        """
        h, w = frame.shape[:2]
        
        # 计算缩放比例
        scale = min(MODEL_INPUT_SIZE / w, MODEL_INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 计算 padding（灰色填充 114）
        pad_top = (MODEL_INPUT_SIZE - new_h) // 2
        pad_bottom = MODEL_INPUT_SIZE - new_h - pad_top
        pad_left = (MODEL_INPUT_SIZE - new_w) // 2
        pad_right = MODEL_INPUT_SIZE - new_w - pad_left
        
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # BGR → RGB, 归一化 [0,1], HWC → CHW
        padded_rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.transpose(padded_rgb, (2, 0, 1))
        blob = np.expand_dims(chw, axis=0)  # [1, 3, 640, 640]
        
        return blob, scale, (pad_left, pad_top)
    
    def postprocess(self, output, img_w, img_h, scale, pad_offset):
        """
        后处理：解码检测框 + NMS + 解析关键点
        
        output: [1, 56, 8400] (YOLOv8-Pose 输出)
        返回: [{'bbox': [x1,y1,x2,y2], 'conf': float, 'keypoints': [[x,y,c]*17]}, ...]
        """
        predictions = output[0].T  # [8400, 56]
        
        # 1. 置信度过滤
        mask = predictions[:, 4] > CONF_THRESHOLD
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
        
        # 2. 解码框 (xywh → xyxy)
        boxes = predictions[:, :4]  # xywh
        confs = predictions[:, 4]    # 置信度
        kpts = predictions[:, 5:]    # [N, 51] = 17个关键点 × 3(x,y,conf)
        
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # 3. 去掉 padding，缩放回原始尺寸
        pad_x, pad_y = pad_offset
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        
        # 4. 关键点同样处理
        kpts_reshaped = kpts.reshape(-1, 17, 3)
        kpts_reshaped[:, :, 0] = (kpts_reshaped[:, :, 0] - pad_x) / scale
        kpts_reshaped[:, :, 1] = (kpts_reshaped[:, :, 1] - pad_y) / scale
        
        # 5. NMS 非极大值抑制
        keep = self._nms(xyxy, confs, IOU_THRESHOLD)
        
        # 6. 组装结果
        results = []
        for idx in keep:
            x1, y1, x2, y2 = map(int, xyxy[idx])
            # 限制在画面内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'conf': float(confs[idx]),
                'keypoints': kpts_reshaped[idx].tolist()  # [17, [x, y, conf]]
            })
        
        return results
    
    def _nms(self, boxes, scores, iou_threshold):
        """非极大值抑制"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]  # 按置信度降序
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算 IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep


class GestureClassifier:
    """
    基于 17 个关键点的手势分类器
    
    识别三种手势：
    - wave (挥手): 手腕水平往复运动
    - cheer (欢呼): 双臂举过头顶
    - heart (比心): 双手靠近胸前
    """
    
    def __init__(self, history_size=15):
        self.history_size = history_size
        self.wrist_history = deque(maxlen=history_size)
        self.last_gesture = "none"
        self.stable_count = 0
        
    def classify(self, keypoints):
        """
        keypoints: [17, [x, y, confidence]]
        返回: (gesture_name, confidence)
        """
        kp = np.array(keypoints)
        
        # 检查手腕关键点是否可见
        lw = kp[L_WRIST]   # [x, y, conf]
        rw = kp[R_WRIST]
        ls = kp[L_SHOULDER]
        rs = kp[R_SHOULDER]
        
        # 手腕置信度太低，无法判断
        if lw[2] < 0.3 or rw[2] < 0.3:
            return "none", 0.0
        
        # 记录历史（归一化坐标 0~1）
        # 用画面尺寸做归一化（这里假设 1280x720，实际代码中会修正）
        frame_w, frame_h = 1280, 720
        self.wrist_history.append({
            'lw_x': lw[0] / frame_w,
            'lw_y': lw[1] / frame_h,
            'rw_x': rw[0] / frame_w,
            'rw_y': rw[1] / frame_h,
            'ls_y': ls[1] / frame_h if ls[2] > 0.3 else None,
            'rs_y': rs[1] / frame_h if rs[2] > 0.3 else None,
        })
        
        # 至少需要 5 帧历史才能判断
        if len(self.wrist_history) < 5:
            return "none", 0.0
        
        # 手势判定
        gesture, conf = self._decide()
        
        # 稳定性过滤：连续 3 帧相同结果才输出
        if gesture == self.last_gesture:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_gesture = gesture
        
        if self.stable_count >= 3 and gesture != "none":
            return gesture, conf
        return "none", 0.0
    
    def _decide(self):
        """核心手势判定逻辑"""
        hist = list(self.wrist_history)
        recent = hist[-5:]  # 看最近 5 帧
        
        # ===== 1. 欢呼 (Cheer): 双臂举起 =====
        cheer_frames = 0
        for h in hist:
            if h['ls_y'] is not None and h['rs_y'] is not None:
                # 手腕在肩膀上方（y坐标更小 = 更高）
                if h['lw_y'] < h['ls_y'] - 0.02 and h['rw_y'] < h['rs_y'] - 0.02:
                    cheer_frames += 1
        
        if cheer_frames >= max(3, len(hist) // 2):
            return "cheer", 0.92
        
        # ===== 2. 比心 (Heart): 双手靠近胸前 =====
        heart_frames = 0
        for h in recent:
            # 双手腕欧氏距离很小
            dist = np.sqrt((h['lw_x'] - h['rw_x'])**2 + (h['lw_y'] - h['rw_y'])**2)
            # 位于画面中上部（胸前区域）
            if dist < 0.15 and 0.2 < h['lw_y'] < 0.7 and 0.2 < h['rw_y'] < 0.7:
                heart_frames += 1
        
        if heart_frames >= 3:
            return "heart", 0.88
        
        # ===== 3. 挥手 (Wave): 手腕水平往复运动 =====
        lw_x = [h['lw_x'] for h in hist]
        rw_x = [h['rw_x'] for h in hist]
        
        lw_range = max(lw_x) - min(lw_x)
        rw_range = max(rw_x) - min(rw_x)
        max_range = max(lw_range, rw_range)
        
        # 计算方向变化次数（摆动）
        def count_zigzag(vals):
            diffs = np.diff(vals)
            signs = np.sign(diffs)
            if len(signs) < 2:
                return 0
            return sum(1 for i in range(len(signs)-1) if signs[i] != signs[i+1])
        
        zigzag = max(count_zigzag(lw_x), count_zigzag(rw_x))
        
        # y轴波动应较小（挥手主要是水平运动）
        lw_y_range = max(h['lw_y'] for h in hist) - min(h['lw_y'] for h in hist)
        rw_y_range = max(h['rw_y'] for h in hist) - min(h['rw_y'] for h in hist)
        max_y_range = max(lw_y_range, rw_y_range)
        
        if max_range > 0.08 and zigzag >= 2 and max_y_range < 0.12:
            conf = min(max_range * 5, 0.95)
            return "wave", conf
        
        return "none", 0.0
    
    def reset(self):
        """重置历史"""
        self.wrist_history.clear()
        self.last_gesture = "none"
        self.stable_count = 0


def draw_pose(frame, person, gesture=None, conf=0.0):
    """
    在画面上绘制人体骨架和手势结果
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person['bbox']
    keypoints = person['keypoints']
    
    # 解析关键点坐标
    pts = []
    for k in keypoints:
        if k[2] > 0.3:  # 置信度足够
            pts.append((int(k[0]), int(k[1])))
        else:
            pts.append(None)
    
    # 画骨架连线
    for (a, b) in SKELETON:
        if pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], (255, 128, 0), 2)
    
    # 画关键点
    for i, pt in enumerate(pts):
        if pt:
            # 手腕用红色大圈标记
            if i in [L_WRIST, R_WRIST]:
                cv2.circle(frame, pt, 6, (0, 0, 255), 2)
                cv2.circle(frame, pt, 3, (0, 255, 255), -1)
            else:
                cv2.circle(frame, pt, 3, (0, 255, 255), -1)
    
    # 画人体框
    color = (0, 200, 255) if gesture != "none" else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 显示手势结果
    if gesture != "none":
        text = f"{gesture.upper()} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        # 文字背景
        cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw, y1), (0, 0, 255), -1)
        cv2.putText(frame, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    return frame


def main():
    """主函数"""
    print("=" * 60)
    print("手势识别测试程序")
    print("=" * 60)
    print("请站在摄像头前 1~2 米处")
    print("支持手势: 挥手(wave) / 双手举高(cheer) / 比心(heart)")
    print("=" * 60)
    
    # 初始化检测器
    detector = PoseDetector('yolov8n-pose.onnx')
    gesture_cls = GestureClassifier(history_size=15)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 统计
    fps = 30
    frame_count = 0
    start_time = time.time()
    
    print("\n按 'q' 退出 | 'r' 重置手势 | 's' 截图")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # ===== 1. 人体检测 =====
        blob, scale, pad = detector.preprocess(frame)
        detector.net.setInput(blob)
        outputs = detector.net.forward()
        persons = detector.postprocess(outputs, w, h, scale, pad)
        
        # ===== 2. 手势识别 =====
        gesture, g_conf = "none", 0.0
        target_person = None
        
        if persons:
            # 取最大的人
            target_person = max(persons, key=lambda x: x['conf'])
            gesture, g_conf = gesture_cls.classify(target_person['keypoints'])
        
        # ===== 3. 渲染 =====
        if target_person:
            frame = draw_pose(frame, target_person, gesture, g_conf)
        
        # FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - start_time + 0.001)
            start_time = time.time()
        
        # 信息面板
        info_lines = [
            f"Gesture: {gesture}",
            f"Confidence: {g_conf:.2f}",
            f"FPS: {fps:.1f}",
            "[q]uit [r]eset [s]creenshot"
        ]
        for i, text in enumerate(info_lines):
            cv2.putText(frame, text, (10, 30 + i * 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
        
        cv2.imshow("Gesture Recognition - Day 1", frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            gesture_cls.reset()
            print("[重置] 手势历史已清空")
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[截图] 已保存: {fname}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")


if __name__ == "__main__":
    main()