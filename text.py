#!/usr/bin/env python3
"""
自动测试脚本 - 记录所有模块的真实性能指标
运行后生成 test_results.txt，用于填写技术文档第5条
"""

import cv2
import numpy as np
import time
import os
import sys

# 创建输出文件
RESULTS_FILE = "test_results.txt"
f = open(RESULTS_FILE, "w", encoding="utf-8")

def log(section, content):
    """记录日志到文件和控制台"""
    f.write(f"\n{'='*60}\n")
    f.write(f"{section}\n")
    f.write(f"{'='*60}\n")
    f.write(content + "\n")
    f.flush()
    print(f"\n{'='*60}")
    print(section)
    print('='*60)
    print(content)

# ==================== 5.1.1 测试环境 ====================
log("5.1.1 测试环境", 
"""开发平台: Windows 11 / 后续移植至昇腾Atlas 310B
摄像头: USB摄像头 720P@30fps
麦克风: 笔记本阵列麦克风
Python: 3.10.11
OpenCV: {} 
ONNX Runtime: 待填写（pip show onnxruntime）
PyTorch: 待填写（pip show torch）
NumPy: 待填写（pip show numpy）
""".format(cv2.__version__))

# ==================== 5.1.2 手势识别测试 ====================
print("\n[手势识别测试] 加载模型...")
try:
    from step2_test_gesture import PoseDetector, GestureClassifier
    
    detector = PoseDetector('yolov8n-pose.onnx')
    classifier = GestureClassifier(history_size=15)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("请依次做以下手势（每个持续3秒，按任意键开始下一个）：")
    gestures_to_test = [
        ("wave", "挥手：手臂左右摆动"),
        ("cheer", "欢呼：双手举过头顶"),
        ("heart", "比心：双手在胸前比爱心"),
    ]
    
    gesture_results = {}
    
    for gesture_name, gesture_desc in gestures_to_test:
        input(f"\n准备做 [{gesture_desc}]，按回车开始...")
        print(f"正在检测 {gesture_name}，请保持动作...")
        
        detected_count = 0
        total_frames = 0
        start_time = time.time()
        
        while time.time() - start_time < 5:  # 测试5秒
            ret, frame = cap.read()
            if not ret:
                continue
            
            try:
                blob, scale, pad = detector.preprocess(frame)
                detector.net.setInput(blob)
                outputs = detector.net.forward()
                persons = detector.postprocess(outputs, frame.shape[1], frame.shape[0], scale, pad)
                
                if persons:
                    target = max(persons, key=lambda x: x['conf'])
                    result, conf = classifier.classify(target['keypoints'])
                    if result == gesture_name:
                        detected_count += 1
                    total_frames += 1
                    
                    # 显示当前检测结果
                    cv2.putText(frame, f"Detect: {result} ({conf:.2f})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No person detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Gesture Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error: {e}")
        
        accuracy = (detected_count / total_frames * 100) if total_frames > 0 else 0
        gesture_results[gesture_name] = {
            'accuracy': accuracy,
            'detected': detected_count,
            'total': total_frames
        }
    
    cap.release()
    cv2.destroyAllWindows()
    
    gesture_content = "手势识别测试结果（真实数据）：\n"
    for name, res in gesture_results.items():
        gesture_content += f"  {name}: 识别率 {res['accuracy']:.1f}% ({res['detected']}/{res['total']}帧)\n"
    
    log("5.1.2 手势识别测试", gesture_content)
    
except Exception as e:
    log("5.1.2 手势识别测试", f"测试失败: {e}\n请确认 step2_test_gesture.py 和 yolov8n-pose.onnx 存在")

# ==================== 5.1.3 表情识别测试 ====================
print("\n[表情识别测试] 加载模型...")
try:
    from step3_expression import FaceDetector, ExpressionClassifier
    
    face_det = FaceDetector()
    expr_cls = ExpressionClassifier("models/expression_mobilenetv3.onnx")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    print("请依次做以下表情（每个持续3秒，按回车开始下一个）：")
    expressions_to_test = [
        ("happy", "开心：露齿笑"),
        ("surprise", "惊讶：睁大眼睛张嘴"),
        ("neutral", "中性：自然表情"),
    ]
    
    expr_results = {}
    face_detect_count = 0
    total_frames = 0
    
    for expr_name, expr_desc in expressions_to_test:
        input(f"\n准备做 [{expr_desc}]，按回车开始...")
        print(f"正在检测 {expr_name}，请保持表情...")
        
        detected_count = 0
        valid_frames = 0
        start_time = time.time()
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                continue
            
            try:
                faces = face_det.detect(frame)
                total_frames += 1
                
                if len(faces) > 0:
                    face_detect_count += 1
                    # 取最大人脸
                    areas = [fw*fh for (x,y,fw,fh) in faces]
                    largest_idx = np.argmax(areas)
                    fx, fy, fw, fh = faces[largest_idx]
                    
                    h, w = frame.shape[:2]
                    margin = int(0.2 * fh)
                    y1, y2 = max(0, fy-margin), min(h, fy+fh+margin)
                    x1, x2 = max(0, fx-margin), min(w, fx+fw+margin)
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        expr, conf = expr_cls.predict(face_crop)
                        if expr == expr_name:
                            detected_count += 1
                        valid_frames += 1
                        
                        cv2.putText(frame, f"Expr: {expr} ({conf:.2f})", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Expression Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error: {e}")
        
        accuracy = (detected_count / valid_frames * 100) if valid_frames > 0 else 0
        expr_results[expr_name] = {
            'accuracy': accuracy,
            'detected': detected_count,
            'valid': valid_frames
        }
    
    cap.release()
    cv2.destroyAllWindows()
    
    face_detect_rate = (face_detect_count / total_frames * 100) if total_frames > 0 else 0
    
    expr_content = "表情识别测试结果（真实数据）：\n"
    for name, res in expr_results.items():
        expr_content += f"  {name}: 识别率 {res['accuracy']:.1f}% ({res['detected']}/{res['valid']}帧)\n"
    expr_content += f"  人脸检测率: {face_detect_rate:.1f}% ({face_detect_count}/{total_frames}帧)\n"
    
    log("5.1.3 表情识别测试", expr_content)
    
except Exception as e:
    log("5.1.3 表情识别测试", f"测试失败: {e}")

# ==================== 5.1.4 语音识别测试 ====================
print("\n[语音识别测试]")
try:
    from step4_speech import SpeechModule
    
    speech = SpeechModule(model_size="tiny")
    speech.start()
    
    commands_to_test = ["hello_bear", "take_photo", "goodbye"]
    command_labels = {"hello_bear": "你好", "take_photo": "拍照", "goodbye": "再见"}
    
    speech_results = {}
    
    for cmd in commands_to_test:
        label = command_labels[cmd]
        input(f"\n请说 '{label}'，按回车开始录音...")
        print("录音中...")
        
        detected_count = 0
        total_tests = 3
        latencies = []
        
        for i in range(total_tests):
            print(f"第{i+1}次测试...")
            start_time = time.time()
            result = speech.get_command()
            latency = time.time() - start_time
            
            if result == cmd:
                detected_count += 1
                latencies.append(latency)
            
            time.sleep(0.5)
        
        accuracy = (detected_count / total_tests * 100)
        avg_latency = np.mean(latencies) if latencies else 0
        speech_results[cmd] = {
            'accuracy': accuracy,
            'latency': avg_latency
        }
    
    speech.stop()
    
    speech_content = "语音识别测试结果（真实数据）：\n"
    for cmd, res in speech_results.items():
        label = command_labels[cmd]
        speech_content += f"  '{label}': 触发率 {res['accuracy']:.0f}%，平均延迟 {res['latency']:.1f}s\n"
    
    log("5.1.4 语音识别测试", speech_content)
    
except Exception as e:
    log("5.1.4 语音识别测试", f"测试失败: {e}\n语音模块可能未安装或模型未下载")

# ==================== 5.1.5 输出层性能 ====================
print("\n[输出层性能测试]")
try:
    from output_engine import BearCharacter, create_bear_pair
    
    # 测试FPS
    xiongda, xionger = create_bear_pair(640, 720)
    
    test_frame = np.zeros((720, 640, 3), dtype=np.uint8)
    
    frame_count = 0
    start_time = time.time()
    test_duration = 5  # 测试5秒
    
    while time.time() - start_time < test_duration:
        xiongda.draw(test_frame.copy())
        xionger.draw(test_frame.copy())
        frame_count += 1
    
    actual_duration = time.time() - start_time
    fps = frame_count / actual_duration
    
    perf_content = f"""输出层性能测试结果（真实数据）：
  双熊渲染FPS: {fps:.1f} 帧/秒
  测试帧数: {frame_count}
  测试时长: {actual_duration:.1f}秒
  内存占用: 待填写（使用任务管理器观察）
"""
    log("5.1.5 输出层性能", perf_content)
    
except Exception as e:
    log("5.1.5 输出层性能", f"测试失败: {e}")

# ==================== 总结 ====================
f.close()

print(f"\n{'='*60}")
print(f"[OK] 所有测试完成！结果保存在: {RESULTS_FILE}")
print(f"{'='*60}")
print("\n请:")
print("1. 打开 test_results.txt 查看真实数据")
print("2. 把数据发给我，我会回填到技术文档中")
print("3. 对于'待填写'的项，运行对应命令获取版本号")