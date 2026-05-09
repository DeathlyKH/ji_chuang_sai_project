# CICEC 昇腾开发板部署包 - 熊大熊二互动系统

## 1. 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 主程序（多模态融合+渲染引擎+网络摄像头） |
| `output_engine.py` | 熊大熊二渲染引擎 |
| `step2_test_gesture.py` | 手势识别（YOLOv8-Pose ONNX） |
| `step3_expression.py` | 表情识别（MobileNetV3 ONNX + Haar人脸检测） |
| `step4_speech.py` | 语音识别（Whisper tiny） |
| `camera_server.py` | 【在电脑上运行】USB摄像头推流服务器 |
| `images/` | 熊大熊二素材图 |
| `models/` | ONNX模型文件 |
| `README.md` | 本文件 |

## 2. 网络摄像头方案（开发板无摄像头时使用）

### 2.1 电脑端操作（Windows）

1. 把 `camera_server.py` 复制到Windows电脑
2. 确认电脑有USB摄像头
3. 打开命令提示符，进入该文件所在目录：
```bash
cd /d D:\\你的目录
python camera_server.py
```
4. 记录电脑的IP地址（运行 `ipconfig` 查看，如 `192.168.1.100`）
5. 保持此窗口运行，不要关闭

### 2.2 开发板端配置

编辑 `main.py` 第29-30行：
```python
USE_NETWORK_CAMERA = True       # 改成 True
PC_IP = "192.168.1.100"         # 改成你电脑的实际IP！
PC_PORT = 8080
```

> 注意：电脑和开发板必须在同一局域网内！

### 2.3 网络连通性测试

在开发板上执行：
```bash
ping 192.168.1.100   # 替换成你电脑的实际IP
```
如果能ping通，说明网络正常。

## 3. 开发板部署步骤

### 3.1 安装依赖

```bash
# 进入项目目录
cd ~/ciciec_bear_arm

# 安装基础依赖
pip install opencv-python numpy Pillow

# 安装ONNX Runtime（CPU版本）
pip install onnxruntime

# 可选：语音识别依赖（开发板性能有限，可能较卡）
pip install openai-whisper sounddevice
```

> ARM平台注意：如果 `pip install onnxruntime` 失败，尝试 `pip install onnxruntime`，ARM64一般已有预编译包。如果仍失败，可用 `apt install python3-onnxruntime`。

### 3.2 运行主程序

```bash
python main.py
```

如果连接成功，你会看到：
```
[INFO] 网络摄像头模式：连接 192.168.1.100:8080
[INFO] 正在连接电脑 192.168.1.100:8080...
[OK] 已连接到电脑摄像头！
```

### 3.3 常见问题

| 问题 | 解决方法 |
|------|---------|
| 连接超时 | 检查电脑防火墙，确保8080端口开放；检查IP地址是否正确 |
| 电脑端显示"开发板断开连接" | 开发板端崩溃了，查看开发板报错 |
| 画面卡顿 | 正常现象，ARM算力有限。已在代码中降至320x240分辨率 |
| 语音识别初始化卡住 | 开发板性能不足，可编辑 `main.py` 将 `ENABLE_SPEECH = False` |
| 中文显示方块 | PIL会自动使用系统字体，如仍有问题安装 `fonts-noto-cjk` |

## 4. 文档第5条测试数据收集

由于开发板通常没有显示器，在开发板上直接运行 `run_tests.py` 会报错（因为没有GUI）。

**推荐做法**：
1. 先在Windows电脑上运行 `run_tests.py`，收集基础数据
2. 在开发板上运行简化性能测试（见下方脚本）
3. 把两部分数据合并填入文档

### 开发板专用性能测试

```bash
python -c "
import cv2, numpy as np, time, os, platform
print('平台:', platform.system(), platform.machine())
print('OpenCV:', cv2.__version__)

# 测试模型加载时间
import time
t0 = time.time()
from step2_test_gesture import PoseDetector
det = PoseDetector('models/yolov8n-pose.onnx')
print('手势模型加载:', round(time.time()-t0, 2), 's')

# 测试单帧推理
t0 = time.time()
frame = np.zeros((240, 320, 3), dtype=np.uint8)
blob, scale, pad = det.preprocess(frame)
det.net.setInput(blob)
outs = det.net.forward()
persons = det.postprocess(outs, 320, 240, scale, pad)
print('单帧推理:', round(time.time()-t0, 3), 's')
print('=> FPS:', round(1/(time.time()-t0), 1))
"
```

## 5. 无显示器运行模式（Headless）

如果开发板接的是远程终端（SSH）没有屏幕：

```bash
# 设置虚拟显示
export DISPLAY=:1
Xvfb :1 -screen 0 1024x768x16 &

# 然后运行程序
python main.py
```

或修改 `main.py` 去掉所有 `cv2.imshow` 调用（仅保留 `cv2.waitKey`）。

## 6. 技术参数速查

| 项目 | 数值 |
|------|------|
| 输入分辨率 | 320x240（ARM）/ 640x480（PC） |
| 手势模型 | yolov8n-pose.onnx |
| 表情模型 | expression_mobilenetv3.onnx |
| 语音模型 | Whisper tiny |
| 推理后端 | OpenCV DNN（CPU） |
| 目标FPS | PC: 15-20fps / ARM: 5-8fps |
