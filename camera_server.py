#!/usr/bin/env python3
"""
camera_server.py - 电脑端摄像头推流
放在Windows电脑上运行，把USB摄像头画面传给开发板
"""

import cv2
import socket
import struct
import numpy as np

def main():
    # 读取本地摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERR] 电脑摄像头打不开")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    # 创建TCP服务器
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 8080))  # 8080端口
    server.listen(1)
    
    print("=" * 50)
    print("摄像头推流服务器已启动")
    print("等待开发板连接...")
    print("=" * 50)
    
    conn, addr = server.accept()
    print(f"[OK] 开发板已连接: {addr}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # JPEG压缩（减小网络传输量）
            encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1]
            data = encoded.tobytes()
            size = len(data)
            
            # 发送：4字节长度 + 图像数据
            conn.sendall(struct.pack('!I', size) + data)
            
    except (ConnectionResetError, BrokenPipeError):
        print("[WARN] 开发板断开连接")
    finally:
        conn.close()
        cap.release()
        print("[OK] 服务器已关闭")

if __name__ == "__main__":
    main()
