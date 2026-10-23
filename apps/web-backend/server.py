#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""ODPlatform 公网安全 Web 推理端(FastAPI + WebSocket).

安全边界(绝不碰服务器硬件/本地文件):
    服务端只处理浏览器主动发来的 Base64 图片, 不调用 infer_yolo,
    不 import frame_source, 不打开任何摄像头或磁盘上的媒体文件.

协议:
    浏览器 --(text: Base64 JPG, 可带 data:image/... 前缀)--> /ws
    服务端 --(text: Base64 JPG 结果)-----------------------> 浏览器
    服务端 --(text: JSON {"type":"error"/"ready", ...})----> 浏览器
    前端靠首字符区分: '{' 开头是 JSON 控制消息, 否则是纯 Base64 图像.

背压: 前端收到一帧结果后才发下一帧(见 HTML 里的 isProcessing 标志).

运行(目录名带连字符, 不能当 Python 包导入, 必须进入目录再启动):
    cd apps\\web-backend
    uvicorn server:app --host 0.0.0.0 --port 8000
或:
    python server.py
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

# ============================================================
# 配置(可用环境变量覆盖)
# ============================================================
MODEL_NAME = os.getenv("WEB_YOLO_MODEL", "yolo11n.pt")     # 权重名/路径
JPEG_QUALITY = int(os.getenv("WEB_JPEG_QUALITY", "90"))    # 回传图片质量
MAX_B64_CHARS = int(os.getenv("WEB_MAX_B64_CHARS", "12000000"))   # 单帧 Base64 上限(~9MB)
MAX_PIXELS = int(os.getenv("WEB_MAX_PIXELS", "20000000"))         # 解码后像素上限(防解压炸弹)
HOST = os.getenv("WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("WEB_PORT", "8000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web-infer")

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="ODPlatform Web Inference", version="0.1.0")

# 公网场景下前后端可能不同源; allow_credentials 必须为 False, 否则浏览器拒绝 "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 模型: 进程内单例 + 串行化推理(避免多连接并发抢显存)
# ============================================================
_model = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def _get_model():
    """首次调用时加载 YOLO(延迟导入, 缺依赖时只影响推理不影响进程启动)。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO      # 延迟导入
                logger.info(f"加载 YOLO 模型: {MODEL_NAME}")
                _model = YOLO(MODEL_NAME)
                logger.info("模型加载完成")
    return _model


def _decode_image(text: str) -> np.ndarray:
    """Base64(可带 data URL 前缀) → BGR ndarray. 任何非法输入抛 ValueError."""
    payload = text.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")      # 只留逗号后的纯 Base64
    payload = payload.strip()
    if not payload:
        raise ValueError("图片数据为空")

    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Base64 解码失败: {e}") from e
    if not raw:
        raise ValueError("图片数据为空")

    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法识别的图片格式")
    h, w = img.shape[:2]
    if w * h > MAX_PIXELS:
        raise ValueError(f"图片过大: {w}x{h} 超过 {MAX_PIXELS} 像素上限")
    return img


def _infer_and_encode(text: str) -> str:
    """单帧推理 + 画框 + JPEG + Base64。阻塞函数, 由线程池执行。"""
    img = _decode_image(text)

    with _infer_lock:                          # 串行化: 同一个模型实例不并发跑
        model = _get_model()
        results = model.predict(source=img, verbose=False)

    if not results:
        raise ValueError("模型没有返回结果")
    annotated = results[0].plot()              # BGR ndarray, 已画框/标签
    ok, buf = cv2.imencode(
        ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        raise ValueError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def _safe_send_error(ws: WebSocket, message: str) -> None:
    """错误消息发送失败(对端已走)就安静收场."""
    try:
        await ws.send_text(json.dumps({"type": "error", "message": message}, ensure_ascii=False))
    except (WebSocketDisconnect, RuntimeError):
        pass


# ============================================================
# 路由
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """返回内嵌的前端页面(HTML + JS, 无需静态目录)."""
    return INDEX_HTML


@app.websocket("/ws")
async def ws_infer(ws: WebSocket) -> None:
    """WebSocket 单帧推理入口.

    单连接内天然串行(收一帧 → 推理 → 回一帧), 坏数据只回错误不中断连接。
    """
    await ws.accept()
    client = f"{ws.client[0]}:{ws.client[1]}" if ws.client else "unknown"
    logger.info(f"WebSocket 连接: {client}")

    try:
        await ws.send_text(json.dumps({"type": "ready", "model": MODEL_NAME}))
    except (WebSocketDisconnect, RuntimeError):
        return

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            text: Optional[str] = msg.get("text")
            if text is None:
                raw = msg.get("bytes") or b""
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    await _safe_send_error(ws, "只支持文本(Base64)消息")
                    continue

            if not text or not text.strip():
                await _safe_send_error(ws, "收到空消息")
                continue
            if len(text) > MAX_B64_CHARS:
                await _safe_send_error(ws, "图片数据过大, 请降低分辨率或压缩质量")
                continue

            try:
                # 推理是阻塞调用, 丢线程池, 不卡事件循环
                out_b64 = await run_in_threadpool(_infer_and_encode, text)
            except ValueError as e:
                await _safe_send_error(ws, str(e))
                continue
            except Exception as e:                      # 兜底: 绝不让单帧异常打挂连接
                logger.exception("推理异常")
                await _safe_send_error(ws, f"推理失败: {e}")
                continue

            try:
                await ws.send_text(out_b64)
            except (WebSocketDisconnect, RuntimeError):
                break
    except WebSocketDisconnect:
        pass
    finally:
        logger.info(f"WebSocket 断开: {client}")


# ============================================================
# 内嵌前端
# ============================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ODPlatform · Web 推理</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #0b0f17; color: #e6edf3;
  }
  header {
    display: flex; align-items: baseline; gap: 12px;
    padding: 14px 20px; border-bottom: 1px solid #1f2937;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header span { font-size: 12px; color: #7d8590; }
  main {
    display: grid; grid-template-columns: minmax(0, 1fr) 320px;
    gap: 16px; padding: 16px; height: calc(100vh - 56px);
  }
  .stage {
    display: flex; align-items: center; justify-content: center;
    background: #111827; border: 1px solid #1f2937; border-radius: 10px; overflow: hidden;
  }
  .stage img { max-width: 100%; max-height: 100%; display: block; }
  aside { display: flex; flex-direction: column; gap: 12px; min-height: 0; }
  .row { display: flex; gap: 8px; }
  button, label.file {
    flex: 1; padding: 9px 12px; border-radius: 8px; border: 1px solid #30363d;
    background: #21262d; color: #e6edf3; cursor: pointer; font-size: 13px;
    text-align: center; user-select: none;
  }
  button:hover:not(:disabled), label.file:hover { background: #30363d; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button.primary { background: #238636; border-color: #2ea043; }
  button.primary:hover:not(:disabled) { background: #2ea043; }
  button.danger { background: #b62324; border-color: #da3633; }
  button.switch {
    background: #1f6feb; border-color: #388bfd; font-weight: 600;
    padding: 11px 12px; font-size: 14px;
  }
  button.switch:hover:not(:disabled) { background: #388bfd; }
  #status { font-size: 13px; color: #58a6ff; min-height: 18px; }
  #log {
    flex: 1; overflow: auto; background: #0d1117; border: 1px solid #1f2937;
    border-radius: 8px; padding: 8px; font: 12px/1.5 Consolas, monospace;
    color: #8b949e; white-space: pre-wrap; word-break: break-all;
  }
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; height: auto; }
    aside { height: 320px; }
  }
</style>
</head>
<body>
<header>
  <h1>ODPlatform · Web 推理</h1>
  <span>浏览器采集 → 服务器单帧推理 → 回传画框图</span>
</header>

<main>
  <div class="stage"><img id="view" alt="等待画面..."/></div>
  <aside>
    <div class="row">
      <button id="btnStart" class="primary">Start</button>
      <button id="btnStop" class="danger" disabled>Stop</button>
    </div>
    <div class="row">
      <button id="btnSwitchCam" class="switch" title="在前置/后置(外接)摄像头之间切换">
        🔄 切换摄像头 (前置)
      </button>
    </div>
    <div class="row">
      <label class="file" for="camera-input">📷 拍照推理</label>
      <label class="file" for="album-input">📁 上传图片</label>
      <input id="camera-input" type="file" accept="image/*" capture="environment" style="display: none;"/>
      <input id="album-input" type="file" accept="image/*" style="display: none;"/>
    </div>
    <div id="status">未连接</div>
    <div id="log"></div>
  </aside>
</main>

<video id="cam" autoplay playsinline muted hidden></video>
<canvas id="canvas" hidden></canvas>

<script>
(() => {
  "use strict";

  const view    = document.getElementById("view");
  const video   = document.getElementById("cam");
  const canvas  = document.getElementById("canvas");
  const ctx     = canvas.getContext("2d");
  const btnStart = document.getElementById("btnStart");
  const btnStop  = document.getElementById("btnStop");
  const btnSwitchCam = document.getElementById("btnSwitchCam");
  const cameraInput = document.getElementById("camera-input");   // 强制后置相机
  const albumInput  = document.getElementById("album-input");    // 相册/文件(无 capture)
  const statusEl = document.getElementById("status");
  const logEl    = document.getElementById("log");

  const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  const INTERVAL_MS = 100;      // 100ms 一拍(10fps; 实际频率再受背压限制)
  const MAX_SIDE = 640;         // 发送前把最长边压到 640px, 大幅降体积/延迟
  const JPEG_Q = 0.6;           // JPEG 压缩质量 60%
  const UPLOAD_JPEG_Q = 0.8;    // 上传单图的压缩质量 80%(比实时流清楚一点)

  let ws = null;
  let timer = null;
  let streaming = false;
  let isProcessing = false;     // 背压: 上一帧还没返回时不发新帧
  let waitingImage = false;     // 上传单图后等第一次结果
  let currentFacingMode = "user";   // "user"=前置 / "environment"=后置(或外接)

  function log(msg) {
    const t = new Date().toLocaleTimeString();
    logEl.textContent += "[" + t + "] " + msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setStatus(msg) { statusEl.textContent = msg; }

  // ---------- 摄像头(前置/后置可切换) ----------
  function facingLabel() {
    return currentFacingMode === "user" ? "前置" : "后置";
  }

  function updateSwitchLabel() {
    btnSwitchCam.textContent = "🔄 切换摄像头 (" + facingLabel() + ")";
  }

  // facingMode 用 ideal: 桌面端多数摄像头会忽略, 不会抛 OverconstrainedError
  function openCamera() {
    return navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: currentFacingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
  }

  // 停掉当前轨道(切换/回收共用)
  function stopTracks() {
    const s = video.srcObject;
    if (s) s.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }

  // ---------- WebSocket ----------
  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return ws;
    ws = new WebSocket(WS_URL);
    ws.onopen = () => { setStatus("已连接"); log("WebSocket 已连接"); };
    ws.onclose = () => {
      setStatus("连接已断开");
      log("WebSocket 已断开");
      isProcessing = false;
    };
    ws.onerror = () => { log("WebSocket 错误(检查服务是否启动)"); };
    ws.onmessage = (ev) => {
      const data = typeof ev.data === "string" ? ev.data : "";
      if (!data) { isProcessing = false; return; }

      if (data.charCodeAt(0) === 123) {       // 123 = '{' → JSON 控制消息
        try {
          const obj = JSON.parse(data);
          if (obj.type === "error") log("服务器: " + obj.message);
          else if (obj.type === "ready") log("就绪, 模型: " + obj.model);
        } catch (e) { /* 非 JSON, 忽略 */ }
        isProcessing = false;                 // 出错也要解锁, 否则永远停住
        return;
      }
      // 渲染完结果图后再解锁: 保证"收到回复才发下一帧"
      view.src = "data:image/jpeg;base64," + data;
      isProcessing = false;
      if (waitingImage) { waitingImage = false; log("单图推理完成"); }
    };
    return ws;
  }

  function waitOpen(timeoutMs = 4000) {
    return new Promise((resolve, reject) => {
      if (ws && ws.readyState === WebSocket.OPEN) return resolve();
      if (!ws) connect();
      const t0 = Date.now();
      const iv = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) { clearInterval(iv); resolve(); }
        else if (Date.now() - t0 > timeoutMs) { clearInterval(iv); reject(new Error("连接超时")); }
      }, 50);
    });
  }

  // ---------- 摄像头流(每 100ms 抓一帧, 背压控制) ----------
  function sendFrame() {
    if (isProcessing) return;                 // 上一帧未返回 → 跳过本拍
    if (!streaming) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2) return;         // 切换摄像头时暂无新帧, 别发黑帧
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return;

    const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
    canvas.width  = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    isProcessing = true;
    try { ws.send(canvas.toDataURL("image/jpeg", JPEG_Q)); }
    catch (e) { isProcessing = false; log("发送失败: " + e.message); }
  }

  // ---------- 按钮 ----------
  async function start() {
    if (streaming) return;
    try {
      const stream = await openCamera();
      video.srcObject = stream;
      await video.play();
    } catch (e) {
      log("无法打开摄像头: " + (e && e.message ? e.message : e));
      setStatus("摄像头不可用(需 HTTPS 或 localhost)");
      return;
    }
    streaming = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    setStatus("采集并推理中...");
    connect();
    timer = setInterval(sendFrame, INTERVAL_MS);
    log("摄像头已开启(" + facingLabel() + "), 开始发送帧");
  }

  function stop() {
    streaming = false;
    if (timer) { clearInterval(timer); timer = null; }
    stopTracks();
    btnStart.disabled = false;
    btnStop.disabled = true;
    isProcessing = false;
    setStatus("已停止");
    log("已停止采集");
  }

  // ---------- 切换摄像头: 先翻状态, 在工作则无缝重开视频流 ----------
  async function switchCamera() {
    const wasStreaming = streaming;                 // 记录切换前是否在出帧
    currentFacingMode = (currentFacingMode === "user") ? "environment" : "user";
    updateSwitchLabel();
    log("切换摄像头 → " + facingLabel());
    if (!wasStreaming) return;                      // 没在工作: 只改状态, 等 Start 时生效

    // 停旧轨道(不动 isProcessing: 在途帧由 onmessage 自然解锁)
    stopTracks();
    try {
      const stream = await openCamera();
      video.srcObject = stream;
      await video.play();
      setStatus("采集并推理中... (" + facingLabel() + ")");
      log("摄像头已切换到" + facingLabel());
    } catch (e) {
      // 目标摄像头不可用(桌面端常见): 回退到原模式重开, 不让页面变砖
      log("切换失败: " + (e && e.message ? e.message : e) + ", 回退...");
      currentFacingMode = (currentFacingMode === "user") ? "environment" : "user";
      updateSwitchLabel();
      try {
        const stream = await openCamera();
        video.srcObject = stream;
        await video.play();
        setStatus("采集并推理中... (" + facingLabel() + ")");
        log("已回退到" + facingLabel());
      } catch (e2) {
        stop();
        log("无法重新打开摄像头: " + (e2 && e2.message ? e2.message : e2));
        setStatus("摄像头不可用");
      }
    }
  }

  // ---------- 上传单图 ----------
  function readAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload  = () => resolve(r.result);
      r.onerror = () => reject(r.error || new Error("读取失败"));
      r.readAsDataURL(file);
    });
  }

  // 把任意图片(含手机拍的几千万像素照片)缩到最长边 640px 再压成 JPEG
  function downscaleImage(dataUrl, maxSide, quality) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        try {
          const w0 = img.naturalWidth  || img.width;
          const h0 = img.naturalHeight || img.height;
          if (!w0 || !h0) throw new Error("图片尺寸无效");
          const scale = Math.min(1, maxSide / Math.max(w0, h0));
          const c = document.createElement("canvas");
          c.width  = Math.max(1, Math.round(w0 * scale));
          c.height = Math.max(1, Math.round(h0 * scale));
          c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
          resolve(c.toDataURL("image/jpeg", quality));
        } catch (e) { reject(e); }
      };
      img.onerror = () => reject(new Error("图片解码失败"));
      img.src = dataUrl;
    });
  }

  // 拍照 / 相册 共用一套处理: FileReader → canvas 缩放 → WebSocket
  async function handleFileSelected(inputEl) {
    const file = inputEl.files && inputEl.files[0];
    inputEl.value = "";                        // 清空, 同一张图可重复选
    if (!file) return;
    if (!file.type.startsWith("image/")) { log("请选择图片文件"); return; }
    if (streaming) { log("请先点击 Stop 再上传单图"); return; }

    try {
      const rawUrl = await readAsDataURL(file);
      const dataUrl = await downscaleImage(rawUrl, MAX_SIDE, UPLOAD_JPEG_Q);
      connect();
      await waitOpen();
      isProcessing = true;
      waitingImage = true;
      ws.send(dataUrl);
      setStatus("单图推理中...");
      log("已上传: " + file.name + " (" + Math.round(file.size / 1024) + " KB, 已缩放到 "
          + MAX_SIDE + "px/" + Math.round(UPLOAD_JPEG_Q * 100) + "%)");
    } catch (e) {
      isProcessing = false;
      waitingImage = false;
      log("上传失败: " + (e && e.message ? e.message : e));
      setStatus("上传失败");
    }
  }

  cameraInput.addEventListener("change", () => handleFileSelected(cameraInput));
  albumInput.addEventListener("change",  () => handleFileSelected(albumInput));

  btnStart.addEventListener("click", start);
  btnStop.addEventListener("click", stop);
  btnSwitchCam.addEventListener("click", switchCamera);
  updateSwitchLabel();
  window.addEventListener("beforeunload", () => { if (streaming) stop(); });
  connect();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
