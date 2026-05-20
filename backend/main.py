# คู่มืออ่านไฟล์ main.py นี้:
# - ส่วนต้นไฟล์: ตั้งค่าระบบ, path, env, โมเดล, log
# - ส่วนกลางไฟล์: ฟังก์ชันช่วยคำนวณ/อ่านข้อมูลภาพ/วิเคราะห์
# - ส่วนท้ายไฟล์: API endpoint ที่ frontend เรียกใช้งาน
#
# ถ้าเริ่มอ่านครั้งแรก แนะนำลำดับนี้:
# 1) อ่านค่าคงที่และ config ด้านบน
# 2) อ่านฟังก์ชัน `_analyze_saved_pair()` (แกนหลักของ pipeline)
# 3) อ่าน endpoint `/batch-log`, `/upload-file`, `/analyze`, `/reference-roi`, `/progress`
#
# สรุปความต่างจากโค้ดเก่า:
# 1) จากเดิมรองรับภาพ thermal ไฟล์เดียว
#    -> ใหม่รองรับภาพคู่ thermal + RGB
#
# 2) จากเดิมมีโมเดล YOLO ตัวเดียวสำหรับ hotspot
#    -> ใหม่มี 2 โมเดล:
#       - hotspot model สำหรับภาพ thermal
#       - equipment model สำหรับภาพ RGB
#
# 3) จากเดิม detect ได้แค่ hotspot
#    -> ใหม่สามารถ match hotspot กับ equipment ได้
#
# 4) จากเดิมรายงานแค่ max/min/avg temp
#    -> ใหม่เพิ่ม reference temperature,
#       delta above reference,
#       priority,
#       action required
#
# 5) จากเดิมมี endpoint /upload ตัวเดียว
#    -> ปัจจุบันถอด /upload ออก และใช้ flow แบบอัปโหลดแยกไฟล์:
#       /upload-file อัปโหลดทีละไฟล์ + /analyze ทีหลัง
#
# 6) จากเดิมไม่มี progress tracking
#    -> ใหม่มี request_progress และ endpoint /progress/{request_id}
#
# 7) จากเดิมไม่มี request tracing ชัดเจน
#    -> ใหม่มี request_id, middleware log lifecycle,
#       และ response header x-request-id
#
# 8) จากเดิมส่ง annotated image เป็น base64 data URL
#    -> ใหม่บันทึกเป็นไฟล์จริงใน /uploads แล้วส่ง path กลับ
#
# 9) จากเดิมยังไม่คุม resource ละเอียด
#    -> ใหม่เพิ่ม gc.collect(), ลบ model หลังใช้, คุม torch threads
#
# 10) เวอร์ชันล่าสุดเพิ่ม fixed display range และ /reference-roi
#     -> รองรับการดูภาพด้วยสเกลสีคงที่ และให้ผู้ใช้เลือก ROI เพื่อคำนวณค่าอ้างอิงใหม่ได้
#
# ============================================================

from pathlib import Path
# [ส่วน import เพิ่มเติม]
# ใช้รองรับ dict / function ที่รับค่าหลายชนิด
# จากเดิมมีแค่ Literal, Optional, Tuple
from typing import Any, Literal, Optional, Tuple

# [ส่วน import สำหรับจัดการหน่วยความจำ]
# ใช้ช่วยเก็บกวาดหน่วยความจำหลังโหลดโมเดลหรือประมวลผลเสร็จ
import gc

import io

# [ส่วน import สำหรับระบบ log]
# ใช้ทำ log ฝั่ง backend เพื่อ debug ได้ง่ายขึ้น โดยเฉพาะบน server จริง
import logging

import os
import re
import shutil
import subprocess

# [ส่วน import สำหรับจับเวลา]
# ใช้วัดเวลา request และ progress แต่ละขั้น
import time

import uuid

import exifread
import numpy as np

# [ส่วน import ของ FastAPI ที่ปรับให้รองรับ request object]
# ใช้ Request เป็นหลัก เพราะโค้ดอ่าน json/stream และ header/query จาก request ตรง ๆ
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# [ส่วน import สำหรับส่ง JSON error]
# ใช้คืน error response แบบกำหนด status code ได้ชัดเจน
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

# [ส่วน import สำหรับจับกรณี client หลุด]
# ใช้จับกรณี client หลุดระหว่างอัปโหลดไฟล์
from starlette.requests import ClientDisconnect

# [ส่วน import ของ PyTorch สำหรับคุม resource]
# ใช้คุมจำนวน threads ของ PyTorch และช่วยเรื่อง resource
import torch

from ultralytics import YOLO


# ------------------------------
# ส่วนที่ 1: path หลักและค่าคงที่พื้นฐานของโปรเจกต์
# ใช้กำหนดโฟลเดอร์หลัก ชื่อไฟล์โมเดล และค่าคงที่ที่หลายฟังก์ชันจะเรียกใช้ร่วมกัน
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

# [ส่วน path ของโมเดล]
# เดิมมี model เดียวคือ best.pt สำหรับ hotspot
# เวอร์ชันนี้แยก default model path สำหรับ hotspot และ equipment
DEFAULT_HOTSPOT_MODEL_PATH = BASE_DIR / "model" / "best.pt"
DEFAULT_EQUIPMENT_MODEL_PATH = BASE_DIR / "model" / "equipment.pt"

# [ส่วนรายการนามสกุลไฟล์ที่อนุญาต]
# จำกัดนามสกุลไฟล์ภาพที่ระบบยอมรับ
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

# [ส่วนแปลงรหัสคลาสเป็นชื่ออุปกรณ์]
# map class id ของโมเดล equipment -> ชื่ออุปกรณ์
# ใช้ตอนส่งผลลัพธ์กลับ frontend ให้อ่านง่ายขึ้น
EQUIPMENT_LABELS = {
    0: "inverter",
    1: "transformer",
    2: "conductor",
    3: "connector",
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# ------------------------------
# [ส่วนระบบ log]
# ระบบ logging ของ backend
# ------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("thermal_app")

# ------------------------------
# [ส่วนติดตามสถานะงาน]
# ใช้เก็บ progress ของแต่ละ request
# frontend สามารถ poll มาดูสถานะได้จาก /progress/{request_id}
# ------------------------------
request_progress: dict[str, dict[str, Any]] = {}

# อายุสูงสุดของ progress แต่ละรายการ
PROGRESS_TTL_SECONDS = 60 * 60

# จำนวน entry สูงสุดที่ยอมให้เก็บใน memory
MAX_PROGRESS_ENTRIES = 256


# ------------------------------
# [ฟังก์ชันดูแลหน่วยความจำของ progress]
# ลบ progress ที่เก่าเกินไป หรือเกินจำนวนสูงสุด
# กัน memory โตเรื่อย ๆ
# ------------------------------
def _prune_request_progress(now: float) -> None:
    """
    ล้างรายการ progress เก่าที่หมดอายุ หรือเกินจำนวนที่กำหนด
    เพื่อลดการใช้ RAM ของเซิร์ฟเวอร์เมื่อมี request เข้ามาต่อเนื่อง
    """
    expired_request_ids = [
        request_id
        for request_id, progress in request_progress.items()
        if now - float(progress.get("updated_at", now)) > PROGRESS_TTL_SECONDS
    ]
    for request_id in expired_request_ids:
        request_progress.pop(request_id, None)

    while len(request_progress) > MAX_PROGRESS_ENTRIES:
        oldest_request_id = min(
            request_progress,
            key=lambda item: float(request_progress[item].get("updated_at", now)),
        )
        request_progress.pop(oldest_request_id, None)


# ------------------------------
# [ฟังก์ชันอัปเดต progress]
# ฟังก์ชันกลางสำหรับอัปเดต progress ของ request
# ใช้เก็บ step ปัจจุบัน, รายละเอียดเสริม, เวลาเริ่ม, เวลาอัปเดต
# ------------------------------
def _set_request_progress(request_id: str, step: str, **details: Any) -> None:
    """
    อัปเดตสถานะของ request ทีละขั้น (step) พร้อมรายละเอียดประกอบ
    เช่น ชื่อไฟล์, จำนวน bytes, หรือข้อความบอกสาเหตุที่ล้มเหลว
    """
    now = time.time()
    progress = request_progress.get(request_id)
    if progress is None:
        progress = {
            "request_id": request_id,
            "started_at": now,
            "finished": False,
            "failed": False,
        }
        request_progress[request_id] = progress

    progress["step"] = step
    progress["details"] = details
    progress["updated_at"] = now
    progress["elapsed_seconds"] = round(now - float(progress.get("started_at", now)), 1)
    _prune_request_progress(now)


# ------------------------------
# [middleware ติดตามคำขอ]
# middleware นี้ทำงานกับทุก request
# หน้าที่:
# 1) สร้าง/อ่าน request_id
# 2) log ตอน request เริ่ม
# 3) จับ error ถ้า request ล้มเหลว
# 4) แนบ x-request-id กลับไปใน response
# ------------------------------
@app.middleware("http")
async def log_request_lifecycle(request: Request, call_next):
    """
    middleware กลางที่ทำงานกับทุก endpoint:
    - สร้าง request_id เพื่อใช้ตามรอยงาน
    - บันทึกจุดเริ่ม/จบของคำขอ
    - ติดธง failed/finished เวลามี error
    - แนบ x-request-id กลับใน response
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    started_at = time.perf_counter()

    _set_request_progress(
        request_id,
        "http_request_started",
        method=request.method,
        path=request.url.path,
        content_length=request.headers.get("content-length"),
    )

    logger.info(
        "[%s] http_request_started method=%s path=%s content_length=%s",
        request_id,
        request.method,
        request.url.path,
        request.headers.get("content-length"),
    )

    try:
        response = await call_next(request)
    except Exception:
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        progress = request_progress.get(request_id)
        if progress is None or progress.get("step") == "http_request_started":
            _set_request_progress(request_id, "http_request_failed", elapsed_seconds=elapsed_seconds)
        request_progress[request_id]["failed"] = True
        request_progress[request_id]["finished"] = True
        logger.exception("[%s] http_request_failed elapsed_seconds=%s", request_id, elapsed_seconds)
        raise

    response.headers["x-request-id"] = request_id
    elapsed_seconds = round(time.perf_counter() - started_at, 2)

    progress = request_progress.get(request_id)
    if progress is None:
        _set_request_progress(
            request_id,
            "http_request_finished",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_seconds=elapsed_seconds,
        )
        progress = request_progress[request_id]

    progress["finished"] = True
    progress["status_code"] = response.status_code
    progress["updated_at"] = time.time()
    progress["elapsed_seconds"] = elapsed_seconds

    logger.info(
        "[%s] http_request_finished method=%s path=%s status=%s elapsed_seconds=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_seconds,
    )
    return response


# ------------------------------
# [ตัวช่วยอ่านค่า env แบบ float]
# helper อ่าน env แล้วแปลงเป็น float แบบปลอดภัย
# ถ้าอ่านไม่ได้ให้ fallback เป็น default
# ------------------------------
def _env_float(name: str, default_value: float) -> float:
    """
    อ่านค่าจาก env แล้วแปลงเป็น float
    ถ้าไม่ได้ตั้งหรือแปลงไม่สำเร็จ ให้ใช้ค่า default เพื่อกันแอปล้ม
    """
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        return float(raw_value)
    except ValueError:
        return default_value


# ------------------------------
# [ตัวช่วยอ่านค่า env แบบ int]
# helper อ่าน env แล้วแปลงเป็น int แบบปลอดภัย
# ------------------------------
def _env_int(name: str, default_value: int) -> int:
    """
    อ่านค่าจาก env แล้วแปลงเป็น int แบบทนต่อค่าผิดรูปแบบ
    """
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def _parse_positive_int(value: str | None) -> int | None:
    if value is None:
        return None

    raw_value = value.strip()
    if not raw_value:
        return None

    try:
        parsed_value = int(raw_value)
    except ValueError:
        return None

    return parsed_value if parsed_value > 0 else None


def _get_batch_request_context(request: Request) -> dict[str, Any]:
    return {
        "batch_run_id": (request.headers.get("x-batch-run-id") or "").strip(),
        "file_total": _parse_positive_int(request.headers.get("x-batch-file-total")),
        "file_names": (request.headers.get("x-batch-file-names") or "").strip(),
        "item_index": _parse_positive_int(request.headers.get("x-batch-item-index")),
        "item_total": _parse_positive_int(request.headers.get("x-batch-item-total")),
        "item_label": (request.headers.get("x-batch-item-label") or "").strip(),
        "thermal_file_name": (request.headers.get("x-batch-item-thermal-name") or "").strip(),
        "rgb_file_name": (request.headers.get("x-batch-item-rgb-name") or "").strip(),
    }


def _log_batch_event(request_id: str, event: str, batch_context: dict[str, Any], **details: Any) -> None:
    detail_parts: list[str] = []

    file_total = batch_context.get("file_total")
    if file_total is not None:
        detail_parts.append(f"files={file_total}")

    file_names = batch_context.get("file_names")
    if file_names:
        detail_parts.append(f"file_names={file_names}")

    item_index = batch_context.get("item_index")
    item_total = batch_context.get("item_total")
    if item_index is not None and item_total is not None:
        detail_parts.append(f"pair={item_index}/{item_total}")

    item_label = batch_context.get("item_label")
    if item_label:
        detail_parts.append(f"label={item_label}")

    thermal_file_name = batch_context.get("thermal_file_name")
    if thermal_file_name:
        detail_parts.append(f"thermal_file={thermal_file_name}")

    rgb_file_name = batch_context.get("rgb_file_name")
    if rgb_file_name:
        detail_parts.append(f"rgb_file={rgb_file_name}")

    for key, value in details.items():
        detail_parts.append(f"{key}={value}")

    if detail_parts:
        logger.info("[%s] %s %s", request_id, event, " ".join(detail_parts))
    else:
        logger.info("[%s] %s", request_id, event)


# ------------------------------
# [เพิ่มในเวอร์ชันล่าสุด]
# ส่วนนี้กำหนด "ช่วงอุณหภูมิสำหรับการแสดงผล"
# เพื่อให้ภาพ thermal หลายภาพใช้สเกลสีเดียวกันและเทียบกันได้ตรงขึ้น
# หมายเหตุ: ใช้เพื่อการแสดงผลเท่านั้น ไม่ได้เปลี่ยนค่าที่ใช้วิเคราะห์จริง
# ------------------------------
DISPLAY_TEMP_MIN_C = _env_float("DISPLAY_TEMP_MIN_C", 25.0)
DISPLAY_TEMP_MAX_C = _env_float("DISPLAY_TEMP_MAX_C", 40.0)

THERMAL_DISPLAY_COLOR_STOPS = np.array(
    [
        [0.0, 6, 5, 24],
        [0.16, 47, 15, 104],
        [0.32, 101, 21, 110],
        [0.48, 159, 42, 99],
        [0.64, 212, 72, 66],
        [0.8, 245, 125, 21],
        [0.92, 250, 187, 55],
        [1.0, 252, 255, 164],
    ],
    dtype=np.float32,
)


# ------------------------------
# [ตัวช่วยหา path ของโมเดล]
# resolve path ของ model จาก env หรือ default path
# รองรับทั้ง absolute path และ relative path
# ------------------------------
def _resolve_model_path(raw_path: str, default_path: Path) -> Path:
    """
    หา path ที่จะใช้โหลดโมเดลจริง:
    - ถ้า env ถูกตั้งไว้ จะใช้ค่านั้น
    - ถ้าไม่ตั้ง จะใช้ default
    - ถ้าเป็น relative path จะอ้างอิงจาก BASE_DIR
    """
    candidate = Path(raw_path).expanduser() if raw_path.strip() else default_path
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


# ------------------------------
# ส่วนที่ 2: การตั้งค่า CORS
# ใช้กำหนดว่า frontend จาก origin ไหนเรียก API นี้ได้บ้าง
# ------------------------------
def _load_cors_origins() -> list[str]:
    """
    สร้างรายการต้นทาง (origin) ที่อนุญาตให้เรียก API
    ใช้สำหรับควบคุมความปลอดภัยฝั่ง browser (CORS)
    """
    configured_origins = os.getenv("CORS_ORIGINS", "")
    if configured_origins.strip():
        return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


ALLOW_ALL_CORS = os.getenv("CORS_ALLOW_ALL", "false").strip().lower() == "true"
CORS_ORIGINS = ["*"] if ALLOW_ALL_CORS else _load_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=not ALLOW_ALL_CORS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
# ส่วนที่ 3: config หลักของระบบจาก environment variables
# ใช้ปรับพฤติกรรมระบบ เช่น threshold, ขนาดภาพ, การ align thermal กับ RGB โดยไม่ต้องแก้โค้ดตรง ๆ
# ------------------------------
def _load_yolo_device() -> str:
    """
    เลือก device สำหรับ YOLO:
    - ถ้าไม่ตั้ง YOLO_DEVICE หรือใช้ auto จะเลือก cuda เมื่อมี GPU และ fallback เป็น cpu เมื่อไม่มี GPU
    - ถ้าตั้ง YOLO_DEVICE เองเป็น cuda/cpu จะใช้ค่านั้นตรง ๆ
    - ถ้าบังคับ cuda แต่ PyTorch ไม่เห็น CUDA จะ error เพื่อให้รู้ว่าติดตั้ง torch/GPU ยังไม่พร้อม
    """
    configured_device = os.getenv("YOLO_DEVICE", "auto").strip() or "auto"
    normalized_device = configured_device.lower()

    if normalized_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "YOLO_DEVICE requires CUDA, but PyTorch cannot see CUDA. "
            "Use YOLO_DEVICE=auto/cpu or install a CUDA-enabled torch build in backend\\venv before starting the backend."
        )

    return configured_device


YOLO_DEVICE = _load_yolo_device()
# HOTSPOT_CONFIDENCE = _env_float("HOTSPOT_CONFIDENCE", 0.34)
HOTSPOT_CONFIDENCE = _env_float("HOTSPOT_CONFIDENCE", 0.34)
HOTSPOT_IOU = _env_float("HOTSPOT_IOU", 0.7)
EQUIPMENT_CONFIDENCE = _env_float("EQUIPMENT_CONFIDENCE", 0.2)
EQUIPMENT_IOU = _env_float("EQUIPMENT_IOU", 0.5)

# [ค่าเลื่อนตำแหน่ง thermal]
# ใช้ปรับตำแหน่ง thermal overlay บน RGB
THERMAL_CENTER_SHIFT_X = _env_int("THERMAL_CENTER_SHIFT_X", -10)
THERMAL_CENTER_SHIFT_Y = _env_int("THERMAL_CENTER_SHIFT_Y", -1)

# [ค่าขยายกรอบอุปกรณ์]
# ใช้ขยาย bbox ของ equipment เพื่อช่วยการ match
EQUIPMENT_BBOX_DILATION = _env_int("EQUIPMENT_BBOX_DILATION", 100)

# [ค่าระยะสูงสุดสำหรับการจับคู่]
# ระยะ threshold สูงสุดสำหรับ match แบบ nearest
MATCH_DISTANCE_THRESHOLD = _env_float("MATCH_DISTANCE_THRESHOLD", 40.0)

# [ค่าเกณฑ์สำหรับหา reference temperature]
# ค่าสูงสุดของ pixel ที่จะนับเป็น reference temperature
REFERENCE_TEMP_MAX_C = _env_float("REFERENCE_TEMP_MAX_C", 28.0)

# [ค่าการเตรียมภาพ RGB ก่อนตรวจจับ]
# จำกัดขนาดรูป RGB ตอน detect และ margin ตอน crop
# ถ้าตั้ง RGB_DETECTION_CROP_MARGIN เป็น -1 จะไม่ crop และส่ง RGB เต็มภาพเข้า model
RGB_DETECTION_MAX_DIM = _env_int("RGB_DETECTION_MAX_DIM", 1600)
RGB_DETECTION_CROP_MARGIN = _env_int("RGB_DETECTION_CROP_MARGIN", 800)

# [ค่าขนาดภาพสำหรับโมเดล]
# imgsz แยกของ hotspot / equipment model
HOTSPOT_IMGSZ = _env_int("HOTSPOT_IMGSZ", 960)
# เปลี่ยนจาก 640 เป็น 1280 เพื่อให้โมเดล equipment ตรวจจับอุปกรณ์เล็ก ๆ ได้ดีขึ้น
EQUIPMENT_IMGSZ = _env_int("EQUIPMENT_IMGSZ", 1280)

# [ค่าจำนวน threads ของ PyTorch]
# จำนวน threads ของ PyTorch
TORCH_NUM_THREADS = max(1, _env_int("TORCH_NUM_THREADS", 1))
TORCH_INTEROP_THREADS = max(1, _env_int("TORCH_INTEROP_THREADS", 1))

HOTSPOT_MODEL_PATH = _resolve_model_path(os.getenv("HOTSPOT_MODEL_PATH", ""), DEFAULT_HOTSPOT_MODEL_PATH)
EQUIPMENT_MODEL_PATH = _resolve_model_path(os.getenv("EQUIPMENT_MODEL_PATH", ""), DEFAULT_EQUIPMENT_MODEL_PATH)


# ------------------------------
# [ตั้งค่าการใช้ทรัพยากรของ PyTorch]
# จำกัดจำนวน threads ของ PyTorch
# ช่วยคุม resource บนเครื่องที่มีทรัพยากรจำกัด
# ------------------------------
torch.set_num_threads(TORCH_NUM_THREADS)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    except RuntimeError:
        pass


# ------------------------------
# [ฟังก์ชันโหลดโมเดล]
# โหลด model จาก path
# required=True  -> ถ้าไม่เจอ model ให้ error
# required=False -> ถ้าไม่เจอ model ให้คืน None
# ------------------------------
def _load_yolo_model(model_path: Path, required: bool) -> Optional[YOLO]:
    """
    โหลดไฟล์โมเดล YOLO จาก disk
    required=True  หมายถึงขาดไม่ได้ (ไม่เจอแล้วให้ error)
    required=False หมายถึงถ้าไม่มีก็ข้ามได้ (คืน None)
    """
    if not model_path.exists():
        if required:
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        return None
    return YOLO(str(model_path))


# log config model ตอนเริ่ม backend
logger.info(
    "models_config hotspot_model=%s hotspot_exists=%s equipment_model=%s equipment_exists=%s yolo_device=%s cuda_available=%s hotspot_imgsz=%s equipment_imgsz=%s torch_threads=%s torch_interop_threads=%s",
    HOTSPOT_MODEL_PATH,
    HOTSPOT_MODEL_PATH.exists(),
    EQUIPMENT_MODEL_PATH,
    EQUIPMENT_MODEL_PATH.exists(),
    YOLO_DEVICE,
    torch.cuda.is_available(),
    HOTSPOT_IMGSZ,
    EQUIPMENT_IMGSZ,
    TORCH_NUM_THREADS,
    TORCH_INTEROP_THREADS,
)


# ------------------------------
# path ค่าปริยายของ ExifTool
# ส่วนนี้หลักการเหมือนโค้ดเก่า
# ------------------------------
EXIFTOOL_DEFAULT_PATHS = [
    r"C:\exiftool\exiftool.exe",
    r"C:\Program Files\ExifTool\exiftool.exe",
    r"C:\Program Files (x86)\ExifTool\exiftool.exe",
    str(Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "ExifTool" / "ExifTool.exe"),
]

# path ค่าปริยายของ DJI IRP
DJI_IRP_WINDOWS_X64_DIR = BASE_DIR / "tools" / "dji-tsdk" / "utility" / "bin" / "windows" / "release_x64"
DJI_IRP_LINUX_X64_DIR = BASE_DIR / "tools" / "dji-tsdk" / "utility" / "bin" / "linux" / "release_x64"

if os.name == "nt":
    DJI_IRP_DEFAULT_PATHS = [
        str(DJI_IRP_WINDOWS_X64_DIR / "dji_irp.exe"),
        str(DJI_IRP_WINDOWS_X64_DIR / "dji_irp_omp.exe"),
        str(DJI_IRP_LINUX_X64_DIR / "dji_irp"),
    ]
else:
    DJI_IRP_DEFAULT_PATHS = [
        str(DJI_IRP_LINUX_X64_DIR / "dji_irp"),
        str(DJI_IRP_LINUX_X64_DIR / "dji_irp_omp"),
        str(DJI_IRP_WINDOWS_X64_DIR / "dji_irp.exe"),
    ]

# ขนาดภาพ thermal ที่พบได้บ่อย
KNOWN_THERMAL_SIZES = [
    (512, 640),
    (256, 320),
]


# ------------------------------
# แปลง DMS -> decimal degrees
# ส่วนนี้เหมือนโค้ดเก่า
# ------------------------------
def dms_to_decimal(dms, ref):
    """
    แปลงพิกัด GPS รูปแบบ DMS (องศา-ลิปดา-พิลิปดา) เป็นเลขทศนิยม
    เช่น 13° 45' 30" -> 13.7583
    """
    d = float(dms.values[0])
    m = float(dms.values[1])
    s = float(dms.values[2])

    decimal = d + (m / 60) + (s / 3600)
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal


# ------------------------------
# หา path ของ exiftool
# ส่วนนี้เหมือนโค้ดเก่า
# ------------------------------
def resolve_exiftool_path() -> Optional[str]:
    """
    หา path ของ exiftool ตามลำดับความสำคัญ:
    1) ENV EXIFTOOL_PATH
    2) path ค่าปริยายที่เตรียมไว้
    3) คำสั่ง exiftool ใน PATH ของระบบ
    """
    env_path = os.getenv("EXIFTOOL_PATH")
    candidates = [env_path] if env_path else []
    candidates.extend(EXIFTOOL_DEFAULT_PATHS)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return shutil.which("exiftool") or shutil.which("exiftool.exe")


EXIFTOOL_PATH = resolve_exiftool_path()


# ------------------------------
# หา path ของ dji_irp
# ส่วนนี้เหมือนโค้ดเก่า
# ------------------------------
def resolve_dji_irp_path() -> Optional[str]:
    """
    หา path ของ binary `dji_irp` ที่ใช้ดึงข้อมูล thermal จากไฟล์ DJI
    คืน None ถ้าไม่พบ เพื่อให้โค้ดส่วนเรียกใช้งานตัดสินใจ fallback ได้
    """
    env_path = os.getenv("DJI_IRP_PATH")
    candidates = [env_path] if env_path else []
    candidates.extend(DJI_IRP_DEFAULT_PATHS)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return shutil.which("dji_irp") or shutil.which("dji_irp.exe")


DJI_IRP_PATH = resolve_dji_irp_path()


# ------------------------------
# ฟังก์ชันกลางสำหรับรัน exiftool
# เหมือนเดิม
# ------------------------------
def _run_exiftool(args, image_path: str, text: bool = False):
    """
    ตัวช่วยรันคำสั่ง exiftool แบบรวมศูนย์
    เพื่อให้ทุกจุดในระบบเรียก exiftool ด้วยรูปแบบเดียวกัน
    """
    global EXIFTOOL_PATH
    if not EXIFTOOL_PATH:
        EXIFTOOL_PATH = resolve_exiftool_path()

    if not EXIFTOOL_PATH:
        raise FileNotFoundError(
            "ExifTool not found. Install exiftool and set EXIFTOOL_PATH, or add exiftool to PATH."
        )

    cmd = [EXIFTOOL_PATH, *args, image_path]
    return subprocess.run(cmd, capture_output=True, check=True, text=text)


# ------------------------------
# ดึงเลขตัวแรกจากข้อความแล้วแปลงเป็น float
# เหมือนเดิม
# ------------------------------
def _parse_numeric_value(text: str, default_value: float) -> float:
    """
    ดึง "เลขตัวแรก" ออกจากข้อความ และแปลงเป็น float
    ใช้กับข้อมูล EXIF ที่มักมาเป็นข้อความผสมหน่วย เช่น '5.2 m'
    """
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    if not match:
        return default_value
    try:
        return float(match.group(0))
    except ValueError:
        return default_value


# ------------------------------
# อ่าน measurement params ของ DJI จาก EXIF
# เหมือนเดิม
# ------------------------------
def _get_dji_measurement_params(image_path: str) -> Tuple[float, float, float, float]:
    """
    อ่านค่าพารามิเตอร์ที่มีผลต่อการคำนวณอุณหภูมิจาก EXIF ของภาพ DJI:
    distance, humidity, emissivity, reflection
    ถ้าอ่านไม่ได้ จะใช้ค่า default ที่ปลอดภัย
    """
    default_distance = 5.0
    default_humidity = 50.0
    default_emissivity = 0.95
    default_reflection = 25.0

    try:
        output = _run_exiftool(
            ["-s3", "-ObjectDistance", "-RelativeHumidity", "-Emissivity", "-ReflectedTemperature"],
            image_path,
            text=True,
        ).stdout
    except Exception:
        return default_distance, default_humidity, default_emissivity, default_reflection

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 4:
        return default_distance, default_humidity, default_emissivity, default_reflection

    distance = _parse_numeric_value(lines[0], default_distance)
    humidity = _parse_numeric_value(lines[1], default_humidity)
    emissivity = _parse_numeric_value(lines[2], default_emissivity)
    reflection = _parse_numeric_value(lines[3], default_reflection)

    distance = min(max(distance, 1.0), 25.0)
    humidity = min(max(humidity, 20.0), 100.0)
    emissivity = min(max(emissivity, 0.1), 1.0)
    reflection = min(max(reflection, -40.0), 500.0)

    return distance, humidity, emissivity, reflection


# ------------------------------
# ใช้ DJI IRP สร้างเมทริกซ์อุณหภูมิจากภาพ
# ส่วนนี้หลักเหมือนเดิม
# ------------------------------
def _extract_dji_temperature_matrix(
    image_path: str,
    expected_width: int,
    expected_height: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    เรียกโปรแกรม `dji_irp` เพื่อแปลงภาพเป็นเมทริกซ์อุณหภูมิจริง (องศา C)
    คืนค่าเป็น (matrix, error_message) โดย matrix จะเป็น None เมื่อเกิดปัญหา
    """
    global DJI_IRP_PATH
    if not DJI_IRP_PATH:
        DJI_IRP_PATH = resolve_dji_irp_path()
    if not DJI_IRP_PATH:
        return None, "DJI Thermal SDK binary (dji_irp) not found."
    if os.name != "nt" and DJI_IRP_PATH.lower().endswith(".exe"):
        return None, f"DJI IRP binary is Windows-only and cannot run here: {DJI_IRP_PATH}"

    distance, humidity, emissivity, reflection = _get_dji_measurement_params(image_path)
    output_raw_path = f"{image_path}.dji_measure.float32.raw"

    cmd = [
        DJI_IRP_PATH,
        "-s",
        image_path,
        "-a",
        "measure",
        "-o",
        output_raw_path,
        "--measurefmt",
        "float32",
        "--distance",
        f"{distance:.3f}",
        "--humidity",
        f"{humidity:.3f}",
        "--emissivity",
        f"{emissivity:.3f}",
        "--reflection",
        f"{reflection:.3f}",
    ]

    process_env = os.environ.copy()
    dji_irp_directory = str(Path(DJI_IRP_PATH).resolve().parent)
    if os.name != "nt":
        existing_library_path = process_env.get("LD_LIBRARY_PATH", "")
        process_env["LD_LIBRARY_PATH"] = (
            f"{dji_irp_directory}:{existing_library_path}" if existing_library_path else dji_irp_directory
        )

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, env=process_env)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if not detail:
            detail = "Failed to execute DJI thermal measurement."
        return None, detail
    except FileNotFoundError:
        return None, f"DJI IRP binary not found at runtime: {DJI_IRP_PATH}"
    except OSError as exc:
        return None, f"DJI IRP cannot execute in this environment: {exc}"

    try:
        temperature_values = np.fromfile(output_raw_path, dtype=np.float32)
    except Exception:
        return None, "DJI measurement output cannot be read."
    finally:
        if os.path.exists(output_raw_path):
            os.remove(output_raw_path)

    expected_pixels = expected_width * expected_height
    if temperature_values.size != expected_pixels:
        return (
            None,
            f"Unexpected DJI measurement size: {temperature_values.size}, expected {expected_pixels}.",
        )

    temperature_matrix = temperature_values.reshape((expected_height, expected_width))
    return temperature_matrix, None


# ------------------------------
# พยายาม parse payload TIFF ให้เป็น thermal matrix
# เหมือนเดิม
# ------------------------------
def _parse_thermal_from_tiff(payload: bytes) -> Optional[np.ndarray]:
    """
    พยายามอ่าน binary payload ให้เป็นภาพ TIFF แล้วแปลงเป็น numpy 2D
    ถ้าอ่านไม่ได้หรือรูปแบบไม่ถูกต้อง จะคืน None
    """
    try:
        with Image.open(io.BytesIO(payload)) as thermal_img:
            matrix = np.array(thermal_img)
    except Exception:
        return None

    if matrix.ndim == 3:
        matrix = matrix[..., 0]
    if matrix.ndim != 2:
        return None

    return matrix.astype(np.float32)


# ------------------------------
# ขอขนาด RawThermalImage จาก EXIF
# เหมือนเดิม
# ------------------------------
def _get_raw_thermal_size(image_path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    อ่านขนาดภาพความร้อนดิบจาก EXIF (RawThermalImageWidth/Height)
    เพื่อใช้ reshape ข้อมูล binary ให้ถูกมิติ
    """
    try:
        output = _run_exiftool(
            ["-s3", "-RawThermalImageWidth", "-RawThermalImageHeight"],
            image_path,
            text=True,
        ).stdout
    except Exception:
        return None, None

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return None, None

    try:
        thermal_image_width = int(float(lines[0]))
        thermal_image_height = int(float(lines[1]))
    except ValueError:
        return None, None

    if thermal_image_width <= 0 or thermal_image_height <= 0:
        return None, None

    return thermal_image_width, thermal_image_height


# ------------------------------
# ดึง binary payload จาก EXIF tag ที่กำหนด
# เหมือนเดิม
# ------------------------------
def _extract_binary_tag(image_path: str, tag_name: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    ดึงข้อมูลดิบ (binary) จาก EXIF tag เช่น RawThermalImage/ThermalData
    คืนทั้ง payload และ error message เพื่อให้ผู้เรียกเลือกแนวทางต่อได้
    """
    try:
        result = _run_exiftool(["-b", f"-{tag_name}"], image_path, text=False)
    except FileNotFoundError as exc:
        return None, str(exc)
    except subprocess.CalledProcessError:
        return b"", None

    return result.stdout or b"", None


# ------------------------------
# decode payload แบบ u16 เป็น matrix
# เหมือนเดิม
# ------------------------------
def _decode_u16_payload(
    payload: bytes,
    expected_width: Optional[int],
    expected_height: Optional[int],
) -> Optional[np.ndarray]:
    """
    แปลง payload แบบ unsigned 16-bit เป็นเมทริกซ์ 2 มิติ
    ถ้ารู้ขนาดภาพจะใช้ขนาดนั้นก่อน ไม่รู้ก็ลองเทียบกับขนาดที่พบบ่อย
    """
    if len(payload) % 2 != 0:
        return None

    raw_values = np.frombuffer(payload, dtype="<u2")
    if expected_width and expected_height and raw_values.size == expected_width * expected_height:
        return raw_values.reshape((expected_height, expected_width)).astype(np.float32)

    for known_height, known_width in KNOWN_THERMAL_SIZES:
        if raw_values.size == known_height * known_width:
            return raw_values.reshape((known_height, known_width)).astype(np.float32)

    return None


# ------------------------------
# ฟังก์ชันหลักในการดึง thermal matrix
# โครงหลักยังเหมือนเดิม:
# - absolute
# - relative
# - none
# ------------------------------
def extract_thermal_matrix(
    image_path: str,
    expected_width: int,
    expected_height: int,
) -> Tuple[Optional[np.ndarray], Optional[str], Literal["none", "absolute", "relative"]]:
    """
    จุดรวมการอ่านข้อมูล thermal จากไฟล์ภาพ:
    - absolute: ได้อุณหภูมิจริง
    - relative: มีข้อมูลความร้อนแต่ไม่ใช่อุณหภูมิจริง
    - none: ไม่มีข้อมูล thermal ใช้งานได้
    """
    raw_thermal_payload, extraction_error = _extract_binary_tag(image_path, "RawThermalImage")
    if extraction_error:
        return None, extraction_error, "none"

    if raw_thermal_payload:
        raw_thermal_matrix = _parse_thermal_from_tiff(raw_thermal_payload)
        if raw_thermal_matrix is not None:
            return raw_thermal_matrix, None, "absolute"

        raw_thermal_width, raw_thermal_height = _get_raw_thermal_size(image_path)
        raw_thermal_matrix = _decode_u16_payload(
            raw_thermal_payload,
            raw_thermal_width,
            raw_thermal_height,
        )
        if raw_thermal_matrix is not None:
            return raw_thermal_matrix, None, "absolute"

        return None, "RawThermalImage payload cannot be decoded.", "none"

    thermal_data_payload, extraction_error = _extract_binary_tag(image_path, "ThermalData")
    if extraction_error:
        return None, extraction_error, "none"
    if not thermal_data_payload:
        return None, "RawThermalImage/ThermalData tag not found in this file.", "none"

    dji_temperature_matrix, dji_measure_error = _extract_dji_temperature_matrix(
        image_path,
        expected_width,
        expected_height,
    )
    if dji_temperature_matrix is not None:
        return dji_temperature_matrix, None, "absolute"

    thermal_data_matrix = _parse_thermal_from_tiff(thermal_data_payload)
    if thermal_data_matrix is None:
        thermal_data_matrix = _decode_u16_payload(thermal_data_payload, None, None)

    if thermal_data_matrix is None:
        return None, "ThermalData payload cannot be decoded.", "none"

    return (
        thermal_data_matrix,
        "ThermalData found, but absolute temperature is unavailable for this file format. "
        f"Showing relative hotspot points only. SDK detail: {dji_measure_error or 'not available'}",
        "relative",
    )


# ------------------------------
# หนีบ bbox ให้อยู่ในภาพและกว้าง/สูงอย่างน้อย 1 px
# เหมือนเดิม
# ------------------------------
def _safe_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int):
    """
    ปรับกรอบสี่เหลี่ยม (bbox) ให้อยู่ในขอบภาพเสมอ
    และบังคับให้กว้าง/สูงอย่างน้อย 1 พิกเซล เพื่อกันปัญหา slicing ว่าง
    """
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return x1, y1, x2, y2


# ------------------------------
# [ตัวช่วยจัดรูปแบบนามสกุลไฟล์]
# normalize นามสกุลไฟล์ upload ให้ปลอดภัย
# ถ้าไม่ใช่นามสกุลที่อนุญาต จะ fallback เป็น .jpg
# ------------------------------
def _normalize_upload_suffix(filename: str) -> str:
    """
    อนุญาตเฉพาะนามสกุลภาพที่ระบบรองรับ
    ถ้าชื่อไฟล์ไม่ชัดเจนหรือไม่อยู่ในรายการ จะใช้ .jpg แทน
    """
    file_suffix = Path(filename or "").suffix.lower()
    if file_suffix in ALLOWED_IMAGE_SUFFIXES:
        return file_suffix
    return ".jpg"


# ------------------------------
# [ตัวช่วยบันทึกไฟล์อัปโหลด]
# บันทึก bytes เป็นไฟล์จริงใน uploads
# ตั้งชื่อไฟล์เป็น {file_id}_{label}.{ext}
# เช่น abc123_thermal.jpg, abc123_rgb.jpg
# ------------------------------
def _save_upload_bytes(filename: str, file_id: str, label: str, payload: bytes) -> Tuple[str, Path]:
    """
    บันทึกไฟล์ลงโฟลเดอร์ uploads ด้วยชื่อมาตรฐาน:
    {file_id}_{label}.{ext}
    เพื่อให้ backend หาไฟล์ thermal/rgb ของงานเดียวกันได้ง่าย
    """
    file_suffix = _normalize_upload_suffix(filename)
    upload_filename = f"{file_id}_{label}{file_suffix}"
    upload_path = UPLOAD_DIR / upload_filename
    with upload_path.open("wb") as uploaded_file:
        uploaded_file.write(payload)
    return upload_filename, upload_path


# ------------------------------
# [ตัวช่วยค้นหาไฟล์อัปโหลดเดิม]
# ค้นหาไฟล์ที่อัปโหลดไว้แล้วจาก file_id + label
# ใช้กับ flow /upload-file + /analyze
# ------------------------------
def _find_uploaded_file(file_id: str, label: str) -> Optional[Tuple[str, Path]]:
    """
    ค้นหาไฟล์ที่เคยอัปโหลดไว้จาก file_id + ชนิดไฟล์ (thermal/rgb)
    ใช้ใน flow ที่อัปโหลดแยก endpoint ก่อน แล้วค่อยเรียกวิเคราะห์ทีหลัง
    """
    for suffix in ALLOWED_IMAGE_SUFFIXES:
        candidate = UPLOAD_DIR / f"{file_id}_{label}{suffix}"
        if candidate.exists():
            return candidate.name, candidate
    return None


# ------------------------------
# [ตัวช่วยส่ง error กลับ frontend]
# ฟังก์ชันกลางสำหรับคืน JSON error มาตรฐาน
# ------------------------------
def _json_error(message: str, request_id: str, status_code: int, **extra: Any) -> JSONResponse:
    """
    สร้างโครง error response ให้รูปแบบเดียวกันทั้งระบบ
    ช่วยให้ frontend จัดการ error ได้ง่ายและคงที่
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "request_id": request_id,
            **extra,
        },
    )


# ------------------------------
# [ฟังก์ชันรัน YOLO หนึ่งครั้ง]
# รัน YOLO บนภาพ 1 ภาพ แล้วคืนผลลัพธ์เป็น list ของ detection dict
# ใช้ได้ทั้ง hotspot model และ equipment model
# ------------------------------
def _describe_cuda_device() -> str:
    """
    คืนชื่อ GPU ที่ PyTorch เห็น เพื่อให้ log บอกได้ชัดว่ามี CUDA จริงไหม
    """
    if not torch.cuda.is_available():
        return "none"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda_available_unknown_name"


def _describe_yolo_model_device(model: YOLO) -> str:
    """
    อ่าน device จริงจากตัวโมเดลหลัง YOLO เตรียม inference แล้ว
    ใช้ช่วยแยกว่าโมเดลกำลังอยู่บน cpu หรือ cuda
    """
    inner_model = getattr(model, "model", None)
    parameters = getattr(inner_model, "parameters", None)
    if not callable(parameters):
        return "unknown"

    try:
        return str(next(parameters()).device)
    except StopIteration:
        return "unknown"
    except Exception:
        return "unknown"


def _describe_yolo_result_device(model_results: list[Any]) -> str:
    """
    อ่าน device จาก tensor ผลลัพธ์ YOLO ถ้ามี detection กลับมา
    ถ้าไม่มี box จะคืน unknown เพราะไม่มี tensor ให้ดู
    """
    if not model_results:
        return "unknown"

    boxes = getattr(model_results[0], "boxes", None)
    data = getattr(boxes, "data", None)
    device = getattr(data, "device", None)
    return str(device) if device is not None else "unknown"


def _device_runtime_label(device_name: str) -> str:
    """
    แปลงชื่อ device ให้เป็นคำอ่านง่ายใน log
    """
    normalized = device_name.strip().lower()
    if normalized.startswith("cuda") or normalized.startswith("mps") or normalized.isdigit():
        return "gpu"
    if normalized.startswith("cpu"):
        return "cpu"
    return "unknown"


def _run_yolo_detection(model: YOLO, image_path: Path, conf: float, iou: float, imgsz: int) -> list[dict[str, Any]]:
    """
    รันโมเดล YOLO กับภาพ 1 ใบ และแปลงผลลัพธ์เป็น dict แบบเรียบง่าย:
    bbox, confidence, class_id
    """
    inference_started_at = time.perf_counter()
    logger.info(
        "yolo_inference_started image=%s requested_device=%s expected_runtime=%s cuda_available=%s cuda_device=%s imgsz=%s conf=%.3f iou=%.3f",
        image_path.name,
        YOLO_DEVICE,
        _device_runtime_label(YOLO_DEVICE),
        torch.cuda.is_available(),
        _describe_cuda_device(),
        max(32, imgsz),
        conf,
        iou,
    )

    model_results = model(
        str(image_path),
        conf=conf,
        iou=iou,
        device=YOLO_DEVICE,
        imgsz=max(32, imgsz),
    )

    detections: list[dict[str, Any]] = []
    if not model_results or model_results[0].boxes is None:
        model_device = _describe_yolo_model_device(model)
        result_device = _describe_yolo_result_device(model_results)
        logger.info(
            "yolo_inference_finished image=%s requested_device=%s model_device=%s result_device=%s actual_runtime=%s detection_count=0 elapsed_seconds=%.2f",
            image_path.name,
            YOLO_DEVICE,
            model_device,
            result_device,
            _device_runtime_label(model_device if model_device != "unknown" else result_device),
            time.perf_counter() - inference_started_at,
        )
        return detections

    result = model_results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes))
    class_ids = (
        result.boxes.cls.cpu().numpy().astype(int)
        if result.boxes.cls is not None
        else np.zeros(len(boxes), dtype=int)
    )

    for index, box in enumerate(boxes):
        detections.append(
            {
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "confidence": float(confidences[index]),
                "class_id": int(class_ids[index]),
            }
        )

    model_device = _describe_yolo_model_device(model)
    result_device = _describe_yolo_result_device(model_results)
    logger.info(
        "yolo_inference_finished image=%s requested_device=%s model_device=%s result_device=%s actual_runtime=%s detection_count=%s elapsed_seconds=%.2f",
        image_path.name,
        YOLO_DEVICE,
        model_device,
        result_device,
        _device_runtime_label(model_device if model_device != "unknown" else result_device),
        len(detections),
        time.perf_counter() - inference_started_at,
    )

    return detections


# ------------------------------
# [ฟังก์ชันรัน YOLO แบบโหลดแล้วปล่อยทันที]
# เวอร์ชันที่รับ model path แล้วโหลด/รัน/ลบ model ภายใน function
# ช่วยลด memory usage เพราะลบ model หลังใช้เสร็จ
# ------------------------------
def _run_yolo_detection_from_path(
    model_path: Path,
    required: bool,
    image_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
) -> list[dict[str, Any]]:
    """
    เวอร์ชันที่รับ path ของโมเดลแทน object:
    - โหลดโมเดล
    - รัน detect
    - ปล่อยหน่วยความจำทันทีหลังใช้งาน
    """
    model = _load_yolo_model(model_path, required=required)
    if model is None:
        return []
    try:
        return _run_yolo_detection(model, image_path, conf, iou, imgsz)
    finally:
        del model
        gc.collect()


# ------------------------------
# [ตัวช่วยคำนวณตำแหน่ง thermal บน RGB]
# คำนวณว่า thermal image ทั้งภาพ ควรถูก overlay อยู่ตรงไหนใน RGB
# โดยอาศัยฟังก์ชัน project bbox จาก thermal -> RGB
# ------------------------------
def _thermal_overlay_bbox_on_rgb(
    thermal_width: int,
    thermal_height: int,
    rgb_width: int,
    rgb_height: int,
) -> Tuple[int, int, int, int]:
    """
    คำนวณกรอบตำแหน่งโดยประมาณของภาพ thermal บนภาพ RGB ทั้งภาพ
    ใช้เป็นฐานในการ crop พื้นที่ที่น่าจะมีอุปกรณ์อยู่
    """
    return _project_thermal_bbox_to_rgb(
        (0, 0, thermal_width, thermal_height),
        thermal_width,
        thermal_height,
        rgb_width,
        rgb_height,
    )


# ------------------------------
# [ตัวช่วยเตรียมภาพ RGB ก่อน detect]
# crop ภาพ RGB เฉพาะบริเวณที่เกี่ยวข้อง แล้ว resize ถ้าภาพใหญ่เกิน
# จุดประสงค์:
# - ลดพื้นที่ detection
# - เร็วขึ้น
# - ลด false positive
# ------------------------------
def _prepare_cropped_resized_inference_image(
    image_path: Path,
    file_id: str,
    label: str,
    crop_bbox: Tuple[int, int, int, int],
    max_dim: int,
) -> Tuple[Path, float, float, int, int, Optional[Path], Tuple[int, int], Tuple[int, int], Tuple[int, int, int, int]]:
    """
    เตรียมภาพสำหรับ detect:
    1) crop เฉพาะบริเวณที่สนใจ
    2) resize ถ้าใหญ่เกิน max_dim
    3) save เป็นไฟล์ชั่วคราวเพื่อส่งเข้า YOLO

    พร้อมคืน scale/offset เพื่อ map bbox กลับพิกัดภาพจริงภายหลัง
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    with Image.open(image_path) as source_image:
        cropped_image = source_image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("RGB")

    crop_width, crop_height = cropped_image.size
    if max_dim > 0 and max(crop_width, crop_height) > max_dim:
        scale = max_dim / float(max(crop_width, crop_height))
        resized_width = max(1, int(round(crop_width * scale)))
        resized_height = max(1, int(round(crop_height * scale)))
        detection_image = cropped_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        cropped_image.close()
    else:
        detection_image = cropped_image
        resized_width = crop_width
        resized_height = crop_height

    detection_path = UPLOAD_DIR / f"{file_id}_{label}_detect.jpg"
    detection_image.save(detection_path, format="JPEG", quality=90)
    detection_image.close()
    return (
        detection_path,
        crop_width / float(resized_width),
        crop_height / float(resized_height),
        crop_x1,
        crop_y1,
        detection_path,
        (crop_width, crop_height),
        (resized_width, resized_height),
        crop_bbox,
    )


# ------------------------------
# [ตัวช่วย align thermal กับ RGB]
# คำนวณ offset ของ thermal เมื่อวางไว้กลางภาพ RGB
# และบวก shift เพิ่มจาก env เพื่อ fine-tune alignment
# ------------------------------
def _centered_thermal_offset(
    rgb_width: int,
    rgb_height: int,
    thermal_width: int,
    thermal_height: int,
) -> Tuple[int, int]:
    """
    คำนวณระยะเลื่อน (offset) ของ thermal เมื่อวางให้อยู่กลาง RGB
    และเพิ่มค่า shift จาก env เพื่อปรับ alignment แบบ fine-tune
    """
    offset_x = int(np.floor((rgb_width - thermal_width) / 2.0)) + THERMAL_CENTER_SHIFT_X
    offset_y = int(np.floor((rgb_height - thermal_height) / 2.0)) + THERMAL_CENTER_SHIFT_Y
    return offset_x, offset_y


# ------------------------------
# [ตัวช่วยแปลงจุด thermal ไป RGB]
# project จุดจาก thermal -> RGB
# ใช้ตอนเอา hotspot center ไป match กับ equipment
# ------------------------------
def _project_thermal_point_to_rgb(
    point_x: float,
    point_y: float,
    thermal_width: int,
    thermal_height: int,
    rgb_width: int,
    rgb_height: int,
) -> Tuple[int, int]:
    """
    แปลงตำแหน่งจุดจากพิกัด thermal ไปเป็นพิกัด RGB
    จากนั้นหนีบให้อยู่ในขอบภาพ RGB เสมอ
    """
    offset_x, offset_y = _centered_thermal_offset(rgb_width, rgb_height, thermal_width, thermal_height)
    rgb_point_x = int(round(point_x + offset_x))
    rgb_point_y = int(round(point_y + offset_y))
    rgb_point_x = max(0, min(rgb_point_x, rgb_width - 1))
    rgb_point_y = max(0, min(rgb_point_y, rgb_height - 1))
    return rgb_point_x, rgb_point_y


# ------------------------------
# [ตัวช่วยแปลงกรอบ thermal ไป RGB]
# project bbox จาก thermal -> RGB
# ------------------------------
def _project_thermal_bbox_to_rgb(
    bbox: Tuple[int, int, int, int],
    thermal_width: int,
    thermal_height: int,
    rgb_width: int,
    rgb_height: int,
) -> Tuple[int, int, int, int]:
    """
    แปลงกรอบ bbox จาก thermal ไป RGB โดยแปลงทีละมุม
    แล้วใช้ _safe_bbox ป้องกันกรอบเลยขอบภาพ
    """
    x1, y1 = _project_thermal_point_to_rgb(
        bbox[0],
        bbox[1],
        thermal_width,
        thermal_height,
        rgb_width,
        rgb_height,
    )
    x2, y2 = _project_thermal_point_to_rgb(
        bbox[2],
        bbox[3],
        thermal_width,
        thermal_height,
        rgb_width,
        rgb_height,
    )
    return _safe_bbox(x1, y1, x2, y2, rgb_width, rgb_height)


# ------------------------------
# [ตัวช่วยขยายกรอบ bbox]
# ขยาย bbox ออกทุกด้านตาม dilation ที่กำหนด
# ใช้ช่วยการ match hotspot กับ equipment
# ------------------------------
def _dilate_bbox(
    bbox: Tuple[int, int, int, int],
    dilation: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    ขยายกรอบ bbox รอบด้านตามค่า dilation
    ใช้เผื่อ margin เวลาจับคู่ hotspot กับอุปกรณ์ที่อาจเยื้องเล็กน้อย
    """
    return _safe_bbox(
        bbox[0] - dilation,
        bbox[1] - dilation,
        bbox[2] + dilation,
        bbox[3] + dilation,
        image_width,
        image_height,
    )


# ------------------------------
# [ตัวช่วยตรวจว่าจุดอยู่ในกรอบหรือไม่]
# เช็คว่าจุดอยู่ใน bbox หรือไม่
# ------------------------------
def _bbox_contains_point(bbox: Tuple[int, int, int, int], point_x: int, point_y: int) -> bool:
    """
    ตรวจว่าจุด (x, y) อยู่ภายในกรอบ bbox หรือไม่
    """
    return bbox[0] <= point_x <= bbox[2] and bbox[1] <= point_y <= bbox[3]


# ------------------------------
# [ตัวช่วยคำนวณระยะจากจุดถึงกรอบ]
# คำนวณระยะจากจุดไปยัง bbox
# ถ้าจุดอยู่ใน bbox ระยะจะเป็น 0
# ------------------------------
def _distance_to_bbox(bbox: Tuple[int, int, int, int], point_x: int, point_y: int) -> float:
    """
    คำนวณระยะทางสั้นที่สุดจากจุดไปยังกรอบ bbox
    ถ้าจุดอยู่ในกรอบ ระยะจะเป็น 0
    """
    delta_x = max(bbox[0] - point_x, 0, point_x - bbox[2])
    delta_y = max(bbox[1] - point_y, 0, point_y - bbox[3])
    return float(np.hypot(delta_x, delta_y))


# ------------------------------
# [ตัวช่วยวาดหมายเลข hotspot]
# วาดเลข hotspot ลงบนกรอบในภาพ thermal
# ------------------------------
def _draw_hotspot_index_label(
    draw: Any,
    thermal_box: Tuple[int, int, int, int],
    hotspot_index: int,
    thermal_image_width: int,
    thermal_image_height: int,
) -> None:
    """
    วาด label เช่น #1, #2, #3 ไว้บนกรอบ hotspot เพื่อให้เทียบกับรายการด้านล่างได้ง่าย
    """
    label_text = f"#{hotspot_index}"
    padding_x = 5
    padding_y = 2

    try:
        text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), label_text)
        text_width = int(text_right - text_left)
        text_height = int(text_bottom - text_top)
    except Exception:
        if hasattr(draw, "textsize"):
            text_width, text_height = draw.textsize(label_text)
        else:
            text_width = max(10, 6 * len(label_text))
            text_height = 11

    label_width = int(text_width + (padding_x * 2))
    label_height = int(text_height + (padding_y * 2))

    label_x1 = int(thermal_box[0] + 2)
    label_y1 = int(thermal_box[1] + 2)
    label_x1 = max(0, min(label_x1, max(0, thermal_image_width - label_width)))
    label_y1 = max(0, min(label_y1, max(0, thermal_image_height - label_height)))
    label_x2 = min(thermal_image_width, label_x1 + label_width)
    label_y2 = min(thermal_image_height, label_y1 + label_height)

    draw.rectangle((label_x1, label_y1, label_x2, label_y2), fill="orange")
    draw.text((label_x1 + padding_x, label_y1 + padding_y), label_text, fill="black")


# ------------------------------
# [ตัวช่วยแปลง class id เป็นชื่ออุปกรณ์]
# แปลง class id -> ชื่ออุปกรณ์
# ------------------------------
def _equipment_label_for_class(class_id: int) -> str:
    """
    แปลง class_id จากโมเดลเป็นชื่ออุปกรณ์ที่มนุษย์อ่านง่าย
    """
    return EQUIPMENT_LABELS.get(class_id, f"class_{class_id}")


# ------------------------------
# [ฟังก์ชันจับคู่ hotspot กับอุปกรณ์]
# ฟังก์ชัน match hotspot กับ equipment
#
# หลักการ:
# 1) ถ้าศูนย์กลาง hotspot อยู่ใน bbox ของ equipment (หลังขยาย) -> inside
# 2) ถ้าไม่อยู่ แต่ใกล้พอ -> nearest
# 3) ถ้าไกลเกิน threshold -> unknown
# ------------------------------
def _match_equipment(
    hotspot_center: Tuple[int, int],
    equipments: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """
    จับคู่ hotspot หนึ่งจุดกับอุปกรณ์ที่เหมาะที่สุดบนภาพ RGB
    เงื่อนไข:
    - ถ้า hotspot อยู่ในกรอบอุปกรณ์ (หลังขยาย) => inside
    - ถ้าไม่อยู่ แต่ใกล้พอ => nearest
    - ถ้าไกลเกิน threshold => unknown
    """
    if not equipments:
        return {
            "equipment_class": "unknown",
            "equipment_confidence": None,
            "equipment_bbox": None,
            "match_method": "unknown",
            "match_distance": None,
        }

    scored_candidates = []
    for equipment in equipments:
        bbox = equipment["bbox"]
        dilated_bbox = _dilate_bbox(bbox, EQUIPMENT_BBOX_DILATION, image_width, image_height)
        contains_center = _bbox_contains_point(dilated_bbox, hotspot_center[0], hotspot_center[1])
        distance = _distance_to_bbox(dilated_bbox, hotspot_center[0], hotspot_center[1])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        scored_candidates.append(
            (
                0 if contains_center else 1,
                distance,
                -float(equipment["confidence"]),
                area,
                equipment,
            )
        )

    scored_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, distance, _, _, best_equipment = scored_candidates[0]

    if _bbox_contains_point(
        _dilate_bbox(best_equipment["bbox"], EQUIPMENT_BBOX_DILATION, image_width, image_height),
        hotspot_center[0],
        hotspot_center[1],
    ):
        match_method = "inside"
    elif distance <= MATCH_DISTANCE_THRESHOLD:
        match_method = "nearest"
    else:
        return {
            "equipment_class": "unknown",
            "equipment_confidence": None,
            "equipment_bbox": None,
            "match_method": "unknown",
            "match_distance": round(distance, 2),
        }

    return {
        "equipment_class": best_equipment["label"],
        "equipment_confidence": best_equipment["confidence"],
        "equipment_bbox": list(best_equipment["bbox"]),
        "match_method": match_method,
        "match_distance": round(distance, 2),
    }


# ------------------------------
# [ส่วนคำนวณค่าอ้างอิง]
# ฟังก์ชันนี้ใช้หาค่าอุณหภูมิอ้างอิงของทั้งภาพ
# แล้วใช้ค่านั้นเป็นฐานในการบอกว่าจุดร้อนสูงกว่าพื้นหลังเท่าไร
# ------------------------------
def _compute_reference_temperature(thermal_matrix: np.ndarray) -> Optional[float]:
    """
    คำนวณอุณหภูมิอ้างอิงของภาพจากพิกเซลที่ไม่ร้อนเกินเกณฑ์
    ใช้เป็น baseline สำหรับหาว่าจุดร้อนสูงกว่าพื้นหลังเท่าไร
    """
    finite_values = thermal_matrix[np.isfinite(thermal_matrix)]
    if finite_values.size == 0:
        return None
    reference_pixels = finite_values[finite_values <= REFERENCE_TEMP_MAX_C]
    if reference_pixels.size == 0:
        return None
    return float(reference_pixels.mean())


# ------------------------------
# [เพิ่มในเวอร์ชันล่าสุด]
# กลุ่ม helper ด้านล่างนี้เพิ่มเข้ามาเพื่อรองรับการเลือก ROI จากหน้าเว็บ
# หน้าที่หลักคือ:
# 1) แปลงกรอบที่ผู้ใช้เลือกบนภาพ -> พิกัดใน thermal matrix
# 2) คำนวณสถิติในกรอบที่เลือก
# 3) คำนวณ reference temperature ใหม่จาก ROI
# 4) recalculation ค่าของ hotspot เดิมโดยไม่ต้องรันโมเดลใหม่
# ------------------------------
def _image_bbox_to_matrix_bounds(
    image_box: Tuple[int, int, int, int],
    matrix_width: int,
    matrix_height: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    แปลงกรอบในพิกัดภาพต้นฉบับ -> พิกัด matrix ที่ใช้วิเคราะห์จริง
    """
    x1 = int(np.floor(image_box[0] * matrix_width / image_width))
    x2 = int(np.ceil(image_box[2] * matrix_width / image_width))
    y1 = int(np.floor(image_box[1] * matrix_height / image_height))
    y2 = int(np.ceil(image_box[3] * matrix_height / image_height))

    x1 = max(0, min(x1, matrix_width - 1))
    y1 = max(0, min(y1, matrix_height - 1))
    x2 = max(x1 + 1, min(x2, matrix_width))
    y2 = max(y1 + 1, min(y2, matrix_height))
    return x1, y1, x2, y2


def _extract_region_statistics(
    thermal_analysis_matrix: np.ndarray,
    image_box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Optional[dict[str, Any]]:
    """
    คำนวณ max/min/avg และตำแหน่งจุด max/min จากกรอบที่อ้างอิงพิกัดภาพ thermal
    """
    matrix_height, matrix_width = thermal_analysis_matrix.shape
    matrix_x1, matrix_y1, matrix_x2, matrix_y2 = _image_bbox_to_matrix_bounds(
        image_box,
        matrix_width,
        matrix_height,
        image_width,
        image_height,
    )

    thermal_region = thermal_analysis_matrix[matrix_y1:matrix_y2, matrix_x1:matrix_x2]
    finite_region = thermal_region[np.isfinite(thermal_region)]
    if finite_region.size == 0:
        return None

    max_value = float(np.nanmax(thermal_region))
    min_value = float(np.nanmin(thermal_region))
    avg_value = float(np.nanmean(thermal_region))

    max_position = np.unravel_index(int(np.nanargmax(thermal_region)), thermal_region.shape)
    min_position = np.unravel_index(int(np.nanargmin(thermal_region)), thermal_region.shape)

    max_point_x = int((matrix_x1 + max_position[1]) * image_width / matrix_width)
    max_point_y = int((matrix_y1 + max_position[0]) * image_height / matrix_height)
    min_point_x = int((matrix_x1 + min_position[1]) * image_width / matrix_width)
    min_point_y = int((matrix_y1 + min_position[0]) * image_height / matrix_height)

    return {
        "max_value": max_value,
        "min_value": min_value,
        "avg_value": avg_value,
        "max_point": [max_point_x, max_point_y],
        "min_point": [min_point_x, min_point_y],
    }


def _compute_reference_temperature_from_roi(
    thermal_analysis_matrix: np.ndarray,
    roi_box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Optional[float]:
    """
    คำนวณ reference temperature จาก ROI ที่ผู้ใช้ลากเอง
    ใช้ค่าเฉลี่ยของ finite pixels ในกรอบโดยไม่กรอง threshold แบบทั้งภาพ
    """
    matrix_height, matrix_width = thermal_analysis_matrix.shape
    matrix_x1, matrix_y1, matrix_x2, matrix_y2 = _image_bbox_to_matrix_bounds(
        roi_box,
        matrix_width,
        matrix_height,
        image_width,
        image_height,
    )
    roi_region = thermal_analysis_matrix[matrix_y1:matrix_y2, matrix_x1:matrix_x2]
    finite_pixels = roi_region[np.isfinite(roi_region)]
    if finite_pixels.size == 0:
        return None
    return float(finite_pixels.mean())


def _render_fixed_range_thermal_image(
    thermal_matrix: Optional[np.ndarray],
    image_width: int,
    image_height: int,
    display_min_c: Optional[float] = None,
    display_max_c: Optional[float] = None,
) -> Optional[Image.Image]:
    """
    render thermal matrix ให้เป็นภาพสีด้วยช่วงอุณหภูมิคงที่
    ใช้เฉพาะสำหรับการแสดงผลเพื่อให้เทียบหลายภาพได้ตรงกัน
    """
    if thermal_matrix is None:
        return None

    finite_mask = np.isfinite(thermal_matrix)
    if not np.any(finite_mask):
        return None

    min_c = DISPLAY_TEMP_MIN_C if display_min_c is None else display_min_c
    max_c = DISPLAY_TEMP_MAX_C if display_max_c is None else display_max_c
    display_range = max_c - min_c
    if display_range <= 0:
        return None

    normalized_matrix = np.clip((thermal_matrix - min_c) / display_range, 0.0, 1.0).astype(np.float32)
    rgb_matrix = np.zeros((*thermal_matrix.shape, 3), dtype=np.uint8)

    stop_positions = THERMAL_DISPLAY_COLOR_STOPS[:, 0]
    stop_red = THERMAL_DISPLAY_COLOR_STOPS[:, 1]
    stop_green = THERMAL_DISPLAY_COLOR_STOPS[:, 2]
    stop_blue = THERMAL_DISPLAY_COLOR_STOPS[:, 3]

    flat_values = normalized_matrix[finite_mask]
    interpolated_colors = np.stack(
        [
            np.interp(flat_values, stop_positions, stop_red),
            np.interp(flat_values, stop_positions, stop_green),
            np.interp(flat_values, stop_positions, stop_blue),
        ],
        axis=-1,
    ).astype(np.uint8)
    rgb_matrix[finite_mask] = interpolated_colors

    rendered_image = Image.fromarray(rgb_matrix, mode="RGB")
    if rendered_image.size != (image_width, image_height):
        rendered_image = rendered_image.resize((image_width, image_height), resample=Image.Resampling.BILINEAR)
    return rendered_image


def _get_thermal_matrix_temperature_range(thermal_matrix: Optional[np.ndarray]) -> tuple[Optional[float], Optional[float]]:
    """
    หาช่วงอุณหภูมิจริงของภาพ thermal จาก matrix
    ใช้เพื่อบอกผู้ใช้ว่าภาพเดิมตอนนี้มี min/max ประมาณเท่าไหร่
    """
    if thermal_matrix is None:
        return None, None

    finite_values = thermal_matrix[np.isfinite(thermal_matrix)]
    if finite_values.size == 0:
        return None, None

    return float(np.nanmin(finite_values)), float(np.nanmax(finite_values))


def _draw_detection_annotations_on_thermal(
    thermal_image: Image.Image,
    detections: list[dict[str, Any]],
    thermal_image_width: int,
    thermal_image_height: int,
) -> None:
    """
    วาดกรอบ hotspot เดิมซ้ำบนภาพ thermal ที่ render ใหม่
    ใช้ตอนผู้ใช้เปลี่ยน display range โดยไม่ต้อง rerun model
    """
    thermal_draw = ImageDraw.Draw(thermal_image)

    for hotspot_index, detection in enumerate(detections, start=1):
        raw_box = detection.get("thermal_bbox")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raw_box = detection.get("bbox")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue

        try:
            thermal_box = tuple(
                _safe_bbox(
                    int(round(float(raw_box[0]))),
                    int(round(float(raw_box[1]))),
                    int(round(float(raw_box[2]))),
                    int(round(float(raw_box[3]))),
                    thermal_image_width,
                    thermal_image_height,
                )
            )
        except (TypeError, ValueError):
            continue

        thermal_draw.rectangle(thermal_box, outline="orange", width=3)
        _draw_hotspot_index_label(
            draw=thermal_draw,
            thermal_box=thermal_box,
            hotspot_index=hotspot_index,
            thermal_image_width=thermal_image_width,
            thermal_image_height=thermal_image_height,
        )

        for point_key, point_color in (("max_point", "red"), ("min_point", "blue")):
            raw_point = detection.get(point_key)
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                continue
            try:
                point_x = int(round(float(raw_point[0])))
                point_y = int(round(float(raw_point[1])))
            except (TypeError, ValueError):
                continue
            thermal_draw.ellipse([point_x - 4, point_y - 4, point_x + 4, point_y + 4], fill=point_color)

        max_temp = detection.get("max_temp")
        min_temp = detection.get("min_temp")
        avg_temp = detection.get("avg_temp")
        if isinstance(max_temp, (int, float)) and isinstance(min_temp, (int, float)) and isinstance(avg_temp, (int, float)):
            thermal_draw.text(
                (thermal_box[0], max(0, thermal_box[1] - 15)),
                f"max {float(max_temp):.1f}C min {float(min_temp):.1f}C avg {float(avg_temp):.1f}C",
                fill="white",
            )


def _draw_debug_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label_text: str, color: str) -> None:
    """
    วาด label เล็ก ๆ บนภาพ debug ของโมเดล
    ใช้เพื่อดูว่าโมเดล detect อะไรได้บ้างก่อนเอาไป match กัน
    """
    padding_x = 6
    padding_y = 4
    try:
        text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), label_text)
        text_width = text_right - text_left
        text_height = text_bottom - text_top
    except AttributeError:
        text_width, text_height = draw.textsize(label_text)

    label_x1 = box[0]
    label_y1 = max(0, box[1] - text_height - padding_y * 2)
    label_x2 = label_x1 + text_width + padding_x * 2
    label_y2 = label_y1 + text_height + padding_y * 2

    draw.rounded_rectangle([label_x1, label_y1, label_x2, label_y2], radius=8, fill=color)
    draw.text((label_x1 + padding_x, label_y1 + padding_y), label_text, fill="white")


def _draw_hotspot_model_debug_image(
    image: Image.Image,
    hotspot_predictions: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> None:
    """
    วาดผลจาก hotspot model อย่างเดียวบนภาพ thermal
    ภาพนี้ตั้งใจให้ใช้ตรวจว่า AI เจอ hotspot ตรงไหน ก่อนขั้นจับคู่กับอุปกรณ์
    """
    draw = ImageDraw.Draw(image)
    for hotspot_index, hotspot_prediction in enumerate(hotspot_predictions, start=1):
        try:
            raw_box = hotspot_prediction["bbox"]
            box = tuple(
                _safe_bbox(
                    int(round(float(raw_box[0]))),
                    int(round(float(raw_box[1]))),
                    int(round(float(raw_box[2]))),
                    int(round(float(raw_box[3]))),
                    image_width,
                    image_height,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

        confidence = float(hotspot_prediction.get("confidence", 0.0))
        draw.rectangle(box, outline="#ffb02e", width=4)
        _draw_debug_label(draw, box, f"Hotspot {hotspot_index} {confidence:.2f}", "#1f1b16")


def _draw_equipment_model_debug_image(
    image: Image.Image,
    equipments: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    source_offset_x: int = 0,
    source_offset_y: int = 0,
) -> None:
    """
    วาดผลจาก equipment model อย่างเดียวบนภาพ RGB
    ภาพนี้ช่วยดูว่าโมเดล RGB ตรวจเจออุปกรณ์อะไรบ้าง ก่อนเอา hotspot ไป match
    """
    draw = ImageDraw.Draw(image)
    line_width = max(6, int(round(min(image_width, image_height) * 0.008)))
    for equipment_index, equipment in enumerate(equipments, start=1):
        try:
            raw_box = equipment["bbox"]
            x1 = int(round(float(raw_box[0]) - source_offset_x))
            y1 = int(round(float(raw_box[1]) - source_offset_y))
            x2 = int(round(float(raw_box[2]) - source_offset_x))
            y2 = int(round(float(raw_box[3]) - source_offset_y))
            if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
                continue
            box = tuple(
                _safe_bbox(
                    x1,
                    y1,
                    x2,
                    y2,
                    image_width,
                    image_height,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

        label = str(equipment.get("label") or f"Equipment {equipment_index}")
        confidence = float(equipment.get("confidence", 0.0))
        draw.rectangle(box, outline="#2f5d96", width=line_width)
        _draw_debug_label(draw, box, f"{label} {confidence:.2f}", "#2f5d96")


def _draw_projected_hotspots_on_rgb_debug_image(
    image: Image.Image,
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    scale_x: float,
    scale_y: float,
    crop_offset_x: int,
    crop_offset_y: int,
) -> None:
    """
    วาดกรอบ hotspot ที่ project ไปบนภาพ RGB crop/resize
    ภาพนี้เป็นภาพเดียวกับที่ส่งเข้า equipment model จึงต้องแปลงพิกัด RGB เต็มใบกลับเป็นพิกัดภาพ detect
    """
    draw = ImageDraw.Draw(image)
    line_width = max(6, int(round(min(image_width, image_height) * 0.008)))
    point_radius = max(4, int(round(min(image_width, image_height) * 0.004)))
    for hotspot_index, detection in enumerate(detections, start=1):
        try:
            raw_box = detection["bbox"]
            x1 = int(round((float(raw_box[0]) - crop_offset_x) / scale_x))
            y1 = int(round((float(raw_box[1]) - crop_offset_y) / scale_y))
            x2 = int(round((float(raw_box[2]) - crop_offset_x) / scale_x))
            y2 = int(round((float(raw_box[3]) - crop_offset_y) / scale_y))
            if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
                continue
            box = tuple(_safe_bbox(x1, y1, x2, y2, image_width, image_height))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue

        draw.rectangle(box, outline="#ffb02e", width=line_width)
        _draw_hotspot_index_label(
            draw=draw,
            thermal_box=box,
            hotspot_index=hotspot_index,
            thermal_image_width=image_width,
            thermal_image_height=image_height,
        )

        # วาดจุด max/min และข้อความอุณหภูมิแบบเดียวกับภาพ thermal
        # แต่แปลงพิกัด thermal -> RGB crop/resize ก่อน เพื่อให้ตำแหน่งตรงกับภาพที่แสดง
        raw_thermal_box = detection.get("thermal_bbox")
        if isinstance(raw_thermal_box, list) and len(raw_thermal_box) == 4:
            for point_key, point_color in (("max_point", "red"), ("min_point", "blue")):
                raw_point = detection.get(point_key)
                if not isinstance(raw_point, list) or len(raw_point) != 2:
                    continue
                try:
                    projected_x = float(raw_point[0]) + float(raw_box[0]) - float(raw_thermal_box[0])
                    projected_y = float(raw_point[1]) + float(raw_box[1]) - float(raw_thermal_box[1])
                    point_x = int(round((projected_x - crop_offset_x) / scale_x))
                    point_y = int(round((projected_y - crop_offset_y) / scale_y))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if point_x < 0 or point_y < 0 or point_x >= image_width or point_y >= image_height:
                    continue
                draw.ellipse(
                    [
                        point_x - point_radius,
                        point_y - point_radius,
                        point_x + point_radius,
                        point_y + point_radius,
                    ],
                    fill=point_color,
                )

        max_temp = detection.get("max_temp")
        min_temp = detection.get("min_temp")
        avg_temp = detection.get("avg_temp")
        if isinstance(max_temp, (int, float)) and isinstance(min_temp, (int, float)) and isinstance(avg_temp, (int, float)):
            draw.text(
                (box[0], max(0, box[1] - 18)),
                f"max {float(max_temp):.1f}C min {float(min_temp):.1f}C avg {float(avg_temp):.1f}C",
                fill="white",
                stroke_width=2,
                stroke_fill="#1f1b16",
            )


def _draw_rgb_crop_context_debug_image(
    image: Image.Image,
    crop_bbox: tuple[int, int, int, int],
    equipments: list[dict[str, Any]],
    original_width: int,
    original_height: int,
) -> None:
    """
    วาดภาพ RGB เต็มใบสำหรับ debug ว่าระบบ crop ส่วนไหนไปเข้า equipment model
    ภาพนี้ไม่ใช่ภาพเข้าโมเดลโดยตรง แต่ใช้เป็นแผนที่อธิบาย crop area ให้ดูง่ายขึ้น
    """
    preview_width, preview_height = image.size
    scale_x = preview_width / float(original_width)
    scale_y = preview_height / float(original_height)

    def scale_box(raw_box: tuple[int, int, int, int] | list[float]) -> tuple[int, int, int, int]:
        return _safe_bbox(
            int(round(float(raw_box[0]) * scale_x)),
            int(round(float(raw_box[1]) * scale_y)),
            int(round(float(raw_box[2]) * scale_x)),
            int(round(float(raw_box[3]) * scale_y)),
            preview_width,
            preview_height,
        )

    draw = ImageDraw.Draw(image)
    crop_box = scale_box(crop_bbox)
    crop_line_width = max(5, int(round(min(preview_width, preview_height) * 0.007)))
    draw.rectangle(crop_box, outline="#ffb02e", width=crop_line_width)
    _draw_debug_label(draw, crop_box, "RGB crop sent to equipment model", "#1f1b16")

    scaled_equipments: list[dict[str, Any]] = []
    for equipment in equipments:
        raw_box = equipment.get("bbox")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        scaled_equipments.append(
            {
                "bbox": list(scale_box(raw_box)),
                "label": equipment.get("label"),
                "confidence": equipment.get("confidence"),
            }
        )

    _draw_equipment_model_debug_image(scaled_equipments and image or image, scaled_equipments, preview_width, preview_height)


def _parse_normalized_roi(payload: Any) -> dict[str, float]:
    """
    รับ ROI แบบ normalized 0..1 จาก frontend และตรวจความถูกต้อง
    """
    if not isinstance(payload, dict):
        raise ValueError("ROI payload must be an object.")

    normalized_roi: dict[str, float] = {}
    for field_name in ("x", "y", "width", "height"):
        raw_value = payload.get(field_name)
        if not isinstance(raw_value, (int, float)):
            raise ValueError(f"ROI field '{field_name}' must be a number.")
        numeric_value = float(raw_value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"ROI field '{field_name}' must be finite.")
        normalized_roi[field_name] = numeric_value

    if normalized_roi["width"] <= 0 or normalized_roi["height"] <= 0:
        raise ValueError("ROI width and height must be greater than zero.")

    if normalized_roi["x"] < 0 or normalized_roi["y"] < 0:
        raise ValueError("ROI must stay inside the image bounds.")

    if normalized_roi["x"] >= 1 or normalized_roi["y"] >= 1:
        raise ValueError("ROI must start inside the image bounds.")

    if normalized_roi["x"] + normalized_roi["width"] > 1 or normalized_roi["y"] + normalized_roi["height"] > 1:
        raise ValueError("ROI must stay inside the image bounds.")

    return normalized_roi


def _normalized_roi_to_image_box(
    normalized_roi: dict[str, float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    แปลง normalized ROI -> bbox ในพิกัดภาพ thermal จริง
    """
    x1 = int(np.floor(normalized_roi["x"] * image_width))
    y1 = int(np.floor(normalized_roi["y"] * image_height))
    x2 = int(np.ceil((normalized_roi["x"] + normalized_roi["width"]) * image_width))
    y2 = int(np.ceil((normalized_roi["y"] + normalized_roi["height"]) * image_height))
    return _safe_bbox(x1, y1, x2, y2, image_width, image_height)


def _coerce_detection_thermal_bbox(
    raw_bbox: Any,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    อ่าน thermal_bbox ของ detection ที่ frontend ส่งกลับมา และหนีบให้อยู่ในภาพ
    """
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("Each detection requires a thermal_bbox with four numbers.")

    coordinates: list[int] = []
    for coordinate in raw_bbox:
        if not isinstance(coordinate, (int, float)):
            raise ValueError("thermal_bbox values must be numbers.")
        numeric_value = float(coordinate)
        if not np.isfinite(numeric_value):
            raise ValueError("thermal_bbox values must be finite.")
        coordinates.append(int(round(numeric_value)))

    return _safe_bbox(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        coordinates[3],
        image_width,
        image_height,
    )


def _recalculate_detections_for_reference_roi(
    detections: list[dict[str, Any]],
    thermal_analysis_matrix: np.ndarray,
    thermal_image_width: int,
    thermal_image_height: int,
    reference_temperature: float,
) -> list[dict[str, Any]]:
    """
    ใช้ hotspot เดิมจาก thermal_bbox แล้วคำนวณ reference/priority/action ใหม่
    โดยไม่ rerun model
    """
    recalculated_detections: list[dict[str, Any]] = []

    for detection in detections:
        thermal_box = _coerce_detection_thermal_bbox(
            detection.get("thermal_bbox"),
            thermal_image_width,
            thermal_image_height,
        )
        next_detection = dict(detection)
        next_detection["thermal_bbox"] = list(thermal_box)
        next_detection["reference_temp"] = reference_temperature
        next_detection["delta_above_reference"] = None
        next_detection["priority"] = None
        next_detection["action_required"] = None

        region_statistics = _extract_region_statistics(
            thermal_analysis_matrix,
            thermal_box,
            thermal_image_width,
            thermal_image_height,
        )

        if region_statistics is not None:
            next_detection["max_temp"] = region_statistics["max_value"]
            next_detection["min_temp"] = region_statistics["min_value"]
            next_detection["avg_temp"] = region_statistics["avg_value"]
            next_detection["max_point"] = region_statistics["max_point"]
            next_detection["min_point"] = region_statistics["min_point"]
            next_detection["max_raw"] = None
            next_detection["min_raw"] = None
            next_detection["avg_raw"] = None

            delta_above_reference = region_statistics["max_value"] - reference_temperature
            priority, action_required = _classify_priority(delta_above_reference)
            next_detection["delta_above_reference"] = delta_above_reference
            next_detection["priority"] = priority
            next_detection["action_required"] = action_required

        recalculated_detections.append(next_detection)

    return recalculated_detections


# ------------------------------
# [ตัวช่วยจัดระดับความเร่งด่วน]
# แปลง delta_above_reference -> priority และ action required
# ------------------------------
def _classify_priority(delta_above_reference: float) -> Tuple[str, str]:
    """
    แปลงค่าอุณหภูมิที่สูงกว่าค่าอ้างอิง (delta) เป็นระดับความเร่งด่วน
    เพื่อสรุป action ที่ทีมซ่อมควรทำต่อ
    """
    if delta_above_reference > 40.0:
        return "Priority 1", "Immediate repair"
    if delta_above_reference >= 21.0:
        return "Priority 2", "Schedule ASAP"
    if delta_above_reference >= 11.0:
        return "Priority 3", "Plan repair"
    if delta_above_reference >= 1.0:
        return "Priority 4", "Monitor"
    return "Normal", "No action required"


# ------------------------------
# [ตัวช่วย log และอัปเดต progress พร้อมกัน]
# helper สำหรับ log step และอัปเดต progress พร้อมกัน
# ------------------------------
def _log_upload_step(request_id: str, step: str, **details: Any) -> None:
    """
    helper รวม 2 งานไว้ด้วยกัน:
    1) อัปเดต request_progress
    2) เขียน log ลงระบบ
    ทำให้จุดเรียกใช้งานในโค้ดสั้นลงและคงรูปแบบเดียวกัน
    """
    _set_request_progress(request_id, step, **details)
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    if detail_text:
        logger.info("[%s] %s %s", request_id, step, detail_text)
    else:
        logger.info("[%s] %s", request_id, step)


# [เพิ่มใหม่]
# helper ชุดนี้ใช้ทำ log สำหรับอ่านบน terminal/render ให้เป็นภาษาคนมากขึ้น
# ไม่ได้เปลี่ยนผลวิเคราะห์ภาพ แค่ช่วยให้เห็นว่า batch ไหนกำลังทำไฟล์อะไรอยู่
def _compact_log_details(details: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in details.items() if value not in (None, "")}


def _safe_positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _read_batch_log_headers(request: Request) -> dict[str, Any]:
    total_files = _safe_positive_int(request.headers.get("x-batch-total-files"))
    total_pairs = _safe_positive_int(request.headers.get("x-batch-total-pairs"))
    file_index = _safe_positive_int(request.headers.get("x-batch-file-index"))
    pair_index = _safe_positive_int(request.headers.get("x-batch-pair-index"))

    details = {
        "batch_id": (request.headers.get("x-batch-id") or "").strip(),
        "total_files": total_files,
        "total_pairs": total_pairs,
        "file_index": file_index,
        "pair_index": pair_index,
        "pair_label": (request.headers.get("x-batch-pair-label") or "").strip(),
    }

    # เพิ่ม progress แบบอ่านง่าย เช่น file_progress=3/8, pair_progress=2/4
    if file_index is not None and total_files is not None:
        details["file_progress"] = f"{file_index}/{total_files}"
    if pair_index is not None and total_pairs is not None:
        details["pair_progress"] = f"{pair_index}/{total_pairs}"

    return _compact_log_details(details)


def _log_readable_event(request_id: str, icon: str, event: str, **details: Any) -> None:
    clean_details = _compact_log_details(details)
    detail_text = " ".join(f"{key}={value}" for key, value in clean_details.items())
    if detail_text:
        logger.info("[%s] %s %s %s", request_id, icon, event, detail_text)
    else:
        logger.info("[%s] %s %s", request_id, icon, event)


# ------------------------------
# [ฟังก์ชันแกนกลางของระบบ]
# ฟังก์ชันแกนกลางของระบบวิเคราะห์ภาพคู่
#
# หน้าที่:
# 1) อ่าน GPS จาก thermal
# 2) เปิดภาพ thermal / rgb
# 3) รัน hotspot model บน thermal
# 4) crop และรัน equipment model บน rgb
# 5) extract thermal matrix
# 6) คำนวณ reference temperature
# 7) match hotspot กับ equipment
# 8) คำนวณ priority
# 9) save annotated image และคืน response
# ------------------------------
def _analyze_saved_pair(
    request_id: str,
    started_at: float,
    file_id: str,
    thermal_uploaded_image_filename: str,
    thermal_uploaded_image_path: Path,
    rgb_uploaded_image_filename: Optional[str] = None,
    rgb_uploaded_image_path: Optional[Path] = None,
    analysis_mode: str = "paired",
):
    """
    แกนหลักของระบบวิเคราะห์ภาพคู่ thermal + RGB
    ภาพรวม:
    1) อ่านตำแหน่ง GPS จากภาพ thermal (ถ้ามี)
    2) ใช้โมเดล thermal หา hotspot
    3) ใช้โมเดล RGB หาอุปกรณ์
    4) ดึงข้อมูลอุณหภูมิจากไฟล์ thermal
    5) จับคู่ hotspot กับอุปกรณ์ที่ใกล้/ครอบอยู่
    6) คำนวณความรุนแรง แล้วส่งผลลัพธ์กลับ
    """
    # อ่าน EXIF/GPS จาก thermal image
    with thermal_uploaded_image_path.open("rb") as thermal_image_stream:
        tags = exifread.process_file(thermal_image_stream)

    lat = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")

    has_gps = bool(lat and lon and lat_ref and lon_ref)

    latitude = None
    longitude = None
    if has_gps:
        latitude = dms_to_decimal(lat, lat_ref.values)
        longitude = dms_to_decimal(lon, lon_ref.values)
    _log_upload_step(request_id, "gps_checked", has_gps=has_gps)

    # อ่านขนาดภาพ thermal
    _log_upload_step(request_id, "thermal_image_probe_started")
    with Image.open(thermal_uploaded_image_path) as thermal_source_image:
        thermal_image_width, thermal_image_height = thermal_source_image.size
    _log_upload_step(
        request_id,
        "thermal_image_probe_finished",
        thermal_size=f"{thermal_image_width}x{thermal_image_height}",
    )

    is_thermal_only = analysis_mode == "thermal_only"

    # อ่านขนาดภาพ RGB เฉพาะงานที่มีคู่ RGB จริง
    rgb_image_width = thermal_image_width
    rgb_image_height = thermal_image_height
    if not is_thermal_only and rgb_uploaded_image_path is not None:
        _log_upload_step(request_id, "rgb_image_probe_started")
        with Image.open(rgb_uploaded_image_path) as rgb_source_image:
            rgb_image_width, rgb_image_height = rgb_source_image.size
        _log_upload_step(
            request_id,
            "rgb_image_probe_finished",
            rgb_size=f"{rgb_image_width}x{rgb_image_height}",
        )
    gc.collect()

    _log_upload_step(
        request_id,
        "images_opened",
        thermal_size=f"{thermal_image_width}x{thermal_image_height}",
        rgb_size=f"{rgb_image_width}x{rgb_image_height}" if not is_thermal_only else None,
        analysis_mode=analysis_mode,
    )

    # ------------------------------
    # รัน hotspot model บนภาพ thermal
    # ------------------------------
    _log_upload_step(request_id, "thermal_model_started", image_path=thermal_uploaded_image_filename)
    hotspot_predictions = _run_yolo_detection_from_path(
        HOTSPOT_MODEL_PATH,
        True,
        thermal_uploaded_image_path,
        HOTSPOT_CONFIDENCE,
        HOTSPOT_IOU,
        HOTSPOT_IMGSZ,
    )
    _log_upload_step(request_id, "thermal_model_done", hotspot_count=len(hotspot_predictions))

    # ------------------------------
    # คำนวณ overlay ของ thermal บน RGB
    # แล้ว crop ภาพ RGB เฉพาะส่วนที่เกี่ยวข้องก่อน detect equipment
    # ------------------------------
    rgb_overlay_bbox = None
    rgb_crop_bbox = None
    rgb_detection_crop_margin_used = None
    rgb_detection_size = None
    rgb_scale_x = 1.0
    rgb_scale_y = 1.0
    rgb_crop_offset_x = 0
    rgb_crop_offset_y = 0
    equipment_predictions: list[dict[str, Any]] = []
    equipment_predictions_for_debug: list[dict[str, Any]] = []
    equipment_detection_debug_image = None

    if is_thermal_only:
        _log_upload_step(request_id, "rgb_model_skipped", reason="thermal_only")
    else:
        if rgb_uploaded_image_path is None:
            raise ValueError("RGB image path is required for paired analysis.")

        rgb_overlay_bbox = _thermal_overlay_bbox_on_rgb(
            thermal_image_width,
            thermal_image_height,
            rgb_image_width,
            rgb_image_height,
        )
        if RGB_DETECTION_CROP_MARGIN < 0:
            rgb_crop_bbox = (0, 0, rgb_image_width, rgb_image_height)
        else:
            rgb_crop_bbox = _dilate_bbox(
                rgb_overlay_bbox,
                RGB_DETECTION_CROP_MARGIN,
                rgb_image_width,
                rgb_image_height,
            )
        rgb_detection_crop_margin_used = RGB_DETECTION_CROP_MARGIN
        (
            rgb_detection_path,
            rgb_scale_x,
            rgb_scale_y,
            rgb_crop_offset_x,
            rgb_crop_offset_y,
            rgb_temp_path,
            rgb_crop_size,
            rgb_detection_size,
            _,
        ) = _prepare_cropped_resized_inference_image(
            rgb_uploaded_image_path,
            file_id,
            "rgb",
            rgb_crop_bbox,
            RGB_DETECTION_MAX_DIM,
        )

        _log_upload_step(
            request_id,
            "rgb_model_started",
            image_path=rgb_detection_path.name,
            original_size=f"{rgb_image_width}x{rgb_image_height}",
            overlay_bbox=f"{rgb_overlay_bbox[0]},{rgb_overlay_bbox[1]},{rgb_overlay_bbox[2]},{rgb_overlay_bbox[3]}",
            crop_bbox=f"{rgb_crop_bbox[0]},{rgb_crop_bbox[1]},{rgb_crop_bbox[2]},{rgb_crop_bbox[3]}",
            crop_margin=rgb_detection_crop_margin_used,
            crop_mode="full_image" if RGB_DETECTION_CROP_MARGIN < 0 else "thermal_overlay_margin",
            crop_size=f"{rgb_crop_size[0]}x{rgb_crop_size[1]}",
            detect_size=f"{rgb_detection_size[0]}x{rgb_detection_size[1]}",
            resized=rgb_detection_size != rgb_crop_size,
        )

        try:
            equipment_predictions = _run_yolo_detection_from_path(
                EQUIPMENT_MODEL_PATH,
                False,
                rgb_detection_path,
                EQUIPMENT_CONFIDENCE,
                EQUIPMENT_IOU,
                EQUIPMENT_IMGSZ,
            )
            # เก็บผล detect ดิบไว้ก่อน map bbox กลับภาพ RGB เต็มใบ
            # ภาพ debug ต้องใช้พิกัดนี้ เพราะเป็นพิกัดเดียวกับภาพที่ส่งเข้า equipment model จริง
            for equipment_prediction in equipment_predictions:
                equipment_predictions_for_debug.append(
                    {
                        "bbox": list(equipment_prediction["bbox"]),
                        "class_id": equipment_prediction["class_id"],
                        "confidence": round(float(equipment_prediction["confidence"]), 4),
                        "label": _equipment_label_for_class(equipment_prediction["class_id"]),
                    }
                )
            with Image.open(rgb_detection_path) as rgb_debug_source:
                equipment_detection_debug_image = rgb_debug_source.convert("RGB")
            if rgb_detection_size is not None and equipment_detection_debug_image is not None:
                _draw_equipment_model_debug_image(
                    equipment_detection_debug_image,
                    equipment_predictions_for_debug,
                    rgb_detection_size[0],
                    rgb_detection_size[1],
                )
        finally:
            # ลบไฟล์ detect ชั่วคราวทิ้งหลังใช้
            if rgb_temp_path is not None and rgb_temp_path.exists():
                rgb_temp_path.unlink()

        # map bbox ของผล detect บนภาพ cropped/resized กลับไปยังพิกัดภาพ RGB จริง
        for equipment_prediction in equipment_predictions:
            equipment_prediction["bbox"] = [
                float(equipment_prediction["bbox"][0] * rgb_scale_x + rgb_crop_offset_x),
                float(equipment_prediction["bbox"][1] * rgb_scale_y + rgb_crop_offset_y),
                float(equipment_prediction["bbox"][2] * rgb_scale_x + rgb_crop_offset_x),
                float(equipment_prediction["bbox"][3] * rgb_scale_y + rgb_crop_offset_y),
            ]
        _log_upload_step(request_id, "rgb_model_done", equipment_count=len(equipment_predictions))

    # ------------------------------
    # extract thermal matrix
    # ------------------------------
    thermal_matrix, thermal_error, thermal_mode = extract_thermal_matrix(
        str(thermal_uploaded_image_path),
        expected_width=thermal_image_width,
        expected_height=thermal_image_height,
    )
    _log_upload_step(
        request_id,
        "thermal_extraction_done",
        thermal_mode=thermal_mode,
        has_thermal_matrix=thermal_matrix is not None,
        thermal_error=bool(thermal_error),
    )

    # ------------------------------
    # เตรียม thermal matrix ที่จะใช้วิเคราะห์จริง
    # และคำนวณ reference temperature
    # ------------------------------
    thermal_analysis_matrix = None
    has_absolute_temperature = False
    thermal_height, thermal_width = 0, 0
    reference_temperature = None
    thermal_image_min_temperature = None
    thermal_image_max_temperature = None

    if thermal_matrix is not None:
        finite_values = thermal_matrix[np.isfinite(thermal_matrix)]
        if thermal_mode == "absolute":
            # รองรับกรณีที่ค่ามาในสเกลสูง เช่น Kelvin*25
            # ตัวอย่าง: กล้องบางรุ่นไม่ให้ค่าเป็นองศา C ตรง ๆ
            # จึงต้องแปลงสเกลก่อนเพื่อให้ตัวเลขอ่านเป็นอุณหภูมิจริง
            if finite_values.size > 0 and float(finite_values.max()) > 1000.0:
                thermal_analysis_matrix = thermal_matrix * 0.04 - 273.15
            else:
                thermal_analysis_matrix = thermal_matrix
            has_absolute_temperature = True
            reference_temperature = _compute_reference_temperature(thermal_analysis_matrix)
        else:
            # relative mode ใช้ค่าดิบสำหรับเทียบจุดร้อน
            thermal_analysis_matrix = thermal_matrix

        thermal_height, thermal_width = thermal_analysis_matrix.shape
        thermal_image_min_temperature, thermal_image_max_temperature = _get_thermal_matrix_temperature_range(
            thermal_analysis_matrix if has_absolute_temperature else None
        )

    _log_upload_step(
        request_id,
        "thermal_matrix_ready",
        absolute=has_absolute_temperature,
        reference_temp=round(reference_temperature, 2) if reference_temperature is not None else None,
        matrix_size=f"{thermal_width}x{thermal_height}" if thermal_analysis_matrix is not None else None,
    )

    # ------------------------------
    # เปิดภาพ thermal เพื่อวาด annotation
    # ------------------------------
    _log_upload_step(request_id, "annotation_image_open_started")
    with Image.open(thermal_uploaded_image_path) as thermal_source_image:
        # ภาพนี้คือภาพ thermal แบบที่ได้จากไฟล์ต้นฉบับ/กล้อง
        annotated_image_camera = thermal_source_image.convert("RGB")

    # สร้างภาพ thermal แบบ fixed range เพิ่มอีกชุด
    # ภาพนี้ใช้ช่วงสีคงที่จาก DISPLAY_TEMP_MIN_C ถึง DISPLAY_TEMP_MAX_C
    # จุดประสงค์คือให้คนดูเปรียบเทียบหลายภาพได้ง่ายขึ้นว่า สีที่ใกล้กันหมายถึงอุณหภูมิที่ใกล้กัน
    # สำคัญ: ภาพนี้ใช้เพื่อการแสดงผลเท่านั้น ไม่ได้เอาไปเปลี่ยนผล model หรือค่าคำนวณ hotspot
    annotated_image_fixed_range = (
        _render_fixed_range_thermal_image(
            thermal_analysis_matrix,
            thermal_image_width,
            thermal_image_height,
        )
        if has_absolute_temperature
        else None
    )

    # เก็บ fixed-range แบบไม่มีกรอบ hotspot แยกไว้ก่อน
    # frontend ใช้รูปนี้เป็นไฟล์ดาวน์โหลด เพื่อให้ผู้ใช้ได้ภาพล้วน ๆ ไม่ติด annotation
    fixed_range_image_plain = annotated_image_fixed_range.copy() if annotated_image_fixed_range is not None else None

    # thermal_draws คือรายการ canvas ที่ต้องวาด hotspot ทับ
    # มีภาพกล้องปกติเสมอ และมีภาพ fixed-range เพิ่มเมื่ออ่าน absolute temperature ได้
    thermal_draws = [ImageDraw.Draw(annotated_image_camera)]
    if annotated_image_fixed_range is not None:
        thermal_draws.append(ImageDraw.Draw(annotated_image_fixed_range))
    _log_upload_step(request_id, "annotation_image_open_finished")

    # ------------------------------
    # เตรียมรายการ equipment detections ให้อยู่ในรูปพร้อม match
    # ------------------------------
    equipments: list[dict[str, Any]] = []
    for equipment_prediction in equipment_predictions:
        equipment_box = tuple(
            _safe_bbox(
                int(round(equipment_prediction["bbox"][0])),
                int(round(equipment_prediction["bbox"][1])),
                int(round(equipment_prediction["bbox"][2])),
                int(round(equipment_prediction["bbox"][3])),
                rgb_image_width,
                rgb_image_height,
            )
        )
        equipment_label = _equipment_label_for_class(equipment_prediction["class_id"])
        equipment = {
            "bbox": equipment_box,
            "class_id": equipment_prediction["class_id"],
            "confidence": round(float(equipment_prediction["confidence"]), 4),
            "label": equipment_label,
        }
        equipments.append(equipment)

    # ------------------------------
    # [เพิ่มใหม่]
    # บันทึกภาพ debug ก่อน match
    # 1) hotspot model บน thermal
    # 2) RGB crop/resize ที่ส่งเข้า equipment model แต่ยังไม่วาดกรอบ equipment
    # ภาพ RGB จะถูกวาดเฉพาะกรอบ hotspot ที่ project จาก thermal หลังสร้าง detections แล้ว
    # ------------------------------
    with Image.open(thermal_uploaded_image_path) as thermal_debug_source:
        hotspot_detection_debug_image = thermal_debug_source.convert("RGB")
    _draw_hotspot_model_debug_image(
        hotspot_detection_debug_image,
        hotspot_predictions,
        thermal_image_width,
        thermal_image_height,
    )

    if equipment_detection_debug_image is None and rgb_detection_size is not None:
        equipment_detection_debug_image = Image.new("RGB", (rgb_detection_size[0], rgb_detection_size[1]), "#f4f1ed")

    detections = []

    # ------------------------------
    # วนลูป hotspot ทีละตัว
    # ------------------------------
    for hotspot_index, hotspot_prediction in enumerate(hotspot_predictions, start=1):
        thermal_box = tuple(
            _safe_bbox(
                int(round(hotspot_prediction["bbox"][0])),
                int(round(hotspot_prediction["bbox"][1])),
                int(round(hotspot_prediction["bbox"][2])),
                int(round(hotspot_prediction["bbox"][3])),
                thermal_image_width,
                thermal_image_height,
            )
        )

        if is_thermal_only:
            # thermal-only ไม่มีภาพ RGB ให้ project ไปหาอุปกรณ์
            # จึงใช้พิกัด thermal เดิมเป็น bbox/center สำหรับแสดงผล hotspot
            rgb_box = thermal_box
            hotspot_center = (
                (thermal_box[0] + thermal_box[2]) / 2.0,
                (thermal_box[1] + thermal_box[3]) / 2.0,
            )
        else:
            # project กรอบ thermal -> RGB
            rgb_box = _project_thermal_bbox_to_rgb(
                thermal_box,
                thermal_image_width,
                thermal_image_height,
                rgb_image_width,
                rgb_image_height,
            )

            # หา center ของ hotspot ในพิกัด RGB
            hotspot_center = _project_thermal_point_to_rgb(
                (thermal_box[0] + thermal_box[2]) / 2.0,
                (thermal_box[1] + thermal_box[3]) / 2.0,
                thermal_image_width,
                thermal_image_height,
                rgb_image_width,
                rgb_image_height,
            )

        # [แก้สำคัญจากโค้ดเก่า]
        # detection ใหม่มีข้อมูลละเอียดขึ้นมาก
        # อธิบายแบบสั้น:
        # - bbox / thermal_bbox: กรอบในพิกัด RGB และ thermal
        # - max/min/avg_temp: อุณหภูมิจริง (ถ้าระบบอ่าน absolute ได้)
        # - max/min/avg_raw: ค่าดิบ (ใช้เมื่อเป็น relative mode)
        # - reference_temp + delta_above_reference: เทียบความร้อนกับพื้นหลัง
        # - priority / action_required: ข้อเสนอความเร่งด่วนการซ่อม
        detection = {
            "bbox": list(rgb_box),
            "thermal_bbox": list(thermal_box),
            "hotspot_confidence": round(float(hotspot_prediction["confidence"]), 4),
            "hotspot_center": list(hotspot_center),
            "max_temp": None,
            "min_temp": None,
            "avg_temp": None,
            "max_point": None,
            "min_point": None,
            "max_raw": None,
            "min_raw": None,
            "avg_raw": None,
            "reference_temp": reference_temperature,
            "delta_above_reference": None,
            "priority": None,
            "action_required": None,
        }

        # วาดกรอบ hotspot บนภาพ thermal
        for thermal_draw in thermal_draws:
            thermal_draw.rectangle(thermal_box, outline="orange", width=3)
            _draw_hotspot_index_label(
                draw=thermal_draw,
                thermal_box=thermal_box,
                hotspot_index=hotspot_index,
                thermal_image_width=thermal_image_width,
                thermal_image_height=thermal_image_height,
            )

        # ------------------------------
        # ถ้ามี thermal matrix ให้คำนวณค่าสถิติในกรอบ
        # ------------------------------
        if thermal_analysis_matrix is not None:
            thermal_x1 = int(np.floor(thermal_box[0] * thermal_width / thermal_image_width))
            thermal_x2 = int(np.ceil(thermal_box[2] * thermal_width / thermal_image_width))
            thermal_y1 = int(np.floor(thermal_box[1] * thermal_height / thermal_image_height))
            thermal_y2 = int(np.ceil(thermal_box[3] * thermal_height / thermal_image_height))

            thermal_x1 = max(0, min(thermal_x1, thermal_width - 1))
            thermal_y1 = max(0, min(thermal_y1, thermal_height - 1))
            thermal_x2 = max(thermal_x1 + 1, min(thermal_x2, thermal_width))
            thermal_y2 = max(thermal_y1 + 1, min(thermal_y2, thermal_height))

            thermal_region = thermal_analysis_matrix[thermal_y1:thermal_y2, thermal_x1:thermal_x2]
            finite_region = thermal_region[np.isfinite(thermal_region)]

            if finite_region.size > 0:
                max_value = float(np.nanmax(thermal_region))
                min_value = float(np.nanmin(thermal_region))
                avg_value = float(np.nanmean(thermal_region))

                max_position = np.unravel_index(int(np.nanargmax(thermal_region)), thermal_region.shape)
                min_position = np.unravel_index(int(np.nanargmin(thermal_region)), thermal_region.shape)

                # แปลงจุด max/min กลับไปยังพิกัดภาพ thermal
                max_point_thermal_x = int((thermal_x1 + max_position[1]) * thermal_image_width / thermal_width)
                max_point_thermal_y = int((thermal_y1 + max_position[0]) * thermal_image_height / thermal_height)
                min_point_thermal_x = int((thermal_x1 + min_position[1]) * thermal_image_width / thermal_width)
                min_point_thermal_y = int((thermal_y1 + min_position[0]) * thermal_image_height / thermal_height)

                # วาดจุดร้อน/เย็น
                for thermal_draw in thermal_draws:
                    thermal_draw.ellipse(
                        [
                            max_point_thermal_x - 4,
                            max_point_thermal_y - 4,
                            max_point_thermal_x + 4,
                            max_point_thermal_y + 4,
                        ],
                        fill="red",
                    )
                    thermal_draw.ellipse(
                        [
                            min_point_thermal_x - 4,
                            min_point_thermal_y - 4,
                            min_point_thermal_x + 4,
                            min_point_thermal_y + 4,
                        ],
                        fill="blue",
                    )
                detection["max_point"] = [max_point_thermal_x, max_point_thermal_y]
                detection["min_point"] = [min_point_thermal_x, min_point_thermal_y]

                if has_absolute_temperature:
                    for thermal_draw in thermal_draws:
                        thermal_draw.text(
                            (thermal_box[0], max(0, thermal_box[1] - 15)),
                            f"max {max_value:.1f}C min {min_value:.1f}C avg {avg_value:.1f}C",
                            fill="white",
                        )
                    detection["max_temp"] = max_value
                    detection["min_temp"] = min_value
                    detection["avg_temp"] = avg_value

                    # [ส่วนคำนวณความร้อนเทียบค่าอ้างอิง]
                    # คำนวณ delta_above_reference และ priority
                    if reference_temperature is not None:
                        delta_above_reference = max_value - reference_temperature
                        priority, action_required = _classify_priority(delta_above_reference)
                        detection["delta_above_reference"] = delta_above_reference
                        detection["priority"] = priority
                        detection["action_required"] = action_required
                else:
                    # ถ้าไม่มี absolute temperature ให้เก็บค่าดิบแทน
                    detection["max_raw"] = max_value
                    detection["min_raw"] = min_value
                    detection["avg_raw"] = avg_value

        # [ส่วนจับคู่ hotspot กับอุปกรณ์]
        # match hotspot จุดนี้กับ equipment บน RGB
        if is_thermal_only:
            detection.update(
                {
                    "equipment_class": "unknown",
                    "equipment_confidence": None,
                    "equipment_bbox": None,
                    "match_method": "unknown",
                    "match_distance": None,
                }
            )
        else:
            detection.update(_match_equipment(hotspot_center, equipments, rgb_image_width, rgb_image_height))
        detections.append(detection)

    _log_upload_step(request_id, "matching_done", detection_count=len(detections))

    # ------------------------------
    # [เปลี่ยนจากโค้ดเก่า]
    # เดิมแปลง annotated image เป็น base64 data URL ส่งกลับเลย
    # ใหม่บันทึกเป็นไฟล์จริงใน /uploads แล้วส่ง path กลับ
    # ------------------------------
    annotated_image_filename = f"{file_id}_annotated.jpg"
    annotated_image_path = UPLOAD_DIR / annotated_image_filename
    annotated_image_camera.save(annotated_image_path, format="JPEG", quality=90)
    annotated_image_camera.close()

    hotspot_detection_debug_filename = f"{file_id}_hotspot_detection_debug.jpg"
    hotspot_detection_debug_path = UPLOAD_DIR / hotspot_detection_debug_filename
    hotspot_detection_debug_image.save(hotspot_detection_debug_path, format="JPEG", quality=90)
    hotspot_detection_debug_image.close()

    equipment_detection_debug_filename = None
    if equipment_detection_debug_image is not None:
        equipment_detection_debug_filename = f"{file_id}_equipment_detection_debug.jpg"
        equipment_detection_debug_path = UPLOAD_DIR / equipment_detection_debug_filename
        equipment_detection_debug_image.save(equipment_detection_debug_path, format="JPEG", quality=90)
        equipment_detection_debug_image.close()

    # บันทึกภาพ fixed-range เพิ่ม 2 แบบ
    # 1) annotated_image_fixed_range = มีกรอบ hotspot และข้อความเหมือนภาพหลัก
    # 2) fixed_range_image = ไม่มีกรอบ hotspot เอาไว้ให้ปุ่ม Download โหลดภาพล้วน
    annotated_image_fixed_range_filename = None
    if annotated_image_fixed_range is not None:
        annotated_image_fixed_range_filename = f"{file_id}_annotated_fixed_range.jpg"
        annotated_image_fixed_range_path = UPLOAD_DIR / annotated_image_fixed_range_filename
        annotated_image_fixed_range.save(annotated_image_fixed_range_path, format="JPEG", quality=90)
        annotated_image_fixed_range.close()

    fixed_range_image_filename = None
    if fixed_range_image_plain is not None:
        fixed_range_image_filename = f"{file_id}_fixed_range.jpg"
        fixed_range_image_path = UPLOAD_DIR / fixed_range_image_filename
        fixed_range_image_plain.save(fixed_range_image_path, format="JPEG", quality=90)
        fixed_range_image_plain.close()
    gc.collect()

    _log_upload_step(
        request_id,
        "annotated_image_saved",
        annotated_path=annotated_image_filename,
        annotated_fixed_range_path=annotated_image_fixed_range_filename,
        hotspot_debug_path=hotspot_detection_debug_filename,
        equipment_debug_path=equipment_detection_debug_filename,
    )

    response = {
        "success": True,
        "file_id": file_id,
        "analysis_mode": analysis_mode,
        "uploaded_image": f"/uploads/{thermal_uploaded_image_filename}",
        "uploaded_rgb_image": f"/uploads/{rgb_uploaded_image_filename}" if rgb_uploaded_image_filename else None,
        "annotated_image": f"/uploads/{annotated_image_filename}",
        "annotated_image_camera": f"/uploads/{annotated_image_filename}",
        "hotspot_detection_image": f"/uploads/{hotspot_detection_debug_filename}",
        "equipment_detection_image": f"/uploads/{equipment_detection_debug_filename}" if equipment_detection_debug_filename else None,
        "annotated_image_fixed_range": (
            f"/uploads/{annotated_image_fixed_range_filename}" if annotated_image_fixed_range_filename else None
        ),
        "fixed_range_image": f"/uploads/{fixed_range_image_filename}" if fixed_range_image_filename else None,
        "detections": detections,
        "rgb_detection_crop_margin": rgb_detection_crop_margin_used,
        "rgb_detection_crop_bbox": list(rgb_crop_bbox) if rgb_crop_bbox is not None else None,
        "rgb_detection_size": list(rgb_detection_size) if rgb_detection_size is not None else None,
        "has_gps": has_gps,
        "message": None,
        "thermal_available": has_absolute_temperature,
        "thermal_mode": thermal_mode,
        "thermal_error": thermal_error,
        "reference_temperature": reference_temperature,
        "thermal_image_min_temperature": thermal_image_min_temperature,
        "thermal_image_max_temperature": thermal_image_max_temperature,
        "fixed_range_min_temperature": DISPLAY_TEMP_MIN_C if annotated_image_fixed_range_filename else None,
        "fixed_range_max_temperature": DISPLAY_TEMP_MAX_C if annotated_image_fixed_range_filename else None,
        "request_id": request_id,
    }

    # ถ้ามีพิกัด GPS ให้แนบกลับไปเพื่อใช้แสดงตำแหน่งในแผนที่
    # ถ้าไม่มี จะส่งข้อความบอกสาเหตุให้ frontend แสดงแจ้งผู้ใช้แทน
    if has_gps:
        response["latitude"] = latitude
        response["longitude"] = longitude
    else:
        response["message"] = "No GPS data found in thermal image"

    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    _log_upload_step(request_id, "upload_completed", elapsed_seconds=elapsed_seconds)
    return response


# ------------------------------
# [เพิ่มในเวอร์ชันล่าสุด]
# endpoint นี้เปิดทางให้ frontend ส่งกรอบ ROI ที่ผู้ใช้ลากเองเข้ามา
# เพื่อคำนวณ reference temperature ใหม่ แล้วอัปเดต priority/action ของ hotspot เดิม
# จุดเด่นคือ "ไม่ต้องรันโมเดลตรวจจับใหม่" ทำให้ตอบกลับเร็วกว่า
# ------------------------------
@app.post("/reference-roi")
async def apply_reference_roi(request: Request):
    """
    คำนวณ reference temperature ใหม่จาก ROI ที่ผู้ใช้ลากบนภาพ thermal
    แล้วอัปเดต priority/action ของ hotspot เดิมโดยไม่ rerun model
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])

    try:
        payload = await request.json()
    except Exception:
        return _json_error("Reference ROI request must be valid JSON.", request_id, 400)

    file_id = str(payload.get("file_id") or "").strip() if isinstance(payload, dict) else ""
    if not file_id:
        return _json_error("Reference ROI request requires file_id.", request_id, 400)

    try:
        normalized_roi = _parse_normalized_roi(payload.get("roi") if isinstance(payload, dict) else None)
    except ValueError as error:
        return _json_error(str(error), request_id, 400, file_id=file_id)

    raw_detections = payload.get("detections") if isinstance(payload, dict) else None
    if not isinstance(raw_detections, list):
        return _json_error("Reference ROI request requires detections.", request_id, 400, file_id=file_id)

    detections: list[dict[str, Any]] = []
    for detection in raw_detections:
        if not isinstance(detection, dict):
            return _json_error("Each detection in the ROI request must be an object.", request_id, 400, file_id=file_id)
        detections.append(dict(detection))

    thermal_file = _find_uploaded_file(file_id, "thermal")
    if thermal_file is None:
        return _json_error("Thermal upload not found for the requested file_id.", request_id, 404, file_id=file_id)

    thermal_uploaded_image_filename, thermal_uploaded_image_path = thermal_file

    with Image.open(thermal_uploaded_image_path) as thermal_source_image:
        thermal_image_width, thermal_image_height = thermal_source_image.size

    thermal_matrix, thermal_error, thermal_mode = extract_thermal_matrix(
        str(thermal_uploaded_image_path),
        expected_width=thermal_image_width,
        expected_height=thermal_image_height,
    )

    if thermal_matrix is None or thermal_mode != "absolute":
        return _json_error(
            thermal_error or "ROI reference requires absolute thermal temperature data.",
            request_id,
            400,
            file_id=file_id,
        )

    finite_values = thermal_matrix[np.isfinite(thermal_matrix)]
    if finite_values.size > 0 and float(finite_values.max()) > 1000.0:
        thermal_analysis_matrix = thermal_matrix * 0.04 - 273.15
    else:
        thermal_analysis_matrix = thermal_matrix

    roi_box = _normalized_roi_to_image_box(normalized_roi, thermal_image_width, thermal_image_height)
    reference_temperature = _compute_reference_temperature_from_roi(
        thermal_analysis_matrix,
        roi_box,
        thermal_image_width,
        thermal_image_height,
    )
    if reference_temperature is None:
        return _json_error(
            "Selected ROI does not contain valid temperature pixels.",
            request_id,
            400,
            file_id=file_id,
        )

    try:
        recalculated_detections = _recalculate_detections_for_reference_roi(
            detections,
            thermal_analysis_matrix,
            thermal_image_width,
            thermal_image_height,
            reference_temperature,
        )
    except ValueError as error:
        return _json_error(str(error), request_id, 400, file_id=file_id)

    return {
        "success": True,
        "file_id": file_id,
        "request_id": request_id,
        "reference_source": "roi",
        "reference_temperature": reference_temperature,
        "roi": normalized_roi,
        "detections": recalculated_detections,
        "thermal_image": f"/uploads/{thermal_uploaded_image_filename}",
    }


# ------------------------------
# [เพิ่มใหม่]
# endpoint สำหรับ render ภาพ thermal ด้วย display range ที่ผู้ใช้เลือกเอง
#
# จุดสำคัญ:
# - ไม่ rerun model
# - ใช้ไฟล์ thermal เดิมจาก file_id
# - ใช้ detections เดิมเพื่อวาด hotspot ซ้ำบนภาพ range ใหม่
# ------------------------------
@app.post("/display-range")
async def apply_display_range(request: Request):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])

    try:
        payload = await request.json()
    except Exception:
        return _json_error("Display range request must be valid JSON.", request_id, 400)

    if not isinstance(payload, dict):
        return _json_error("Display range request payload must be an object.", request_id, 400)

    file_id = str(payload.get("file_id") or "").strip()
    if not file_id:
        return _json_error("Display range request requires file_id.", request_id, 400)

    try:
        display_min_c = float(payload.get("min_temp"))
        display_max_c = float(payload.get("max_temp"))
    except (TypeError, ValueError):
        return _json_error("Display range min_temp and max_temp must be numbers.", request_id, 400, file_id=file_id)

    if not np.isfinite(display_min_c) or not np.isfinite(display_max_c):
        return _json_error("Display range min_temp and max_temp must be finite.", request_id, 400, file_id=file_id)
    if display_max_c <= display_min_c:
        return _json_error("Display range max_temp must be greater than min_temp.", request_id, 400, file_id=file_id)

    raw_detections = payload.get("detections")
    if not isinstance(raw_detections, list):
        return _json_error("Display range request requires detections.", request_id, 400, file_id=file_id)

    detections: list[dict[str, Any]] = []
    for detection in raw_detections:
        if not isinstance(detection, dict):
            return _json_error("Each detection in the display range request must be an object.", request_id, 400, file_id=file_id)
        detections.append(dict(detection))

    thermal_file = _find_uploaded_file(file_id, "thermal")
    if thermal_file is None:
        return _json_error("Thermal upload not found for the requested file_id.", request_id, 404, file_id=file_id)

    thermal_uploaded_image_filename, thermal_uploaded_image_path = thermal_file

    with Image.open(thermal_uploaded_image_path) as thermal_source_image:
        thermal_image_width, thermal_image_height = thermal_source_image.size

    thermal_matrix, thermal_error, thermal_mode = extract_thermal_matrix(
        str(thermal_uploaded_image_path),
        expected_width=thermal_image_width,
        expected_height=thermal_image_height,
    )

    if thermal_matrix is None or thermal_mode != "absolute":
        return _json_error(
            thermal_error or "Custom display range requires absolute thermal temperature data.",
            request_id,
            400,
            file_id=file_id,
        )

    finite_values = thermal_matrix[np.isfinite(thermal_matrix)]
    if finite_values.size > 0 and float(finite_values.max()) > 1000.0:
        thermal_analysis_matrix = thermal_matrix * 0.04 - 273.15
    else:
        thermal_analysis_matrix = thermal_matrix

    rendered_image = _render_fixed_range_thermal_image(
        thermal_analysis_matrix,
        thermal_image_width,
        thermal_image_height,
        display_min_c=display_min_c,
        display_max_c=display_max_c,
    )
    if rendered_image is None:
        return _json_error("Display range image could not be rendered.", request_id, 400, file_id=file_id)

    thermal_image_min_temperature, thermal_image_max_temperature = _get_thermal_matrix_temperature_range(
        thermal_analysis_matrix
    )
    fixed_range_image_plain = rendered_image.copy()
    _draw_detection_annotations_on_thermal(
        rendered_image,
        detections,
        thermal_image_width,
        thermal_image_height,
    )

    range_label = f"{display_min_c:.2f}_{display_max_c:.2f}".replace("-", "m").replace(".", "p")
    annotated_image_fixed_range_filename = f"{file_id}_range_{range_label}_{request_id}_annotated.jpg"
    fixed_range_image_filename = f"{file_id}_range_{range_label}_{request_id}.jpg"
    annotated_image_fixed_range_path = UPLOAD_DIR / annotated_image_fixed_range_filename
    fixed_range_image_path = UPLOAD_DIR / fixed_range_image_filename

    rendered_image.save(annotated_image_fixed_range_path, format="JPEG", quality=90)
    rendered_image.close()
    fixed_range_image_plain.save(fixed_range_image_path, format="JPEG", quality=90)
    fixed_range_image_plain.close()
    gc.collect()

    _log_readable_event(
        request_id,
        "🎚️",
        "display_range_rendered",
        file_id=file_id,
        min_temp=round(display_min_c, 2),
        max_temp=round(display_max_c, 2),
        annotated_path=annotated_image_fixed_range_filename,
    )

    return {
        "success": True,
        "file_id": file_id,
        "request_id": request_id,
        "annotated_image_fixed_range": f"/uploads/{annotated_image_fixed_range_filename}",
        "fixed_range_image": f"/uploads/{fixed_range_image_filename}",
        "fixed_range_min_temperature": display_min_c,
        "fixed_range_max_temperature": display_max_c,
        "thermal_image_min_temperature": thermal_image_min_temperature,
        "thermal_image_max_temperature": thermal_image_max_temperature,
    }


# ------------------------------
# [เพิ่มใหม่]
# endpoint สำหรับเขียน log สรุป batch ก่อนเริ่ม upload จริง
#
# frontend จะเรียก endpoint นี้ 1 ครั้งต่อการกด Analyze All Pairs
# เพื่อให้ terminal/render เห็นว่า batch นี้มีทั้งหมดกี่ไฟล์ และชื่อไฟล์อะไรบ้าง
# ------------------------------
@app.post("/batch-log")
async def log_upload_batch(request: Request):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])

    try:
        payload = await request.json()
    except Exception:
        return _json_error("Batch log request must be valid JSON.", request_id, 400)

    raw_file_names = payload.get("fileNames") or payload.get("file_names") or []
    file_names = [str(file_name) for file_name in raw_file_names] if isinstance(raw_file_names, list) else []
    total_files = _safe_positive_int(payload.get("totalFiles") or payload.get("total_files")) or len(file_names)
    total_pairs = _safe_positive_int(payload.get("totalPairs") or payload.get("total_pairs"))
    batch_id = str(payload.get("batchId") or payload.get("batch_id") or request_id).strip()

    _log_readable_event(
        request_id,
        "📦",
        "upload_batch_summary",
        batch_id=batch_id,
        total_files=total_files,
        total_pairs=total_pairs,
        filenames=" | ".join(file_names),
    )

    for file_index, file_name in enumerate(file_names, start=1):
        _log_readable_event(
            request_id,
            "📄",
            "upload_batch_file",
            batch_id=batch_id,
            file_index=f"{file_index}/{total_files}",
            filename=file_name,
        )

    return {
        "success": True,
        "request_id": request_id,
        "batch_id": batch_id,
        "total_files": total_files,
        "total_pairs": total_pairs,
    }


# ------------------------------
# ส่วนที่ 5: endpoint สำหรับอัปโหลดไฟล์ทีละใบ
# endpoint /upload-file
#
# ใช้สำหรับ raw streaming upload ทีละไฟล์
# รองรับ:
# - kind=thermal หรือ rgb
# - file_id=... (optional)
#
# flow ใหม่:
# 1) upload thermal
# 2) upload rgb
# 3) เรียก /analyze ทีหลัง
# ------------------------------
@app.post("/upload-file")
async def upload_file_raw(request: Request):
    """
    endpoint อัปโหลดไฟล์แบบ stream ทีละไฟล์ (thermal หรือ rgb)
    เหมาะกับงานที่อยากแยกอัปโหลดก่อน แล้วค่อยเรียก /analyze ทีหลัง
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
    started_at = time.perf_counter()

    kind = (request.query_params.get("kind") or "").strip().lower()
    file_id = (request.query_params.get("file_id") or uuid.uuid4().hex).strip()
    original_name = (request.headers.get("x-file-name") or f"{kind or 'upload'}.jpg").strip()
    batch_details = _read_batch_log_headers(request)

    if kind not in {"thermal", "rgb"}:
        return _json_error("Upload kind must be 'thermal' or 'rgb'.", request_id, 400)

    # สร้าง path ปลายทางก่อน แล้วเขียนทับด้วย stream ทีละ chunk
    upload_filename, upload_path = _save_upload_bytes(original_name, file_id, kind, b"")
    bytes_written = 0

    try:
        _log_readable_event(
            request_id,
            "🚀",
            "upload_file_started",
            file_id=file_id,
            kind=kind,
            filename=original_name,
            **batch_details,
        )
        _log_upload_step(
            request_id,
            "raw_upload_started",
            file_id=file_id,
            kind=kind,
            filename=original_name,
            content_length=request.headers.get("content-length"),
            **batch_details,
        )

        # อ่าน request body แบบ stream เพื่อลดการใช้ memory
        with upload_path.open("wb") as uploaded_file:
            async for chunk in request.stream():
                if not chunk:
                    continue
                uploaded_file.write(chunk)
                bytes_written += len(chunk)

        if bytes_written <= 0:
            if upload_path.exists():
                upload_path.unlink()
            return _json_error("Uploaded file was empty.", request_id, 400, file_id=file_id, kind=kind)

        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        _log_readable_event(
            request_id,
            "✅",
            "upload_file_finished",
            file_id=file_id,
            kind=kind,
            filename=original_name,
            bytes_written=bytes_written,
            elapsed_seconds=elapsed_seconds,
            **batch_details,
        )
        _log_upload_step(
            request_id,
            "raw_upload_finished",
            file_id=file_id,
            kind=kind,
            filename=original_name,
            saved_path=upload_filename,
            bytes_written=bytes_written,
            elapsed_seconds=elapsed_seconds,
            **batch_details,
        )

        return {
            "success": True,
            "file_id": file_id,
            "kind": kind,
            "uploaded_image": f"/uploads/{upload_filename}",
            "request_id": request_id,
        }

    except ClientDisconnect:
        if upload_path.exists():
            upload_path.unlink()
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        _log_readable_event(
            request_id,
            "⚠️",
            "upload_file_disconnected",
            file_id=file_id,
            kind=kind,
            filename=original_name,
            elapsed_seconds=elapsed_seconds,
            **batch_details,
        )
        _set_request_progress(request_id, "raw_upload_client_disconnected", kind=kind, elapsed_seconds=elapsed_seconds)
        request_progress[request_id]["failed"] = True
        logger.warning("[%s] raw_upload_client_disconnected kind=%s elapsed_seconds=%s", request_id, kind, elapsed_seconds)
        return _json_error(
            "Upload connection dropped before the backend received the full file.",
            request_id,
            499,
            file_id=file_id,
            kind=kind,
        )
    except Exception:
        if upload_path.exists():
            upload_path.unlink()
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        _log_readable_event(
            request_id,
            "❌",
            "upload_file_failed",
            file_id=file_id,
            kind=kind,
            filename=original_name,
            elapsed_seconds=elapsed_seconds,
            **batch_details,
        )
        _set_request_progress(request_id, "raw_upload_failed", kind=kind, elapsed_seconds=elapsed_seconds)
        request_progress[request_id]["failed"] = True
        logger.exception("[%s] raw_upload_failed kind=%s elapsed_seconds=%s", request_id, kind, elapsed_seconds)
        return _json_error(
            "Backend failed while receiving the uploaded file.",
            request_id,
            500,
            file_id=file_id,
            kind=kind,
        )


# ------------------------------
# ส่วนที่ 6: endpoint สำหรับสั่งวิเคราะห์จากไฟล์ที่อัปโหลดไว้แล้ว
# endpoint /analyze
#
# รับ JSON { "file_id": "..." }
# แล้วไปหาไฟล์ thermal/rgb ที่ upload ไว้ก่อนหน้า
# จากนั้นค่อยวิเคราะห์
# ------------------------------
@app.post("/analyze")
async def analyze_uploaded_pair(request: Request):
    """
    endpoint วิเคราะห์จากไฟล์ที่อัปโหลดค้างไว้ก่อนหน้า
    รับเพียง `file_id` แล้วระบบจะไปหาไฟล์ thermal/rgb ที่จับคู่กันเอง
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
    started_at = time.perf_counter()
    batch_details = _read_batch_log_headers(request)
    thermal_original_name = (request.headers.get("x-thermal-file-name") or "").strip()
    rgb_original_name = (request.headers.get("x-rgb-file-name") or "").strip()

    try:
        payload = await request.json()
    except Exception:
        return _json_error("Analyze request must be valid JSON.", request_id, 400)

    file_id = str(payload.get("file_id") or "").strip()
    if not file_id:
        return _json_error("Analyze request requires file_id.", request_id, 400)

    analysis_mode = str(payload.get("analysis_mode") or "paired").strip().lower()
    if analysis_mode not in {"paired", "thermal_only"}:
        return _json_error("analysis_mode must be 'paired' or 'thermal_only'.", request_id, 400, file_id=file_id)

    if analysis_mode == "paired" and not EQUIPMENT_MODEL_PATH.exists():
        return _json_error(
            f"Equipment model not found at {EQUIPMENT_MODEL_PATH}. Set EQUIPMENT_MODEL_PATH first.",
            request_id,
            500,
        )

    thermal_file = _find_uploaded_file(file_id, "thermal")
    rgb_file = _find_uploaded_file(file_id, "rgb")

    if thermal_file is None:
        return _json_error("Thermal upload not found for the requested file_id.", request_id, 404, file_id=file_id)
    if analysis_mode == "paired" and rgb_file is None:
        return _json_error("RGB upload not found for the requested file_id.", request_id, 404, file_id=file_id)

    thermal_uploaded_image_filename, thermal_uploaded_image_path = thermal_file
    rgb_uploaded_image_filename, rgb_uploaded_image_path = rgb_file if rgb_file is not None else (None, None)
    thermal_log_name = thermal_original_name or thermal_uploaded_image_filename
    rgb_log_name = rgb_original_name or rgb_uploaded_image_filename or ""

    _log_readable_event(
        request_id,
        "🔎",
        "analyze_pair_started",
        file_id=file_id,
        analysis_mode=analysis_mode,
        thermal_file=thermal_log_name,
        rgb_file=rgb_log_name,
        **batch_details,
    )
    _log_upload_step(
        request_id,
        "analyze_started",
        file_id=file_id,
        analysis_mode=analysis_mode,
        thermal_path=thermal_uploaded_image_filename,
        rgb_path=rgb_uploaded_image_filename,
        thermal_file=thermal_log_name,
        rgb_file=rgb_log_name,
        **batch_details,
    )

    try:
        response = _analyze_saved_pair(
            request_id=request_id,
            started_at=started_at,
            file_id=file_id,
            thermal_uploaded_image_filename=thermal_uploaded_image_filename,
            thermal_uploaded_image_path=thermal_uploaded_image_path,
            rgb_uploaded_image_filename=rgb_uploaded_image_filename,
            rgb_uploaded_image_path=rgb_uploaded_image_path,
            analysis_mode=analysis_mode,
        )
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        _log_readable_event(
            request_id,
            "✅",
            "analyze_pair_finished",
            file_id=file_id,
            analysis_mode=analysis_mode,
            thermal_file=thermal_log_name,
            rgb_file=rgb_log_name,
            elapsed_seconds=elapsed_seconds,
            **batch_details,
        )
        return response
    except Exception:
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        _log_readable_event(
            request_id,
            "❌",
            "analyze_pair_failed",
            file_id=file_id,
            analysis_mode=analysis_mode,
            thermal_file=thermal_log_name,
            rgb_file=rgb_log_name,
            elapsed_seconds=elapsed_seconds,
            **batch_details,
        )
        _set_request_progress(request_id, "analyze_failed", file_id=file_id, elapsed_seconds=elapsed_seconds)
        request_progress[request_id]["failed"] = True
        logger.exception("[%s] analyze_failed file_id=%s elapsed_seconds=%s", request_id, file_id, elapsed_seconds)
        return _json_error(
            "Backend failed while analyzing the uploaded image.",
            request_id,
            500,
            file_id=file_id,
        )


# ------------------------------
# ส่วนที่ 7: endpoint สำหรับดู progress ของงาน
# endpoint นี้ให้ frontend มาเช็คได้ว่าตอนนี้ backend ทำถึงขั้นไหนแล้ว
# ------------------------------
@app.get("/progress/{request_id}")
def get_request_progress(request_id: str):
    """
    endpoint สำหรับ frontend เรียกเช็คสถานะงานตาม request_id
    เหมาะกับงานที่ใช้เวลานานและต้องการแสดง progress ให้ผู้ใช้เห็น
    """
    progress = request_progress.get(request_id)
    if progress is None:
        return {"success": False, "request_id": request_id, "message": "Progress not found."}
    now = time.time()
    response = {
        "success": True,
        "request_id": request_id,
        "step": progress.get("step"),
        "details": progress.get("details", {}),
        "started_at": progress.get("started_at"),
        "updated_at": progress.get("updated_at"),
        "elapsed_seconds": round(now - float(progress.get("started_at", now)), 1),
        "finished": bool(progress.get("finished", False)),
        "failed": bool(progress.get("failed", False)),
        "status_code": progress.get("status_code"),
    }
    return response
