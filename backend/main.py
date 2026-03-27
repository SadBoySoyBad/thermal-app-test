# คู่มืออ่านไฟล์ main.py นี้:
# - ส่วนต้นไฟล์: ตั้งค่าระบบ, path, env, โมเดล, log
# - ส่วนกลางไฟล์: ฟังก์ชันช่วยคำนวณ/อ่านข้อมูลภาพ/วิเคราะห์
# - ส่วนท้ายไฟล์: API endpoint ที่ frontend เรียกใช้งาน
#
# ถ้าเริ่มอ่านครั้งแรก แนะนำลำดับนี้:
# 1) อ่านค่าคงที่และ config ด้านบน
# 2) อ่านฟังก์ชัน `_analyze_saved_pair()` (แกนหลักของ pipeline)
# 3) อ่าน endpoint `/upload-file`, `/analyze`, `/progress`
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
# ============================================================

from pathlib import Path
# [เพิ่ม Any]
# ใช้รองรับ dict / function ที่รับค่าหลายชนิด
# จากเดิมมีแค่ Literal, Optional, Tuple
from typing import Any, Literal, Optional, Tuple

# [เพิ่ม gc]
# ใช้ช่วยเก็บกวาดหน่วยความจำหลังโหลดโมเดลหรือประมวลผลเสร็จ
import gc

import io

# [เพิ่ม logging]
# ใช้ทำ log ฝั่ง backend เพื่อ debug ได้ง่ายขึ้น โดยเฉพาะบน server จริง
import logging

import os
import re
import shutil
import subprocess

# [เพิ่ม time]
# ใช้วัดเวลา request และ progress แต่ละขั้น
import time

import uuid

import exifread
import numpy as np

# [แก้ import FastAPI]
# ใช้ Request เป็นหลัก เพราะโค้ดอ่าน json/stream และ header/query จาก request ตรง ๆ
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# [เพิ่ม JSONResponse]
# ใช้คืน error response แบบกำหนด status code ได้ชัดเจน
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

# [เพิ่ม ClientDisconnect]
# ใช้จับกรณี client หลุดระหว่างอัปโหลดไฟล์
from starlette.requests import ClientDisconnect

# [เพิ่ม torch]
# ใช้คุมจำนวน threads ของ PyTorch และช่วยเรื่อง resource
import torch

from ultralytics import YOLO


# ------------------------------
# path และค่าคงที่พื้นฐานของโปรเจกต์
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

# [แก้จากเดิม]
# เดิมมี model เดียวคือ best.pt สำหรับ hotspot
# ใหม่แยก default model path สำหรับ hotspot และ equipment
DEFAULT_HOTSPOT_MODEL_PATH = BASE_DIR / "model" / "best.pt"
DEFAULT_EQUIPMENT_MODEL_PATH = BASE_DIR / "model" / "equipment.pt"

# [เพิ่มใหม่]
# จำกัดนามสกุลไฟล์ภาพที่ระบบยอมรับ
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

# [เพิ่มใหม่]
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
# [เพิ่มใหม่ทั้งหมด]
# ระบบ logging ของ backend
# ------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("thermal_app")

# ------------------------------
# [เพิ่มใหม่ทั้งหมด]
# ใช้เก็บ progress ของแต่ละ request
# frontend สามารถ poll มาดูสถานะได้จาก /progress/{request_id}
# ------------------------------
request_progress: dict[str, dict[str, Any]] = {}

# อายุสูงสุดของ progress แต่ละรายการ
PROGRESS_TTL_SECONDS = 60 * 60

# จำนวน entry สูงสุดที่ยอมให้เก็บใน memory
MAX_PROGRESS_ENTRIES = 256


# ------------------------------
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่ทั้งหมด]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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


# ------------------------------
# [เพิ่มใหม่]
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
# โหลดรายการ CORS origins
# โครงหลักยังเหมือนโค้ดเก่า
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
# [ขยายจากเดิม]
# config ต่าง ๆ ของระบบ อ่านจาก env ได้
# ทำให้ปรับ threshold / imgsz / alignment ได้โดยไม่ต้องแก้ source code
# ------------------------------
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
HOTSPOT_CONFIDENCE = _env_float("HOTSPOT_CONFIDENCE", 0.2)
HOTSPOT_IOU = _env_float("HOTSPOT_IOU", 0.5)
EQUIPMENT_CONFIDENCE = _env_float("EQUIPMENT_CONFIDENCE", 0.2)
EQUIPMENT_IOU = _env_float("EQUIPMENT_IOU", 0.5)

# [เพิ่มใหม่]
# ใช้ปรับตำแหน่ง thermal overlay บน RGB
THERMAL_CENTER_SHIFT_X = _env_int("THERMAL_CENTER_SHIFT_X", -10)
THERMAL_CENTER_SHIFT_Y = _env_int("THERMAL_CENTER_SHIFT_Y", -1)

# [เพิ่มใหม่]
# ใช้ขยาย bbox ของ equipment เพื่อช่วยการ match
EQUIPMENT_BBOX_DILATION = _env_int("EQUIPMENT_BBOX_DILATION", 12)

# [เพิ่มใหม่]
# ระยะ threshold สูงสุดสำหรับ match แบบ nearest
MATCH_DISTANCE_THRESHOLD = _env_float("MATCH_DISTANCE_THRESHOLD", 40.0)

# [เพิ่มใหม่]
# ค่าสูงสุดของ pixel ที่จะนับเป็น reference temperature
REFERENCE_TEMP_MAX_C = _env_float("REFERENCE_TEMP_MAX_C", 28.0)

# [เพิ่มใหม่]
# จำกัดขนาดรูป RGB ตอน detect และ margin ตอน crop
RGB_DETECTION_MAX_DIM = _env_int("RGB_DETECTION_MAX_DIM", 1600)
RGB_DETECTION_CROP_MARGIN = _env_int("RGB_DETECTION_CROP_MARGIN", 120)

# [เพิ่มใหม่]
# imgsz แยกของ hotspot / equipment model
HOTSPOT_IMGSZ = _env_int("HOTSPOT_IMGSZ", 640)
EQUIPMENT_IMGSZ = _env_int("EQUIPMENT_IMGSZ", 640)

# [เพิ่มใหม่]
# จำนวน threads ของ PyTorch
TORCH_NUM_THREADS = max(1, _env_int("TORCH_NUM_THREADS", 1))
TORCH_INTEROP_THREADS = max(1, _env_int("TORCH_INTEROP_THREADS", 1))

HOTSPOT_MODEL_PATH = _resolve_model_path(os.getenv("HOTSPOT_MODEL_PATH", ""), DEFAULT_HOTSPOT_MODEL_PATH)
EQUIPMENT_MODEL_PATH = _resolve_model_path(os.getenv("EQUIPMENT_MODEL_PATH", ""), DEFAULT_EQUIPMENT_MODEL_PATH)


# ------------------------------
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
    "models_config hotspot_model=%s hotspot_exists=%s equipment_model=%s equipment_exists=%s hotspot_imgsz=%s equipment_imgsz=%s torch_threads=%s torch_interop_threads=%s",
    HOTSPOT_MODEL_PATH,
    HOTSPOT_MODEL_PATH.exists(),
    EQUIPMENT_MODEL_PATH,
    EQUIPMENT_MODEL_PATH.exists(),
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
# รัน YOLO บนภาพ 1 ภาพ แล้วคืนผลลัพธ์เป็น list ของ detection dict
# ใช้ได้ทั้ง hotspot model และ equipment model
# ------------------------------
def _run_yolo_detection(model: YOLO, image_path: Path, conf: float, iou: float, imgsz: int) -> list[dict[str, Any]]:
    """
    รันโมเดล YOLO กับภาพ 1 ใบ และแปลงผลลัพธ์เป็น dict แบบเรียบง่าย:
    bbox, confidence, class_id
    """
    model_results = model(
        str(image_path),
        conf=conf,
        iou=iou,
        device=YOLO_DEVICE,
        imgsz=max(32, imgsz),
    )

    detections: list[dict[str, Any]] = []
    if not model_results or model_results[0].boxes is None:
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

    return detections


# ------------------------------
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
# เช็คว่าจุดอยู่ใน bbox หรือไม่
# ------------------------------
def _bbox_contains_point(bbox: Tuple[int, int, int, int], point_x: int, point_y: int) -> bool:
    """
    ตรวจว่าจุด (x, y) อยู่ภายในกรอบ bbox หรือไม่
    """
    return bbox[0] <= point_x <= bbox[2] and bbox[1] <= point_y <= bbox[3]


# ------------------------------
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
# แปลง class id -> ชื่ออุปกรณ์
# ------------------------------
def _equipment_label_for_class(class_id: int) -> str:
    """
    แปลง class_id จากโมเดลเป็นชื่ออุปกรณ์ที่มนุษย์อ่านง่าย
    """
    return EQUIPMENT_LABELS.get(class_id, f"class_{class_id}")


# ------------------------------
# [เพิ่มใหม่ทั้งหมด]
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
# [เพิ่มใหม่]
# คำนวณ reference temperature ของทั้งภาพ
# โดยใช้ pixel ที่ <= REFERENCE_TEMP_MAX_C
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
# [เพิ่มใหม่]
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
# [เพิ่มใหม่]
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


# ------------------------------
# [เพิ่มใหม่ทั้งหมด]
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
    rgb_uploaded_image_filename: str,
    rgb_uploaded_image_path: Path,
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

    # อ่านขนาดภาพ RGB
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
        rgb_size=f"{rgb_image_width}x{rgb_image_height}",
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
    rgb_overlay_bbox = _thermal_overlay_bbox_on_rgb(
        thermal_image_width,
        thermal_image_height,
        rgb_image_width,
        rgb_image_height,
    )
    rgb_crop_bbox = _dilate_bbox(
        rgb_overlay_bbox,
        RGB_DETECTION_CROP_MARGIN,
        rgb_image_width,
        rgb_image_height,
    )
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
        annotated_image = thermal_source_image.convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
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
        draw.rectangle(thermal_box, outline="orange", width=3)
        _draw_hotspot_index_label(
            draw=draw,
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
                draw.ellipse(
                    [
                        max_point_thermal_x - 4,
                        max_point_thermal_y - 4,
                        max_point_thermal_x + 4,
                        max_point_thermal_y + 4,
                    ],
                    fill="red",
                )
                draw.ellipse(
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
                    draw.text(
                        (thermal_box[0], max(0, thermal_box[1] - 15)),
                        f"max {max_value:.1f}C min {min_value:.1f}C avg {avg_value:.1f}C",
                        fill="white",
                    )
                    detection["max_temp"] = max_value
                    detection["min_temp"] = min_value
                    detection["avg_temp"] = avg_value

                    # [เพิ่มใหม่]
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

        # [เพิ่มใหม่]
        # match hotspot จุดนี้กับ equipment บน RGB
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
    annotated_image.save(annotated_image_path, format="JPEG", quality=90)
    annotated_image.close()
    gc.collect()

    _log_upload_step(
        request_id,
        "annotated_image_saved",
        annotated_path=annotated_image_filename,
    )

    response = {
        "success": True,
        "uploaded_image": f"/uploads/{thermal_uploaded_image_filename}",
        "uploaded_rgb_image": f"/uploads/{rgb_uploaded_image_filename}",
        "annotated_image": f"/uploads/{annotated_image_filename}",
        "detections": detections,
        "has_gps": has_gps,
        "message": None,
        "thermal_available": has_absolute_temperature,
        "thermal_mode": thermal_mode,
        "thermal_error": thermal_error,
        "reference_temperature": reference_temperature,
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
# [เพิ่มใหม่ทั้งหมด]
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

    if kind not in {"thermal", "rgb"}:
        return _json_error("Upload kind must be 'thermal' or 'rgb'.", request_id, 400)

    # สร้าง path ปลายทางก่อน แล้วเขียนทับด้วย stream ทีละ chunk
    upload_filename, upload_path = _save_upload_bytes(original_name, file_id, kind, b"")
    bytes_written = 0

    try:
        _log_upload_step(
            request_id,
            "raw_upload_started",
            file_id=file_id,
            kind=kind,
            filename=original_name,
            content_length=request.headers.get("content-length"),
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
        _log_upload_step(
            request_id,
            "raw_upload_finished",
            file_id=file_id,
            kind=kind,
            saved_path=upload_filename,
            bytes_written=bytes_written,
            elapsed_seconds=elapsed_seconds,
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
# [เพิ่มใหม่ทั้งหมด]
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

    try:
        payload = await request.json()
    except Exception:
        return _json_error("Analyze request must be valid JSON.", request_id, 400)

    file_id = str(payload.get("file_id") or "").strip()
    if not file_id:
        return _json_error("Analyze request requires file_id.", request_id, 400)

    if not EQUIPMENT_MODEL_PATH.exists():
        return _json_error(
            f"Equipment model not found at {EQUIPMENT_MODEL_PATH}. Set EQUIPMENT_MODEL_PATH first.",
            request_id,
            500,
        )

    thermal_file = _find_uploaded_file(file_id, "thermal")
    rgb_file = _find_uploaded_file(file_id, "rgb")

    if thermal_file is None:
        return _json_error("Thermal upload not found for the requested file_id.", request_id, 404, file_id=file_id)
    if rgb_file is None:
        return _json_error("RGB upload not found for the requested file_id.", request_id, 404, file_id=file_id)

    thermal_uploaded_image_filename, thermal_uploaded_image_path = thermal_file
    rgb_uploaded_image_filename, rgb_uploaded_image_path = rgb_file

    _log_upload_step(
        request_id,
        "analyze_started",
        file_id=file_id,
        thermal_path=thermal_uploaded_image_filename,
        rgb_path=rgb_uploaded_image_filename,
    )

    try:
        return _analyze_saved_pair(
            request_id=request_id,
            started_at=started_at,
            file_id=file_id,
            thermal_uploaded_image_filename=thermal_uploaded_image_filename,
            thermal_uploaded_image_path=thermal_uploaded_image_path,
            rgb_uploaded_image_filename=rgb_uploaded_image_filename,
            rgb_uploaded_image_path=rgb_uploaded_image_path,
        )
    except Exception:
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        _set_request_progress(request_id, "analyze_failed", file_id=file_id, elapsed_seconds=elapsed_seconds)
        request_progress[request_id]["failed"] = True
        logger.exception("[%s] analyze_failed file_id=%s elapsed_seconds=%s", request_id, file_id, elapsed_seconds)
        return _json_error(
            "Backend failed while analyzing the uploaded image pair.",
            request_id,
            500,
            file_id=file_id,
        )


# ------------------------------
# [เพิ่มใหม่ทั้งหมด]
# endpoint ให้ frontend มาดู progress ของ request
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
