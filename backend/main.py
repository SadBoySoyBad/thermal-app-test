from pathlib import Path
from typing import Any, Literal, Optional, Tuple
import gc
import io
import logging
import os
import re
import shutil
import subprocess
import time
import uuid

import exifread
import numpy as np
from fastapi import FastAPI, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from starlette.requests import ClientDisconnect
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DEFAULT_HOTSPOT_MODEL_PATH = BASE_DIR / "model" / "best.pt"
DEFAULT_EQUIPMENT_MODEL_PATH = BASE_DIR / "model" / "equipment.pt"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
EQUIPMENT_LABELS = {
    0: "inverter",
    1: "transformer",
    2: "conductor",
    3: "connector",
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("thermal_app")


@app.middleware("http")
async def log_request_lifecycle(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    started_at = time.perf_counter()
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
        logger.exception("[%s] http_request_failed elapsed_seconds=%s", request_id, elapsed_seconds)
        raise

    response.headers["x-request-id"] = request_id
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(
        "[%s] http_request_finished method=%s path=%s status=%s elapsed_seconds=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_seconds,
    )
    return response


def _env_float(name: str, default_value: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        return float(raw_value)
    except ValueError:
        return default_value


def _env_int(name: str, default_value: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def _resolve_model_path(raw_path: str, default_path: Path) -> Path:
    candidate = Path(raw_path).expanduser() if raw_path.strip() else default_path
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def _load_cors_origins() -> list[str]:
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


YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
HOTSPOT_CONFIDENCE = _env_float("HOTSPOT_CONFIDENCE", 0.2)
HOTSPOT_IOU = _env_float("HOTSPOT_IOU", 0.5)
EQUIPMENT_CONFIDENCE = _env_float("EQUIPMENT_CONFIDENCE", 0.2)
EQUIPMENT_IOU = _env_float("EQUIPMENT_IOU", 0.5)
THERMAL_CENTER_SHIFT_X = _env_int("THERMAL_CENTER_SHIFT_X", -10)
THERMAL_CENTER_SHIFT_Y = _env_int("THERMAL_CENTER_SHIFT_Y", -1)
EQUIPMENT_BBOX_DILATION = _env_int("EQUIPMENT_BBOX_DILATION", 12)
MATCH_DISTANCE_THRESHOLD = _env_float("MATCH_DISTANCE_THRESHOLD", 40.0)
REFERENCE_TEMP_MAX_C = _env_float("REFERENCE_TEMP_MAX_C", 28.0)
RGB_DETECTION_MAX_DIM = _env_int("RGB_DETECTION_MAX_DIM", 1600)
HOTSPOT_MODEL_PATH = _resolve_model_path(os.getenv("HOTSPOT_MODEL_PATH", ""), DEFAULT_HOTSPOT_MODEL_PATH)
EQUIPMENT_MODEL_PATH = _resolve_model_path(os.getenv("EQUIPMENT_MODEL_PATH", ""), DEFAULT_EQUIPMENT_MODEL_PATH)


def _load_yolo_model(model_path: Path, required: bool) -> Optional[YOLO]:
    if not model_path.exists():
        if required:
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        return None
    return YOLO(str(model_path))


logger.info(
    "models_config hotspot_model=%s hotspot_exists=%s equipment_model=%s equipment_exists=%s",
    HOTSPOT_MODEL_PATH,
    HOTSPOT_MODEL_PATH.exists(),
    EQUIPMENT_MODEL_PATH,
    EQUIPMENT_MODEL_PATH.exists(),
)

EXIFTOOL_DEFAULT_PATHS = [
    r"C:\exiftool\exiftool.exe",
    r"C:\Program Files\ExifTool\exiftool.exe",
    r"C:\Program Files (x86)\ExifTool\exiftool.exe",
    str(Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "ExifTool" / "ExifTool.exe"),
]

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

KNOWN_THERMAL_SIZES = [
    (512, 640),
    (256, 320),
]


def dms_to_decimal(dms, ref):
    d = float(dms.values[0])
    m = float(dms.values[1])
    s = float(dms.values[2])

    decimal = d + (m / 60) + (s / 3600)
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal


def resolve_exiftool_path() -> Optional[str]:
    env_path = os.getenv("EXIFTOOL_PATH")
    candidates = [env_path] if env_path else []
    candidates.extend(EXIFTOOL_DEFAULT_PATHS)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return shutil.which("exiftool") or shutil.which("exiftool.exe")


EXIFTOOL_PATH = resolve_exiftool_path()


def resolve_dji_irp_path() -> Optional[str]:
    env_path = os.getenv("DJI_IRP_PATH")
    candidates = [env_path] if env_path else []
    candidates.extend(DJI_IRP_DEFAULT_PATHS)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return shutil.which("dji_irp") or shutil.which("dji_irp.exe")


DJI_IRP_PATH = resolve_dji_irp_path()


def _run_exiftool(args, image_path: str, text: bool = False):
    global EXIFTOOL_PATH
    if not EXIFTOOL_PATH:
        EXIFTOOL_PATH = resolve_exiftool_path()

    if not EXIFTOOL_PATH:
        raise FileNotFoundError(
            "ExifTool not found. Install exiftool and set EXIFTOOL_PATH, or add exiftool to PATH."
        )

    cmd = [EXIFTOOL_PATH, *args, image_path]
    return subprocess.run(cmd, capture_output=True, check=True, text=text)


def _parse_numeric_value(text: str, default_value: float) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    if not match:
        return default_value
    try:
        return float(match.group(0))
    except ValueError:
        return default_value


def _get_dji_measurement_params(image_path: str) -> Tuple[float, float, float, float]:
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


def _extract_dji_temperature_matrix(
    image_path: str,
    expected_width: int,
    expected_height: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
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


def _parse_thermal_from_tiff(payload: bytes) -> Optional[np.ndarray]:
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


def _get_raw_thermal_size(image_path: str) -> Tuple[Optional[int], Optional[int]]:
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


def _extract_binary_tag(image_path: str, tag_name: str) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        result = _run_exiftool(["-b", f"-{tag_name}"], image_path, text=False)
    except FileNotFoundError as exc:
        return None, str(exc)
    except subprocess.CalledProcessError:
        return b"", None

    return result.stdout or b"", None


def _decode_u16_payload(
    payload: bytes,
    expected_width: Optional[int],
    expected_height: Optional[int],
) -> Optional[np.ndarray]:
    if len(payload) % 2 != 0:
        return None

    raw_values = np.frombuffer(payload, dtype="<u2")
    if expected_width and expected_height and raw_values.size == expected_width * expected_height:
        return raw_values.reshape((expected_height, expected_width)).astype(np.float32)

    for known_height, known_width in KNOWN_THERMAL_SIZES:
        if raw_values.size == known_height * known_width:
            return raw_values.reshape((known_height, known_width)).astype(np.float32)

    return None


def extract_thermal_matrix(
    image_path: str,
    expected_width: int,
    expected_height: int,
) -> Tuple[Optional[np.ndarray], Optional[str], Literal["none", "absolute", "relative"]]:
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


def _safe_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int):
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return x1, y1, x2, y2


def _normalize_upload_suffix(filename: str) -> str:
    file_suffix = Path(filename or "").suffix.lower()
    if file_suffix in ALLOWED_IMAGE_SUFFIXES:
        return file_suffix
    return ".jpg"


def _save_upload_bytes(filename: str, file_id: str, label: str, payload: bytes) -> Tuple[str, Path]:
    file_suffix = _normalize_upload_suffix(filename)
    upload_filename = f"{file_id}_{label}{file_suffix}"
    upload_path = UPLOAD_DIR / upload_filename
    with upload_path.open("wb") as uploaded_file:
        uploaded_file.write(payload)
    return upload_filename, upload_path


def _save_upload_file(upload: UploadFile, file_id: str, label: str, payload: bytes) -> Tuple[str, Path]:
    return _save_upload_bytes(upload.filename or "", file_id, label, payload)


def _find_uploaded_file(file_id: str, label: str) -> Optional[Tuple[str, Path]]:
    for suffix in ALLOWED_IMAGE_SUFFIXES:
        candidate = UPLOAD_DIR / f"{file_id}_{label}{suffix}"
        if candidate.exists():
            return candidate.name, candidate
    return None


def _json_error(message: str, request_id: str, status_code: int, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "request_id": request_id,
            **extra,
        },
    )


def _run_yolo_detection(model: YOLO, image_path: Path, conf: float, iou: float) -> list[dict[str, Any]]:
    model_results = model(
        str(image_path),
        conf=conf,
        iou=iou,
        device=YOLO_DEVICE,
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


def _run_yolo_detection_from_path(
    model_path: Path,
    required: bool,
    image_path: Path,
    conf: float,
    iou: float,
) -> list[dict[str, Any]]:
    model = _load_yolo_model(model_path, required=required)
    if model is None:
        return []
    try:
        return _run_yolo_detection(model, image_path, conf, iou)
    finally:
        del model
        gc.collect()


def _prepare_resized_inference_image(
    image_path: Path,
    file_id: str,
    label: str,
    max_dim: int,
) -> Tuple[Path, float, float, Optional[Path], Tuple[int, int], Tuple[int, int]]:
    with Image.open(image_path) as source_image:
        source_width, source_height = source_image.size
        if max_dim <= 0 or max(source_width, source_height) <= max_dim:
            return image_path, 1.0, 1.0, None, (source_width, source_height), (source_width, source_height)

        scale = max_dim / float(max(source_width, source_height))
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized_image = source_image.convert("RGB").resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    resized_path = UPLOAD_DIR / f"{file_id}_{label}_detect.jpg"
    resized_image.save(resized_path, format="JPEG", quality=90)
    resized_image.close()
    return (
        resized_path,
        source_width / float(resized_width),
        source_height / float(resized_height),
        resized_path,
        (source_width, source_height),
        (resized_width, resized_height),
    )


def _centered_thermal_offset(
    rgb_width: int,
    rgb_height: int,
    thermal_width: int,
    thermal_height: int,
) -> Tuple[int, int]:
    offset_x = int(np.floor((rgb_width - thermal_width) / 2.0)) + THERMAL_CENTER_SHIFT_X
    offset_y = int(np.floor((rgb_height - thermal_height) / 2.0)) + THERMAL_CENTER_SHIFT_Y
    return offset_x, offset_y


def _project_thermal_point_to_rgb(
    point_x: float,
    point_y: float,
    thermal_width: int,
    thermal_height: int,
    rgb_width: int,
    rgb_height: int,
) -> Tuple[int, int]:
    offset_x, offset_y = _centered_thermal_offset(rgb_width, rgb_height, thermal_width, thermal_height)
    rgb_point_x = int(round(point_x + offset_x))
    rgb_point_y = int(round(point_y + offset_y))
    rgb_point_x = max(0, min(rgb_point_x, rgb_width - 1))
    rgb_point_y = max(0, min(rgb_point_y, rgb_height - 1))
    return rgb_point_x, rgb_point_y


def _project_thermal_bbox_to_rgb(
    bbox: Tuple[int, int, int, int],
    thermal_width: int,
    thermal_height: int,
    rgb_width: int,
    rgb_height: int,
) -> Tuple[int, int, int, int]:
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


def _dilate_bbox(
    bbox: Tuple[int, int, int, int],
    dilation: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    return _safe_bbox(
        bbox[0] - dilation,
        bbox[1] - dilation,
        bbox[2] + dilation,
        bbox[3] + dilation,
        image_width,
        image_height,
    )


def _bbox_contains_point(bbox: Tuple[int, int, int, int], point_x: int, point_y: int) -> bool:
    return bbox[0] <= point_x <= bbox[2] and bbox[1] <= point_y <= bbox[3]


def _distance_to_bbox(bbox: Tuple[int, int, int, int], point_x: int, point_y: int) -> float:
    delta_x = max(bbox[0] - point_x, 0, point_x - bbox[2])
    delta_y = max(bbox[1] - point_y, 0, point_y - bbox[3])
    return float(np.hypot(delta_x, delta_y))


def _equipment_label_for_class(class_id: int) -> str:
    return EQUIPMENT_LABELS.get(class_id, f"class_{class_id}")


def _match_equipment(
    hotspot_center: Tuple[int, int],
    equipments: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
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


def _compute_reference_temperature(thermal_matrix: np.ndarray) -> Optional[float]:
    finite_values = thermal_matrix[np.isfinite(thermal_matrix)]
    if finite_values.size == 0:
        return None
    reference_pixels = finite_values[finite_values <= REFERENCE_TEMP_MAX_C]
    if reference_pixels.size == 0:
        return None
    return float(reference_pixels.mean())


def _classify_priority(delta_above_reference: float) -> Tuple[str, str]:
    if delta_above_reference > 40.0:
        return "Priority 1", "Immediate repair"
    if delta_above_reference >= 21.0:
        return "Priority 2", "Schedule ASAP"
    if delta_above_reference >= 11.0:
        return "Priority 3", "Plan repair"
    if delta_above_reference >= 1.0:
        return "Priority 4", "Monitor"
    return "Normal", "No action required"


def _log_upload_step(request_id: str, step: str, **details: Any) -> None:
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    if detail_text:
        logger.info("[%s] %s %s", request_id, step, detail_text)
    else:
        logger.info("[%s] %s", request_id, step)


def _is_uploaded_file(value: Any) -> bool:
    return value is not None and hasattr(value, "read") and hasattr(value, "filename")


def _analyze_saved_pair(
    request_id: str,
    started_at: float,
    file_id: str,
    thermal_uploaded_image_filename: str,
    thermal_uploaded_image_path: Path,
    rgb_uploaded_image_filename: str,
    rgb_uploaded_image_path: Path,
):
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

    _log_upload_step(request_id, "thermal_image_probe_started")
    with Image.open(thermal_uploaded_image_path) as thermal_source_image:
        thermal_image_width, thermal_image_height = thermal_source_image.size
    _log_upload_step(
        request_id,
        "thermal_image_probe_finished",
        thermal_size=f"{thermal_image_width}x{thermal_image_height}",
    )

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

    _log_upload_step(request_id, "thermal_model_started", image_path=thermal_uploaded_image_filename)
    hotspot_predictions = _run_yolo_detection_from_path(
        HOTSPOT_MODEL_PATH,
        True,
        thermal_uploaded_image_path,
        HOTSPOT_CONFIDENCE,
        HOTSPOT_IOU,
    )
    _log_upload_step(request_id, "thermal_model_done", hotspot_count=len(hotspot_predictions))

    rgb_detection_path, rgb_scale_x, rgb_scale_y, rgb_temp_path, rgb_original_size, rgb_detection_size = (
        _prepare_resized_inference_image(
            rgb_uploaded_image_path,
            file_id,
            "rgb",
            RGB_DETECTION_MAX_DIM,
        )
    )
    _log_upload_step(
        request_id,
        "rgb_model_started",
        image_path=rgb_detection_path.name,
        original_size=f"{rgb_original_size[0]}x{rgb_original_size[1]}",
        detect_size=f"{rgb_detection_size[0]}x{rgb_detection_size[1]}",
        resized=rgb_temp_path is not None,
    )
    try:
        equipment_predictions = _run_yolo_detection_from_path(
            EQUIPMENT_MODEL_PATH,
            False,
            rgb_detection_path,
            EQUIPMENT_CONFIDENCE,
            EQUIPMENT_IOU,
        )
    finally:
        if rgb_temp_path is not None and rgb_temp_path.exists():
            rgb_temp_path.unlink()
    if rgb_scale_x != 1.0 or rgb_scale_y != 1.0:
        for equipment_prediction in equipment_predictions:
            equipment_prediction["bbox"] = [
                float(equipment_prediction["bbox"][0] * rgb_scale_x),
                float(equipment_prediction["bbox"][1] * rgb_scale_y),
                float(equipment_prediction["bbox"][2] * rgb_scale_x),
                float(equipment_prediction["bbox"][3] * rgb_scale_y),
            ]
    _log_upload_step(request_id, "rgb_model_done", equipment_count=len(equipment_predictions))

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

    thermal_analysis_matrix = None
    has_absolute_temperature = False
    thermal_height, thermal_width = 0, 0
    reference_temperature = None
    if thermal_matrix is not None:
        finite_values = thermal_matrix[np.isfinite(thermal_matrix)]
        if thermal_mode == "absolute":
            if finite_values.size > 0 and float(finite_values.max()) > 1000.0:
                thermal_analysis_matrix = thermal_matrix * 0.04 - 273.15
            else:
                thermal_analysis_matrix = thermal_matrix
            has_absolute_temperature = True
            reference_temperature = _compute_reference_temperature(thermal_analysis_matrix)
        else:
            thermal_analysis_matrix = thermal_matrix
        thermal_height, thermal_width = thermal_analysis_matrix.shape
    _log_upload_step(
        request_id,
        "thermal_matrix_ready",
        absolute=has_absolute_temperature,
        reference_temp=round(reference_temperature, 2) if reference_temperature is not None else None,
        matrix_size=f"{thermal_width}x{thermal_height}" if thermal_analysis_matrix is not None else None,
    )

    _log_upload_step(request_id, "annotation_image_open_started")
    with Image.open(thermal_uploaded_image_path) as thermal_source_image:
        annotated_image = thermal_source_image.convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    _log_upload_step(request_id, "annotation_image_open_finished")

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
        rgb_box = _project_thermal_bbox_to_rgb(
            thermal_box,
            thermal_image_width,
            thermal_image_height,
            rgb_image_width,
            rgb_image_height,
        )
        hotspot_center = _project_thermal_point_to_rgb(
            (thermal_box[0] + thermal_box[2]) / 2.0,
            (thermal_box[1] + thermal_box[3]) / 2.0,
            thermal_image_width,
            thermal_image_height,
            rgb_image_width,
            rgb_image_height,
        )

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

        draw.rectangle(thermal_box, outline="orange", width=3)

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

                max_point_thermal_x = int((thermal_x1 + max_position[1]) * thermal_image_width / thermal_width)
                max_point_thermal_y = int((thermal_y1 + max_position[0]) * thermal_image_height / thermal_height)
                min_point_thermal_x = int((thermal_x1 + min_position[1]) * thermal_image_width / thermal_width)
                min_point_thermal_y = int((thermal_y1 + min_position[0]) * thermal_image_height / thermal_height)

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
                    if reference_temperature is not None:
                        delta_above_reference = max_value - reference_temperature
                        priority, action_required = _classify_priority(delta_above_reference)
                        detection["delta_above_reference"] = delta_above_reference
                        detection["priority"] = priority
                        detection["action_required"] = action_required
                else:
                    detection["max_raw"] = max_value
                    detection["min_raw"] = min_value
                    detection["avg_raw"] = avg_value

        detection.update(_match_equipment(hotspot_center, equipments, rgb_image_width, rgb_image_height))
        detections.append(detection)

    _log_upload_step(request_id, "matching_done", detection_count=len(detections))

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

    if has_gps:
        response["latitude"] = latitude
        response["longitude"] = longitude
    else:
        response["message"] = "No GPS data found in thermal image"

    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    _log_upload_step(request_id, "upload_completed", elapsed_seconds=elapsed_seconds)
    return response


@app.post("/upload")
async def upload_image(request: Request):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
    started_at = time.perf_counter()

    try:
        _log_upload_step(request_id, "form_parsing_started")
        form = await request.form()
        _log_upload_step(request_id, "form_parsing_finished", fields=",".join(sorted(form.keys())))

        thermal_file = form.get("thermal_file")
        rgb_file = form.get("rgb_file")
        legacy_file = form.get("file")

        if thermal_file is None and legacy_file is not None:
            thermal_file = legacy_file

        if not _is_uploaded_file(thermal_file):
            thermal_file = None
        if not _is_uploaded_file(rgb_file):
            rgb_file = None

        _log_upload_step(
            request_id,
            "upload_started",
            thermal_name=getattr(thermal_file, "filename", None),
            rgb_name=getattr(rgb_file, "filename", None),
        )

        if thermal_file is None:
            _log_upload_step(request_id, "upload_rejected", reason="missing_thermal")
            return {"success": False, "message": "Thermal image is required.", "request_id": request_id}

        if rgb_file is None:
            _log_upload_step(request_id, "upload_rejected", reason="missing_rgb")
            return {
                "success": False,
                "message": "RGB image is required to identify equipment.",
                "request_id": request_id,
            }

        if not EQUIPMENT_MODEL_PATH.exists():
            _log_upload_step(request_id, "upload_rejected", reason="missing_equipment_model")
            return {
                "success": False,
                "message": f"Equipment model not found at {EQUIPMENT_MODEL_PATH}. Set EQUIPMENT_MODEL_PATH first.",
                "request_id": request_id,
            }

        file_id = uuid.uuid4().hex
        thermal_bytes = await thermal_file.read()
        rgb_bytes = await rgb_file.read()
        _log_upload_step(
            request_id,
            "files_read",
            thermal_bytes=len(thermal_bytes),
            rgb_bytes=len(rgb_bytes),
        )

        thermal_uploaded_image_filename, thermal_uploaded_image_path = _save_upload_file(
            thermal_file,
            file_id,
            "thermal",
            thermal_bytes,
        )
        rgb_uploaded_image_filename, rgb_uploaded_image_path = _save_upload_file(
            rgb_file,
            file_id,
            "rgb",
            rgb_bytes,
        )
        _log_upload_step(
            request_id,
            "files_saved",
            thermal_path=thermal_uploaded_image_filename,
            rgb_path=rgb_uploaded_image_filename,
        )
        return _analyze_saved_pair(
            request_id=request_id,
            started_at=started_at,
            file_id=file_id,
            thermal_uploaded_image_filename=thermal_uploaded_image_filename,
            thermal_uploaded_image_path=thermal_uploaded_image_path,
            rgb_uploaded_image_filename=rgb_uploaded_image_filename,
            rgb_uploaded_image_path=rgb_uploaded_image_path,
        )
    except ClientDisconnect:
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        logger.warning("[%s] upload_client_disconnected elapsed_seconds=%s", request_id, elapsed_seconds)
        return _json_error(
            "Upload connection dropped before the backend received all files.",
            request_id,
            499,
        )
    except Exception:
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        logger.exception("[%s] upload_failed elapsed_seconds=%s", request_id, elapsed_seconds)
        return _json_error(
            "Backend failed while processing the upload. Check backend logs with the request ID.",
            request_id,
            500,
        )


@app.post("/upload-file")
async def upload_file_raw(request: Request):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
    started_at = time.perf_counter()
    kind = (request.query_params.get("kind") or "").strip().lower()
    file_id = (request.query_params.get("file_id") or uuid.uuid4().hex).strip()
    original_name = (request.headers.get("x-file-name") or f"{kind or 'upload'}.jpg").strip()

    if kind not in {"thermal", "rgb"}:
        return _json_error("Upload kind must be 'thermal' or 'rgb'.", request_id, 400)

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
        logger.exception("[%s] raw_upload_failed kind=%s elapsed_seconds=%s", request_id, kind, elapsed_seconds)
        return _json_error(
            "Backend failed while receiving the uploaded file.",
            request_id,
            500,
            file_id=file_id,
            kind=kind,
        )


@app.post("/analyze")
async def analyze_uploaded_pair(request: Request):
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
        logger.exception("[%s] analyze_failed file_id=%s elapsed_seconds=%s", request_id, file_id, elapsed_seconds)
        return _json_error(
            "Backend failed while analyzing the uploaded image pair.",
            request_id,
            500,
            file_id=file_id,
        )
