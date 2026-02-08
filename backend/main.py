from pathlib import Path
from typing import Literal, Optional, Tuple
import base64
import io
import os
import re
import shutil
import subprocess
import uuid

import exifread
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "best.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


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


yolo_model = YOLO(str(MODEL_PATH))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")

EXIFTOOL_DEFAULT_PATHS = [
    r"C:\exiftool\exiftool.exe",
    r"C:\Program Files\ExifTool\exiftool.exe",
    r"C:\Program Files (x86)\ExifTool\exiftool.exe",
    str(Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "ExifTool" / "ExifTool.exe"),
]

DJI_IRP_DEFAULT_PATHS = [
    str(BASE_DIR / "tools" / "dji-tsdk" / "utility" / "bin" / "windows" / "release_x64" / "dji_irp.exe"),
    str(BASE_DIR / "tools" / "dji-tsdk" / "utility" / "bin" / "windows" / "release_x64" / "dji_irp_omp.exe"),
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
        return None, "DJI Thermal SDK (dji_irp.exe) not found."

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

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if not detail:
            detail = "Failed to execute DJI thermal measurement."
        return None, detail

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


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    file_suffix = Path(file.filename or "").suffix.lower()
    if file_suffix not in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}:
        file_suffix = ".jpg"

    uploaded_image_bytes = await file.read()
    uploaded_image_filename = f"{file_id}{file_suffix}"
    uploaded_image_path = UPLOAD_DIR / uploaded_image_filename
    with uploaded_image_path.open("wb") as uploaded_file:
        uploaded_file.write(uploaded_image_bytes)

    image_stream = io.BytesIO(uploaded_image_bytes)
    tags = exifread.process_file(image_stream)

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

    model_results = yolo_model(
        str(uploaded_image_path),
        conf=0.2,
        iou=0.5,
        device=YOLO_DEVICE,
    )

    detection_boxes = []
    if model_results and model_results[0].boxes is not None:
        detection_boxes = model_results[0].boxes.xyxy.cpu().numpy()

    with Image.open(uploaded_image_path) as source_image:
        annotated_image = source_image.convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    image_width, image_height = annotated_image.size

    thermal_matrix, thermal_error, thermal_mode = extract_thermal_matrix(
        str(uploaded_image_path),
        expected_width=image_width,
        expected_height=image_height,
    )

    thermal_analysis_matrix = None
    has_absolute_temperature = False
    thermal_height, thermal_width = 0, 0
    if thermal_matrix is not None:
        if thermal_mode == "absolute":
            if float(np.nanmax(thermal_matrix)) > 1000.0:
                thermal_analysis_matrix = thermal_matrix * 0.04 - 273.15
            else:
                thermal_analysis_matrix = thermal_matrix
            has_absolute_temperature = True
        else:
            thermal_analysis_matrix = thermal_matrix
        thermal_height, thermal_width = thermal_analysis_matrix.shape

    detections = []

    for detection_box in detection_boxes:
        box_x1, box_y1, box_x2, box_y2 = map(int, detection_box)
        box_x1, box_y1, box_x2, box_y2 = _safe_bbox(
            box_x1,
            box_y1,
            box_x2,
            box_y2,
            image_width,
            image_height,
        )

        detection = {
            "bbox": [box_x1, box_y1, box_x2, box_y2],
            "max_temp": None,
            "min_temp": None,
            "avg_temp": None,
            "max_point": None,
            "min_point": None,
            "max_raw": None,
            "min_raw": None,
            "avg_raw": None,
        }

        draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline="orange", width=3)

        if thermal_analysis_matrix is not None:
            thermal_x1 = int(np.floor(box_x1 * thermal_width / image_width))
            thermal_x2 = int(np.ceil(box_x2 * thermal_width / image_width))
            thermal_y1 = int(np.floor(box_y1 * thermal_height / image_height))
            thermal_y2 = int(np.ceil(box_y2 * thermal_height / image_height))

            thermal_x1 = max(0, min(thermal_x1, thermal_width - 1))
            thermal_y1 = max(0, min(thermal_y1, thermal_height - 1))
            thermal_x2 = max(thermal_x1 + 1, min(thermal_x2, thermal_width))
            thermal_y2 = max(thermal_y1 + 1, min(thermal_y2, thermal_height))

            thermal_region = thermal_analysis_matrix[thermal_y1:thermal_y2, thermal_x1:thermal_x2]
            if thermal_region.size > 0:
                max_value = float(thermal_region.max())
                min_value = float(thermal_region.min())
                avg_value = float(thermal_region.mean())

                max_position = np.unravel_index(thermal_region.argmax(), thermal_region.shape)
                min_position = np.unravel_index(thermal_region.argmin(), thermal_region.shape)

                max_point_x = int((thermal_x1 + max_position[1]) * image_width / thermal_width)
                max_point_y = int((thermal_y1 + max_position[0]) * image_height / thermal_height)
                min_point_x = int((thermal_x1 + min_position[1]) * image_width / thermal_width)
                min_point_y = int((thermal_y1 + min_position[0]) * image_height / thermal_height)

                draw.ellipse([max_point_x - 4, max_point_y - 4, max_point_x + 4, max_point_y + 4], fill="red")
                draw.ellipse([min_point_x - 4, min_point_y - 4, min_point_x + 4, min_point_y + 4], fill="blue")
                detection["max_point"] = [max_point_x, max_point_y]
                detection["min_point"] = [min_point_x, min_point_y]
                if has_absolute_temperature:
                    draw.text(
                        (box_x1, max(0, box_y1 - 15)),
                        f"max {max_value:.1f}C min {min_value:.1f}C avg {avg_value:.1f}C",
                        fill="white",
                    )
                    detection["max_temp"] = max_value
                    detection["min_temp"] = min_value
                    detection["avg_temp"] = avg_value
                else:
                    detection["max_raw"] = max_value
                    detection["min_raw"] = min_value
                    detection["avg_raw"] = avg_value

        detections.append(detection)

    annotated_buffer = io.BytesIO()
    annotated_image.save(annotated_buffer, format="JPEG")
    annotated_image_base64 = base64.b64encode(annotated_buffer.getvalue()).decode("ascii")
    annotated_image_data_url = f"data:image/jpeg;base64,{annotated_image_base64}"

    response = {
        "success": True,
        "uploaded_image": f"/uploads/{uploaded_image_filename}",
        "annotated_image": annotated_image_data_url,
        "detections": detections,
        "has_gps": has_gps,
        "message": None,
        "thermal_available": has_absolute_temperature,
        "thermal_mode": thermal_mode,
        "thermal_error": thermal_error,
    }

    if has_gps:
        response["latitude"] = latitude
        response["longitude"] = longitude
    else:
        response["message"] = "No GPS data found in image"

    return response
