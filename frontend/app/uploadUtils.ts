import type { MapMarkerItem } from "./MapView";

// ไฟล์นี้เป็นไฟล์ช่วยของหน้า upload
// เอาไว้เก็บ type และฟังก์ชันย่อยต่าง ๆ
// เพื่อไม่ให้ไฟล์ page.tsx ยาวและรกเกินไป

// วิธีที่ระบบใช้บอกว่า hotspot ไปตรงกับอุปกรณ์แบบไหน
export type MatchMethod = "inside" | "nearest" | "unknown";

// รูปแบบข้อมูล thermal ที่ได้กลับมา
// none = ไม่มีข้อมูลอุณหภูมิ
// absolute = มีอุณหภูมิจริง
// relative = มีแค่ค่าความร้อนแบบเปรียบเทียบ
export type ThermalMode = "none" | "absolute" | "relative";

// ประเภทไฟล์ที่ระบบมองเห็น
type PairKind = "thermal" | "rgb" | "unknown";

// ข้อมูลของ hotspot / detection 1 จุด
export type Detection = {
  // ตำแหน่งกรอบของจุดที่ตรวจเจอ
  bbox: [number, number, number, number];

  // กรอบของ hotspot บนภาพ thermal
  thermal_bbox?: [number, number, number, number];

  // ความมั่นใจของโมเดลว่าตรงนี้คือ hotspot
  hotspot_confidence?: number | null;

  // จุดกึ่งกลางของ hotspot
  hotspot_center?: [number, number] | null;

  // อุณหภูมิสูงสุด ต่ำสุด และเฉลี่ย
  max_temp: number | null;
  min_temp: number | null;
  avg_temp: number | null;

  // ถ้าไม่มีอุณหภูมิจริง อาจได้ค่า raw มาแทน
  max_raw?: number | null;
  min_raw?: number | null;
  avg_raw?: number | null;

  // จุดที่ร้อนสุด / เย็นสุด
  max_point?: [number, number] | null;
  min_point?: [number, number] | null;

  // ข้อมูลอุปกรณ์ที่จับคู่กับ hotspot ได้
  equipment_class?: string | null;
  equipment_confidence?: number | null;
  equipment_bbox?: [number, number, number, number] | null;

  // ระบบจับคู่วิธีไหน
  match_method?: MatchMethod | null;
  match_distance?: number | null;

  // ค่าอ้างอิงและความต่างจากค่าอ้างอิง
  reference_temp?: number | null;
  delta_above_reference?: number | null;

  // ระดับความสำคัญ และสิ่งที่ควรทำ
  priority?: string | null;
  action_required?: string | null;
};

// คู่ไฟล์ที่จับคู่สำเร็จแล้ว
// 1 คู่ = thermal 1 ไฟล์ + rgb 1 ไฟล์
export type MatchedPair = {
  id: string;
  key: string;
  displayName: string;
  thermal: File;
  rgb: File;
};

// ปัญหาที่พบตอนจับคู่ไฟล์
export type PairingIssue = {
  id: string;
  displayName: string;
  fileNames: string[];
  message: string;
};

// ข้อมูลชั่วคราวของไฟล์แต่ละไฟล์ ก่อนเอาไปจับคู่
type PairCandidate = {
  file: File;
  stem: string; // ชื่อไฟล์แบบไม่มีนามสกุล
  key: string;  // key ที่ใช้จัดกลุ่ม
  kind: PairKind;
};

// คู่ไฟล์ที่วิเคราะห์ไม่ผ่าน
export type FailedPair = {
  id: string;
  displayName: string;
  message: string;
};

// ผลวิเคราะห์ที่ frontend จะเอาไปใช้ต่อ
export type AnalysisResult = {
  id: string;
  key: string;
  displayName: string;
  thermalFileName: string;
  rgbFileName: string;
  annotatedImage: string | null;
  detections: Detection[];
  latitude: number | null;
  longitude: number | null;
  thermalAvailable: boolean | null;
  thermalError: string;
  thermalMode: ThermalMode | null;
  referenceTemperature: number | null;
  message: string;
  requestId: string;
};

// สัญลักษณ์องศาเซลเซียส
export const DEGREE_C = "\u00B0C";

// URL หลักของ backend
// ถ้าไม่ได้ตั้ง env ไว้ จะใช้ localhost:8000
const backendBaseUrl = (process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

// คำที่ระบบมองว่าเป็นไฟล์ thermal
const THERMAL_TOKENS = new Set(["thermal", "therm", "infrared", "infra", "ir", "thm", "temp", "t"]);

// คำที่ระบบมองว่าเป็นไฟล์ rgb / ภาพปกติ
const RGB_TOKENS = new Set(["rgb", "visual", "visible", "wide", "vis", "v", "w"]);

// รวมคำทั้งหมดที่ใช้บอกบทบาทของไฟล์
const ROLE_TOKENS = new Set([...THERMAL_TOKENS, ...RGB_TOKENS]);

// ตัดนามสกุลไฟล์ออก
// เช่น abc.jpg -> abc
function getFileStem(fileName: string) {
  const lastDot = fileName.lastIndexOf(".");
  return lastDot > 0 ? fileName.slice(0, lastDot) : fileName;
}

// แยกชื่อไฟล์ออกเป็นคำย่อย ๆ
// เพื่อเอาไปเดาว่าไฟล์นี้เป็น thermal หรือ rgb
function tokenizeStem(stem: string) {
  const normalized = stem.normalize("NFKC").toLowerCase();
  const sanitized = normalized.replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  return sanitized ? sanitized.split(/\s+/) : [];
}

// ถ้าชื่อไฟล์มีตัวเลข จะเอา เลขตัวท้าย มาใช้เป็น key หลัก
// เช่น DJI_xxx_0035_T.JPG -> key = 0035
function extractPairNumber(tokens: string[]) {
  const numericTokens = tokens.filter((token) => /^\d+$/.test(token));
  return numericTokens.length > 0 ? numericTokens[numericTokens.length - 1] : "";
}

// ดูจากชื่อไฟล์ว่าเป็น thermal หรือ rgb
function detectPairKind(tokens: string[]): PairKind {
  let thermalScore = 0;
  let rgbScore = 0;

  for (const token of tokens) {
    if (THERMAL_TOKENS.has(token)) {
      thermalScore += token.length === 1 ? 1 : 3;
    }
    if (RGB_TOKENS.has(token)) {
      rgbScore += token.length === 1 ? 1 : 3;
    }
  }

  if (thermalScore > rgbScore) {
    return "thermal";
  }
  if (rgbScore > thermalScore) {
    return "rgb";
  }
  return "unknown";
}

// ทำชื่อที่ใช้แสดงผลให้อ่านง่ายขึ้น
function formatDisplayName(key: string, fallback: string) {
  const trimmedKey = key.trim();
  if (!trimmedKey) {
    return fallback;
  }

  return trimmedKey
    .split(/\s+/)
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");
}

// แปลงไฟล์ 1 ไฟล์ ให้เป็นข้อมูลที่พร้อมใช้สำหรับจับคู่
function toPairCandidate(file: File): PairCandidate {
  const stem = getFileStem(file.name);
  const tokens = tokenizeStem(stem);
  const pairNumber = extractPairNumber(tokens);

  // ตัดคำพวก thermal / rgb ออก เพื่อให้เหลือ key กลาง
  const keyTokens = tokens.filter((token) => !ROLE_TOKENS.has(token));

  const normalizedStem = stem.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

  return {
    file,
    stem,
    key: pairNumber || keyTokens.join(" ").trim() || normalizedStem || stem.toLowerCase(),
    kind: detectPairKind(tokens),
  };
}

// อ่านขนาดรูป (กว้าง x สูง)
// ใช้ช่วยเดาว่าไฟล์ไหนน่าจะเป็น thermal / rgb
async function readImageArea(file: File) {
  const objectUrl = URL.createObjectURL(file);

  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const nextImage = new Image();
      nextImage.onload = () => resolve(nextImage);
      nextImage.onerror = () => reject(new Error(`Cannot read ${file.name}`));
      nextImage.src = objectUrl;
    });

    return image.naturalWidth * image.naturalHeight;
  } catch {
    return null;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

// กรณีชื่อไฟล์บอกไม่ชัดว่าอะไรคือ thermal / rgb
// ถ้ามีแค่ 2 ไฟล์ในกลุ่ม จะลองเดาจากขนาดภาพ
// โดยมองว่าภาพที่เล็กกว่าน่าจะเป็น thermal
async function assignUnknownPairByImageSize(group: PairCandidate[]) {
  if (group.length !== 2) {
    return null;
  }

  const withArea = await Promise.all(
    group.map(async (candidate) => ({
      candidate,
      area: await readImageArea(candidate.file),
    })),
  );

  if (withArea.some((item) => item.area === null)) {
    return null;
  }

  const sorted = [...withArea].sort((left, right) => {
    const leftArea = left.area ?? 0;
    const rightArea = right.area ?? 0;
    if (leftArea !== rightArea) {
      return leftArea - rightArea;
    }
    return left.candidate.file.name.localeCompare(right.candidate.file.name);
  });

  // ถ้าขนาดเท่ากัน เดาไม่ได้
  if ((sorted[0].area ?? 0) === (sorted[1].area ?? 0)) {
    return null;
  }

  return {
    thermal: sorted[0].candidate,
    rgb: sorted[1].candidate,
  };
}

// ฟังก์ชันหลักของการจับคู่ไฟล์
// รับไฟล์ทั้งหมดเข้ามา แล้วพยายามจับว่าไฟล์ไหนเป็นคู่กัน
export async function matchUploadPairs(files: File[]) {
  const groups = new Map<string, PairCandidate[]>();
  const pairs: MatchedPair[] = [];
  const issues: PairingIssue[] = [];

  // แปลงทุกไฟล์ให้พร้อมใช้ แล้วเรียงลำดับ
  const candidates = files
    .map(toPairCandidate)
    .sort((left, right) => left.key.localeCompare(right.key) || left.file.name.localeCompare(right.file.name));

  // จัดกลุ่มตาม key
  for (const candidate of candidates) {
    const existing = groups.get(candidate.key) ?? [];
    existing.push(candidate);
    groups.set(candidate.key, existing);
  }

  // ทำทีละกลุ่ม
  for (const [key, group] of groups.entries()) {
    const thermals = group.filter((candidate) => candidate.kind === "thermal");
    const rgbs = group.filter((candidate) => candidate.kind === "rgb");
    const unknowns = group.filter((candidate) => candidate.kind === "unknown");

    // ถ้ายังไม่รู้ทั้งคู่ และมี 2 ไฟล์พอดี ให้เดาจากขนาดรูป
    if (thermals.length === 0 && rgbs.length === 0 && unknowns.length === 2) {
      const guessedPair = await assignUnknownPairByImageSize(unknowns);
      if (guessedPair) {
        thermals.push(guessedPair.thermal);
        rgbs.push(guessedPair.rgb);
        unknowns.length = 0;
      }
    }

    // ถ้ามี unknown 1 ไฟล์ และอีกฝั่งมีอยู่แล้ว 1 ไฟล์
    // ก็เดาให้อีกไฟล์เป็นคู่ตรงข้าม
    if (unknowns.length === 1) {
      if (thermals.length === 0 && rgbs.length === 1) {
        thermals.push(unknowns.shift() as PairCandidate);
      } else if (rgbs.length === 0 && thermals.length === 1) {
        rgbs.push(unknowns.shift() as PairCandidate);
      }
    }

    // เรียงชื่อก่อน เพื่อให้จับคู่ได้คงที่
    thermals.sort((left, right) => left.file.name.localeCompare(right.file.name));
    rgbs.sort((left, right) => left.file.name.localeCompare(right.file.name));

    // จำนวนคู่ที่ทำได้จริง
    const pairCount = Math.min(thermals.length, rgbs.length);

    // สร้างคู่ไฟล์
    for (let index = 0; index < pairCount; index += 1) {
      const thermal = thermals[index];
      const rgb = rgbs[index];
      const fallbackName = thermal?.stem ?? rgb?.stem ?? `Pair ${pairs.length + 1}`;

      pairs.push({
        id: `${key || "pair"}-${index + 1}`,
        key,
        displayName: formatDisplayName(key, fallbackName),
        thermal: thermal.file,
        rgb: rgb.file,
      });
    }

    // ไฟล์ที่เหลือและจับคู่ไม่ได้
    const leftovers = [...thermals.slice(pairCount), ...rgbs.slice(pairCount), ...unknowns];

    if (leftovers.length > 0 || pairCount === 0) {
      issues.push({
        id: key || `issue-${issues.length + 1}`,
        displayName: formatDisplayName(key, leftovers[0]?.stem ?? `Group ${issues.length + 1}`),
        fileNames: group.map((candidate) => candidate.file.name),
        message:
          pairCount === 0
            ? "Unable to auto-match this group. Files with the same running number should be uploaded as one T/V pair."
            : "Some files in this group were skipped because more than one file used the same number.",
      });
    }
  }

  pairs.sort((left, right) => left.displayName.localeCompare(right.displayName));
  issues.sort((left, right) => left.displayName.localeCompare(right.displayName));

  return { pairs, issues };
}

// ถ้า backend ส่ง data URL มาอยู่แล้ว ก็ใช้ตรงนั้นได้เลย
// ถ้าเป็น path ธรรมดา ก็เติม backend URL ข้างหน้า
function toAbsoluteImageUrl(rawImagePath: string) {
  return rawImagePath.startsWith("data:") ? rawImagePath : `${backendBaseUrl}${rawImagePath}`;
}

// สร้าง request id ไว้ติดตามงาน
export function createRequestId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  }
  return Math.random().toString(36).slice(2, 10);
}

// เลือก request id ที่จะใช้จริง
// ถ้า backend ส่ง request_id กลับมา ก็ใช้ของ backend
// ถ้าไม่ส่ง ก็ใช้ค่าเดิมจาก header
export function getResponseRequestId(responseData: unknown, headerRequestId: string) {
  if (
    typeof responseData === "object" &&
    responseData !== null &&
    "request_id" in responseData &&
    typeof responseData.request_id === "string" &&
    responseData.request_id.trim()
  ) {
    return responseData.request_id;
  }
  return headerRequestId;
}

// แปลชื่อ step จาก backend ให้เป็นข้อความที่คนอ่านเข้าใจง่าย
export function describeBackendStep(step: string | null | undefined, details: Record<string, unknown> | null | undefined) {
  switch (step) {
    case "raw_upload_started":
      return details?.kind === "rgb" ? "Uploading RGB image..." : "Uploading thermal image...";
    case "raw_upload_finished":
      return details?.kind === "rgb" ? "RGB image uploaded." : "Thermal image uploaded.";
    case "analyze_started":
      return "Analysis request accepted by backend.";
    case "gps_checked":
      return "Reading GPS metadata from thermal image...";
    case "thermal_image_probe_started":
      return "Opening thermal image...";
    case "thermal_image_probe_finished":
      return "Thermal image opened.";
    case "rgb_image_probe_started":
      return "Opening RGB image...";
    case "rgb_image_probe_finished":
      return "RGB image opened.";
    case "images_opened":
      return "Image sizes ready. Preparing model inference...";
    case "thermal_model_started":
      return "Running thermal hotspot model...";
    case "thermal_model_done":
      return "Thermal hotspot model finished.";
    case "rgb_model_started":
      return "Running RGB equipment model...";
    case "rgb_model_done":
      return "RGB equipment model finished.";
    case "thermal_extraction_done":
      return "Thermal temperature data extracted.";
    case "thermal_matrix_ready":
      return "Thermal matrix ready.";
    case "annotation_image_open_started":
      return "Preparing annotated thermal image...";
    case "annotation_image_open_finished":
      return "Annotated thermal image ready.";
    case "matching_done":
      return "Matching hotspot with equipment...";
    case "annotated_image_saved":
      return "Saving final result image...";
    case "upload_completed":
      return "Analysis complete.";
    case "upload_client_disconnected":
    case "raw_upload_client_disconnected":
      return "Upload connection dropped before completion.";
    case "upload_failed":
    case "raw_upload_failed":
    case "analyze_failed":
    case "http_request_failed":
      return "Backend reported a processing failure.";
    default:
      return step ? step.replace(/_/g, " ") : "";
  }
}

// แปลงข้อมูลที่ backend ส่งกลับมา
// ให้เป็นรูปแบบที่หน้าเว็บใช้งานง่าย
export function toAnalysisResult(pair: MatchedPair, responseData: Record<string, unknown>, requestId: string): AnalysisResult {
  let thermalMode: ThermalMode | null = null;

  if (
    responseData.thermal_mode === "none" ||
    responseData.thermal_mode === "absolute" ||
    responseData.thermal_mode === "relative"
  ) {
    thermalMode = responseData.thermal_mode;
  }

  return {
    id: pair.id,
    key: pair.key,
    displayName: pair.displayName,
    thermalFileName: pair.thermal.name,
    rgbFileName: pair.rgb.name,
    annotatedImage:
      typeof responseData.annotated_image === "string" && responseData.annotated_image.trim()
        ? toAbsoluteImageUrl(responseData.annotated_image)
        : null,
    detections: Array.isArray(responseData.detections) ? (responseData.detections as Detection[]) : [],
    latitude: typeof responseData.latitude === "number" ? responseData.latitude : null,
    longitude: typeof responseData.longitude === "number" ? responseData.longitude : null,
    thermalAvailable: typeof responseData.thermal_available === "boolean" ? responseData.thermal_available : null,
    thermalError: typeof responseData.thermal_error === "string" ? responseData.thermal_error : "",
    thermalMode,
    referenceTemperature:
      typeof responseData.reference_temperature === "number" ? responseData.reference_temperature : null,
    message: typeof responseData.message === "string" ? responseData.message : "",
    requestId,
  };
}

// คืนชื่ออุปกรณ์ ถ้าไม่มีข้อมูลก็ใช้ unknown
export function getEquipmentLabel(detection: Detection) {
  return detection.equipment_class ?? "unknown";
}

// สรุปอุณหภูมิแบบสั้น ๆ
function getTemperatureSummary(detection: Detection) {
  if (typeof detection.max_temp === "number") {
    return `${detection.max_temp.toFixed(1)} ${DEGREE_C}`;
  }
  if (typeof detection.max_raw === "number") {
    return `Raw ${detection.max_raw.toFixed(1)}`;
  }
  return "Temperature unavailable";
}

// เปลี่ยนข้อความ priority ให้เป็นตัวเลข
// เพื่อจะได้เทียบได้ว่าอันไหนสำคัญกว่า
function getPriorityRank(priority: string | null | undefined) {
  const normalized = (priority ?? "").toLowerCase();
  if (normalized.includes("priority 1")) {
    return 1;
  }
  if (normalized.includes("priority 2")) {
    return 2;
  }
  if (normalized.includes("priority 3")) {
    return 3;
  }
  return Number.POSITIVE_INFINITY;
}

// หาว่าใน detection ทั้งหมด อันไหนมี priority สูงสุด
function getHighestPriority(detections: Detection[]) {
  let bestPriority: string | null = null;
  let bestRank = Number.POSITIVE_INFINITY;

  for (const detection of detections) {
    const rank = getPriorityRank(detection.priority);
    if (rank < bestRank) {
      bestRank = rank;
      bestPriority = detection.priority ?? null;
    }
  }

  return bestPriority;
}

// สรุปค่าอุณหภูมิแบบเต็ม
export function getTemperatureDetail(detection: Detection) {
  if (
    typeof detection.max_temp === "number" &&
    typeof detection.min_temp === "number" &&
    typeof detection.avg_temp === "number"
  ) {
    return `Max ${detection.max_temp.toFixed(1)} ${DEGREE_C} | Avg ${detection.avg_temp.toFixed(1)} ${DEGREE_C} | Min ${detection.min_temp.toFixed(1)} ${DEGREE_C}`;
  }
  if (
    typeof detection.max_raw === "number" &&
    typeof detection.min_raw === "number" &&
    typeof detection.avg_raw === "number"
  ) {
    return `Max raw ${detection.max_raw.toFixed(1)} | Avg raw ${detection.avg_raw.toFixed(1)} | Min raw ${detection.min_raw.toFixed(1)}`;
  }
  return "Temperature data unavailable";
}

// สรุป hotspot แบบสั้น ๆ สำหรับแสดงหน้า UI
export function getHotspotSummary(detection: Detection) {
  const parts = [getEquipmentLabel(detection), getTemperatureSummary(detection)];

  if (detection.priority) {
    parts.push(detection.priority);
  }

  return parts.join(" | ");
}

// สร้าง id ของ hotspot marker
export function getMarkerId(pairIndex: number, detectionIndex: number) {
  return `${pairIndex}:${detectionIndex}`;
}

// แปลงผลวิเคราะห์ทั้งหมดให้เป็น marker สำหรับแผนที่
export function buildMapMarkers(results: AnalysisResult[]): MapMarkerItem[] {
  return results.flatMap((result, pairIndex) => {
    // ถ้าไม่มีพิกัด ก็ไม่ต้องสร้าง marker
    if (result.latitude === null || result.longitude === null) {
      return [];
    }

    const marker: MapMarkerItem = {
      id: result.id || `${result.key || "pair"}-${pairIndex + 1}`,
      lat: result.latitude,
      lon: result.longitude,
      pairLabel: result.displayName,
      priority: getHighestPriority(result.detections),
      hotspots: result.detections.map((detection, detectionIndex) => ({
        hotspotLabel: `Hotspot ${detectionIndex + 1}`,
        equipmentLabel: getEquipmentLabel(detection),
        temperatureLabel: getTemperatureSummary(detection),
        priority: detection.priority ?? null,
        actionRequired: detection.action_required ?? null,
      })),
    };

    return [marker];
  });
}