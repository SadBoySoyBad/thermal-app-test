import type { MapMarkerItem } from "./MapView";

// ==============================
// ส่วนนี้คืออะไร
// ==============================
// ไฟล์นี้เป็นไฟล์ helper ของหน้า upload
// หน้าที่ของมันคือเก็บ type และฟังก์ชันย่อยต่าง ๆ
// เพื่อไม่ให้ไฟล์ page.tsx ยาวและรกเกินไป

// ไฟล์นี้เป็นไฟล์ช่วยของหน้า upload
// เอาไว้เก็บ type และฟังก์ชันย่อยต่าง ๆ
// เพื่อไม่ให้ไฟล์ page.tsx ยาวและรกเกินไป

// ==============================
// ส่วนกำหนดชนิดข้อมูลพื้นฐาน
// ==============================
// วิธีที่ระบบใช้บอกว่า hotspot ไปตรงกับอุปกรณ์แบบไหน
export type MatchMethod = "inside" | "nearest" | "unknown";

// รูปแบบข้อมูล thermal ที่ได้กลับมา
// none = ไม่มีข้อมูลอุณหภูมิ
// absolute = มีอุณหภูมิจริง
// relative = มีแค่ค่าความร้อนแบบเปรียบเทียบ
export type ThermalMode = "none" | "absolute" | "relative";

// ==============================
// ส่วนนี้คืออะไร
// ==============================
// อันนี้คือกรอบอ้างอิงแบบ normalized
// ใช้เก็บตำแหน่งเป็นสัดส่วนของรูป ไม่ใช่พิกเซลจริง
// ทำให้เอาไปใช้กับรูปหลายขนาดได้ง่ายขึ้น

// [แก้ไขล่าสุดใน version นี้]
// เพิ่มชนิดข้อมูล ROI แบบ normalized
// ใช้เก็บกรอบ reference area ในรูปแบบ x, y, width, height
// เพื่อให้ frontend รู้ว่าผู้ใช้เลือกพื้นที่อ้างอิงตรงไหนของภาพ
export type NormalizedRoi = {
  x: number;
  y: number;
  width: number;
  height: number;
};

// ประเภทไฟล์ที่ระบบมองเห็น
type PairKind = "thermal" | "rgb" | "unknown";

// ==============================
// ส่วนข้อมูล hotspot / detection
// ==============================
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

// ==============================
// ส่วนข้อมูลการจับคู่ไฟล์
// ==============================
// คู่ไฟล์ที่จับคู่สำเร็จแล้ว
// 1 คู่ = thermal 1 ไฟล์ + rgb 1 ไฟล์
export type MatchedPair = {
  id: string;
  key: string;
  displayName: string;
  analysisMode: "paired" | "thermal_only";
  thermal: File;
  rgb: File | null;
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

// ==============================
// ส่วนผลวิเคราะห์ที่หน้าเว็บใช้
// ==============================
// ผลวิเคราะห์ที่ frontend จะเอาไปใช้ต่อ
export type AnalysisResult = {
  id: string;
  key: string;
  displayName: string;

  // [แก้ไขล่าสุดใน version นี้]
  // fileId คือ id ของไฟล์/งานจาก backend
  // ใช้เอาไว้ผูกข้อมูลผลวิเคราะห์กับงานจริงในระบบ
  fileId: string;
  analysisMode: "paired" | "thermal_only";

  thermalFileName: string;
  rgbFileName: string | null;

  // [แก้ไขล่าสุดใน version นี้]
  // thermalImage = รูป thermal ดั้งเดิม
  // rgbImage = รูป RGB ดั้งเดิม
  // fixedRangeImage = รูป thermal ที่ fix ช่วงสีไว้แล้ว
  // ช่วยให้หน้าเว็บสามารถสลับดูรูปแต่ละแบบได้
  thermalImage: string | null;
  rgbImage: string | null;
  fixedRangeImage: string | null;

  // [แก้ไขล่าสุดใน version นี้]
  // annotatedImage = รูปที่มีการวาดกรอบผลวิเคราะห์เพื่อเอาไปแสดงหลัก
  // annotatedImageCamera = รูปที่วาดกรอบบนภาพมุมกล้องปกติ
  // annotatedImageFixedRange = รูปที่วาดกรอบบนภาพ fixed range
  annotatedImage: string | null;
  annotatedImageCamera: string | null;
  hotspotDetectionImage: string | null;
  equipmentDetectionImage: string | null;
  annotatedImageFixedRange: string | null;

  detections: Detection[];

  // [แก้ไขล่าสุดใน version นี้]
  // autoDetections เก็บสำเนาของ detection ตอนเริ่มต้น
  // เอาไว้ใช้เป็นค่าตั้งต้นก่อนมีการแก้ไข/ปรับแต่งในหน้าเว็บ
  autoDetections: Detection[];

  latitude: number | null;
  longitude: number | null;
  thermalAvailable: boolean | null;
  thermalError: string;
  thermalMode: ThermalMode | null;
  referenceTemperature: number | null;
  thermalImageMinTemperature: number | null;
  thermalImageMaxTemperature: number | null;
  fixedRangeMinTemperature: number | null;
  fixedRangeMaxTemperature: number | null;
  rgbDetectionCropMargin: number | null;
  rgbDetectionCropBbox: [number, number, number, number] | null;
  rgbDetectionSize: [number, number] | null;

  // [แก้ไขล่าสุดใน version นี้]
  // autoReferenceTemperature = ค่า reference ที่ระบบคำนวณให้อัตโนมัติ
  // referenceSource = บอกว่าค่า reference มาจาก auto หรือ roi
  // referenceRoi = พื้นที่อ้างอิงที่ผู้ใช้เลือกเอง (ถ้ามี)
  autoReferenceTemperature: number | null;
  referenceSource: "auto" | "roi";
  referenceRoi: NormalizedRoi | null;

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

// ==============================
// ส่วนช่วยแยกและอ่านชื่อไฟล์
// ==============================
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

// ==============================
// ส่วนจับคู่ไฟล์ thermal / rgb
// ==============================
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
        analysisMode: "paired",
        thermal: thermal.file,
        rgb: rgb.file,
      });
    }

    // thermal ที่เหลือโดยไม่มี RGB คู่ จะยังวิเคราะห์ hotspot/temperature ได้
    // แต่จะไม่เอาไป match กับอุปกรณ์ เพราะไม่มีภาพ RGB ให้โมเดล equipment ใช้
    const thermalOnlyItems = thermals.slice(pairCount);
    for (const [thermalOnlyIndex, thermal] of thermalOnlyItems.entries()) {
      pairs.push({
        id: `${key || "thermal"}-thermal-only-${thermalOnlyIndex + 1}`,
        key,
        displayName: formatDisplayName(key, thermal.stem),
        analysisMode: "thermal_only",
        thermal: thermal.file,
        rgb: null,
      });
    }

    // ไฟล์ที่เหลือและจับคู่ไม่ได้
    const leftovers = [...rgbs.slice(pairCount), ...unknowns];

    if (leftovers.length > 0 || (pairCount === 0 && thermalOnlyItems.length === 0)) {
      const hasAnalysisItem = pairCount > 0 || thermalOnlyItems.length > 0;
      issues.push({
        id: key || `issue-${issues.length + 1}`,
        displayName: formatDisplayName(key, leftovers[0]?.stem ?? `Group ${issues.length + 1}`),
        fileNames: group.map((candidate) => candidate.file.name),
        message:
          hasAnalysisItem
            ? "Some RGB or unknown files in this group were skipped because thermal-only analysis does not use RGB-only files."
            : "Unable to auto-match this group. Upload a thermal image alone, or upload matching T/V thermal and RGB files.",
      });
    }
  }

  pairs.sort(
    (left, right) =>
      left.displayName.localeCompare(right.displayName) || left.analysisMode.localeCompare(right.analysisMode),
  );
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

// ==============================
// ส่วนแปลงข้อมูลจาก backend เป็นรูปแบบที่หน้าเว็บใช้
// ==============================
// แปลงข้อมูลที่ backend ส่งกลับมา
// ให้เป็นรูปแบบที่หน้าเว็บใช้งานง่าย
function toNumberTuple(value: unknown, expectedLength: 2): [number, number] | null;
function toNumberTuple(value: unknown, expectedLength: 4): [number, number, number, number] | null;
function toNumberTuple(value: unknown, expectedLength: 2 | 4) {
  if (!Array.isArray(value) || value.length !== expectedLength) {
    return null;
  }

  const numbers = value.map((item) => (typeof item === "number" && Number.isFinite(item) ? item : null));
  if (numbers.some((item) => item === null)) {
    return null;
  }

  return numbers;
}

export function toAnalysisResult(pair: MatchedPair, responseData: Record<string, unknown>, requestId: string): AnalysisResult {
  let thermalMode: ThermalMode | null = null;
  const analysisMode =
    responseData.analysis_mode === "thermal_only" || pair.analysisMode === "thermal_only" ? "thermal_only" : "paired";

  if (
    responseData.thermal_mode === "none" ||
    responseData.thermal_mode === "absolute" ||
    responseData.thermal_mode === "relative"
  ) {
    thermalMode = responseData.thermal_mode;
  }

  // [แก้ไขล่าสุดใน version นี้]
  // cloneDetections ถูกใช้เพื่อแยกข้อมูลออกจาก object ต้นทาง
  // ป้องกันปัญหาเวลาแก้ค่า detections ฝั่ง frontend แล้วกระทบค่าต้นฉบับโดยไม่ตั้งใจ
  const detections = Array.isArray(responseData.detections) ? cloneDetections(responseData.detections as Detection[]) : [];

  // [แก้ไขล่าสุดใน version นี้]
  // ดึงค่า reference temperature ที่ backend ส่งมา
  const referenceTemperature =
    typeof responseData.reference_temperature === "number" ? responseData.reference_temperature : null;

  // [แก้ไขล่าสุดใน version นี้]
  // annotatedImageCamera คือรูปผลลัพธ์ที่วาดกรอบบนภาพ camera view
  // ถ้า backend ยังไม่ส่ง field ใหม่มา จะ fallback ไปใช้ annotated_image เดิมแทน
  const annotatedImageCamera =
    typeof responseData.annotated_image_camera === "string" && responseData.annotated_image_camera.trim()
      ? toAbsoluteImageUrl(responseData.annotated_image_camera)
      : typeof responseData.annotated_image === "string" && responseData.annotated_image.trim()
        ? toAbsoluteImageUrl(responseData.annotated_image)
        : null;

  // [แก้ไขล่าสุดใน version นี้]
  // annotatedImageFixedRange คือรูปผลลัพธ์อีกแบบที่ใช้ fixed color range
  const annotatedImageFixedRange =
    typeof responseData.annotated_image_fixed_range === "string" && responseData.annotated_image_fixed_range.trim()
      ? toAbsoluteImageUrl(responseData.annotated_image_fixed_range)
      : null;

  // [แก้ไขล่าสุดใน version นี้]
  // thermalImage = รูป thermal ที่อัปโหลดเข้าไป
  // [เพิ่มใหม่]
  // รูป debug จากโมเดลก่อนขั้น match:
  // hotspotDetectionImage = ผล hotspot model บน thermal
  // equipmentDetectionImage = ผล equipment model บน RGB
  const hotspotDetectionImage =
    typeof responseData.hotspot_detection_image === "string" && responseData.hotspot_detection_image.trim()
      ? toAbsoluteImageUrl(responseData.hotspot_detection_image)
      : null;
  const equipmentDetectionImage =
    typeof responseData.equipment_detection_image === "string" && responseData.equipment_detection_image.trim()
      ? toAbsoluteImageUrl(responseData.equipment_detection_image)
      : null;

  const thermalImage =
    typeof responseData.uploaded_image === "string" && responseData.uploaded_image.trim()
      ? toAbsoluteImageUrl(responseData.uploaded_image)
      : null;

  // [แก้ไขล่าสุดใน version นี้]
  // rgbImage = รูป RGB ที่อัปโหลดเข้าไป
  const rgbImage =
    typeof responseData.uploaded_rgb_image === "string" && responseData.uploaded_rgb_image.trim()
      ? toAbsoluteImageUrl(responseData.uploaded_rgb_image)
      : null;

  // [แก้ไขล่าสุดใน version นี้]
  // fixedRangeImage = รูป thermal ที่แปลงช่วงสีให้คงที่
  const fixedRangeImage =
    typeof responseData.fixed_range_image === "string" && responseData.fixed_range_image.trim()
      ? toAbsoluteImageUrl(responseData.fixed_range_image)
      : null;

  return {
    id: pair.id,
    key: pair.key,
    displayName: pair.displayName,

    // [แก้ไขล่าสุดใน version นี้]
    // รับ file id จาก backend ถ้ามี
    fileId: typeof responseData.file_id === "string" ? responseData.file_id : "",
    analysisMode,

    thermalFileName: pair.thermal.name,
    rgbFileName: pair.rgb?.name ?? null,
    thermalImage,
    rgbImage,
    fixedRangeImage,

    // [แก้ไขล่าสุดใน version นี้]
    // ให้ annotatedImage หลักชี้ไปที่ภาพ camera ก่อน
    // เพื่อให้หน้าเว็บใช้งาน field เดิมต่อได้โดยไม่ต้องแก้ทุกจุด
    annotatedImage: annotatedImageCamera,
    annotatedImageCamera,
    hotspotDetectionImage,
    equipmentDetectionImage,
    annotatedImageFixedRange,
    detections,

    // [แก้ไขล่าสุดใน version นี้]
    // เก็บสำเนาค่าเริ่มต้นไว้เป็น autoDetections
    autoDetections: cloneDetections(detections),

    latitude: typeof responseData.latitude === "number" ? responseData.latitude : null,
    longitude: typeof responseData.longitude === "number" ? responseData.longitude : null,
    thermalAvailable: typeof responseData.thermal_available === "boolean" ? responseData.thermal_available : null,
    thermalError: typeof responseData.thermal_error === "string" ? responseData.thermal_error : "",
    thermalMode,
    referenceTemperature,
    thermalImageMinTemperature:
      typeof responseData.thermal_image_min_temperature === "number" ? responseData.thermal_image_min_temperature : null,
    thermalImageMaxTemperature:
      typeof responseData.thermal_image_max_temperature === "number" ? responseData.thermal_image_max_temperature : null,
    fixedRangeMinTemperature:
      typeof responseData.fixed_range_min_temperature === "number" ? responseData.fixed_range_min_temperature : null,
    fixedRangeMaxTemperature:
      typeof responseData.fixed_range_max_temperature === "number" ? responseData.fixed_range_max_temperature : null,
    rgbDetectionCropMargin:
      typeof responseData.rgb_detection_crop_margin === "number" ? responseData.rgb_detection_crop_margin : null,
    rgbDetectionCropBbox: toNumberTuple(responseData.rgb_detection_crop_bbox, 4),
    rgbDetectionSize: toNumberTuple(responseData.rgb_detection_size, 2),

    // [แก้ไขล่าสุดใน version นี้]
    // ตอนเริ่มต้นให้ค่า autoReferenceTemperature เท่ากับค่า reference ที่ระบบคำนวณมา
    autoReferenceTemperature: referenceTemperature,

    // [แก้ไขล่าสุดใน version นี้]
    // ค่าเริ่มต้นของแหล่งที่มาคือ auto
    referenceSource: "auto",

    // [แก้ไขล่าสุดใน version นี้]
    // ตอนรับข้อมูลครั้งแรกยังไม่มี roi ที่ผู้ใช้เลือก จึงเป็น null
    referenceRoi: null,

    message: typeof responseData.message === "string" ? responseData.message : "",
    requestId,
  };
}

// ==============================
// ส่วนช่วย clone object
// ==============================
// อธิบาย: clone คือการก็อปปี้ข้อมูลออกมาอีกชุด
// เพื่อให้เวลาแก้ไขค่าชุดใหม่ ไม่ไปกระทบค่าต้นฉบับ

// [แก้ไขล่าสุดใน version นี้]
// clone ROI ออกมาเป็น object ใหม่
export function cloneNormalizedRoi(roi: NormalizedRoi | null | undefined): NormalizedRoi | null {
  if (!roi) {
    return null;
  }

  return {
    x: roi.x,
    y: roi.y,
    width: roi.width,
    height: roi.height,
  };
}

// [แก้ไขล่าสุดใน version นี้]
// clone detection 1 ตัวแบบลึกขึ้นใน field ที่เป็น array
// เพื่อกันปัญหาอ้างอิง object เดิมร่วมกัน
export function cloneDetection(detection: Detection): Detection {
  return {
    ...detection,
    bbox: [...detection.bbox] as [number, number, number, number],
    thermal_bbox: detection.thermal_bbox ? [...detection.thermal_bbox] as [number, number, number, number] : undefined,
    hotspot_center: detection.hotspot_center ? [...detection.hotspot_center] as [number, number] : detection.hotspot_center,
    max_point: detection.max_point ? [...detection.max_point] as [number, number] : detection.max_point,
    min_point: detection.min_point ? [...detection.min_point] as [number, number] : detection.min_point,
    equipment_bbox: detection.equipment_bbox
      ? ([...detection.equipment_bbox] as [number, number, number, number])
      : detection.equipment_bbox,
  };
}

// [แก้ไขล่าสุดใน version นี้]
// clone detection ทั้ง list
export function cloneDetections(detections: Detection[]): Detection[] {
  return detections.map((detection) => cloneDetection(detection));
}

// ==============================
// ส่วนสรุปข้อความเพื่อแสดงใน UI
// ==============================
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
  if (normalized.includes("priority 4")) {
    return 4;
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

// ==============================
// ส่วนสร้าง marker สำหรับแผนที่
// ==============================
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
