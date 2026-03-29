"use client";

// เพิ่ม useEffect เพราะหน้าใหม่มี polling progress และจับเวลา
// เพิ่ม ChangeEvent เพื่อระบุ type ของ event ตอนเลือกไฟล์ให้ชัดเจน
import { useEffect, useState, type ChangeEvent } from "react";
import dynamic from "next/dynamic";

// [เพิ่มใหม่]
// ใช้ next/image แทน img ธรรมดาในโค้ดใหม่
// เพื่อให้เข้ากับแนวทางของ Next.js และจัดการรูปได้เป็นระบบมากขึ้น
import Image from "next/image";

// [เพิ่มใหม่]
// โค้ดใหม่ย้าย type หลายตัวไปไว้ในไฟล์ uploadUtils แล้ว
// ทำให้ไฟล์หน้านี้ไม่ต้องแบก type และ helper ทุกอย่างไว้เอง
import type { AnalysisResult, FailedPair, MatchedPair, PairingIssue } from "./uploadUtils";

// [เพิ่มใหม่]
// helper หลายตัวที่เคยอยู่ในไฟล์เก่า ถูกย้ายไป import จาก uploadUtils
// เช่น createRequestId, describeBackendStep, getResponseRequestId
// รวมถึง helper ใหม่สำหรับ map และการสรุปข้อมูล hotspot
import {
  DEGREE_C,
  buildMapMarkers,
  createRequestId,
  describeBackendStep,
  getEquipmentLabel,
  getHotspotSummary,
  getMarkerId,
  getResponseRequestId,
  getTemperatureDetail,
  matchUploadPairs,
  toAnalysisResult,
} from "./uploadUtils";

// ยังใช้ dynamic import เหมือนเดิม เพื่อให้ MapView โหลดเฉพาะฝั่ง browser
// ป้องกันปัญหา SSR กับ Leaflet คือ ไม่ต้องพยายาม Render Component นี้ที่ฝั่ง Server ให้รอจนกว่าไฟล์ JavaScript จะไปถึง Browser ของผู้ใช้ก่อนค่อยเริ่มทำงาน
const MapView = dynamic(() => import("./MapView"), { ssr: false });

// [เพิ่มใหม่]
// ใช้กำหนดโทนของข้อความสถานะบนหน้า
// default = ข้อความทั่วไป
// warning = ข้อความเตือน
type MessageTone = "default" | "warning";

// เหมือนเดิม: อ่าน backend URL จาก env
const backendBaseUrl = (process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

// helper เหมือนเดิม: แปลงวินาที -> mm:ss
function formatElapsedTime(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

// [เพิ่มใหม่]
// helper สำหรับเช็กว่า value เป็น object แบบ record หรือไม่
// ใช้ช่วยกัน error เวลาตรวจ response จาก backend
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export default function Home() {
  // ------------------------------
  // State กลุ่มข้อความ/สถานะหน้า
  // ------------------------------
  const [message, setMessage] = useState("");

  // [เพิ่มใหม่]
  // เก็บโทนข้อความ เพื่อให้เลือก style ได้ว่าข้อความนี้เป็นคำเตือนหรือไม่
  const [messageTone, setMessageTone] = useState<MessageTone>("default");

  /*
    [เพิ่มใหม่]
    progressMessage = ข้อความบอกว่า backend กำลังทำขั้นตอนไหน
    เช่น Uploading thermal image..., Running RGB equipment model...
  */
  const [progressMessage, setProgressMessage] = useState("");

  /*
    [เพิ่มใหม่]
    elapsedSeconds = เวลาที่ผ่านไประหว่างการประมวลผล
    runStartedAt   = timestamp ตอนเริ่มงาน
  */
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);

  /*
    [เพิ่มใหม่]
    requestId = id ของ request รอบปัจจุบัน
    ใช้ไว้ตาม progress จาก backend และ debug เวลา error
  */
  const [requestId, setRequestId] = useState("");

  const [loading, setLoading] = useState(false);

  // ------------------------------
  // State กลุ่มไฟล์ที่ user เลือกมา
  // ------------------------------

  /*
    [เปลี่ยนสำคัญ]
    โค้ดเก่ามี input file เดียว เพราะรับแค่ thermal image รูปเดียว
    โค้ดใหม่รุ่นก่อนหน้านี้แยกเป็น 2 ไฟล์:
    - thermalFile = ไฟล์ภาพ thermal
    - rgbFile     = ไฟล์ภาพ RGB

    แต่โค้ดใหม่นี้พัฒนาไปอีกขั้น:
    - selectedFiles = ให้ user เลือกหลายไฟล์พร้อมกัน
    - แล้วระบบจะจับคู่ thermal/rgb ให้อัตโนมัติ
  */
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  // [เพิ่มใหม่]
  // ข้อความที่ใช้แสดงชื่อรวมของไฟล์ที่ผู้ใช้เลือก
  // เช่น "8 images selected"
  const [selectedFileLabel, setSelectedFileLabel] = useState("No images chosen");

  // [เพิ่มใหม่]
  // คู่ไฟล์ thermal/rgb ที่ระบบจับคู่ได้สำเร็จ
  const [matchedPairs, setMatchedPairs] = useState<MatchedPair[]>([]);

  // [เพิ่มใหม่]
  // กลุ่มไฟล์ที่ระบบจับคู่ไม่ได้ หรือชื่อไฟล์ยังไม่ชัด
  const [pairingIssues, setPairingIssues] = useState<PairingIssue[]>([]);

  // ------------------------------
  // State กลุ่มผลลัพธ์จาก backend
  // ------------------------------

  // [เปลี่ยนสำคัญ]
  // เดิมเก็บผลลัพธ์แบบภาพเดียว เช่น annotatedImage, detections, lat, lon
  // ใหม่เปลี่ยนเป็น results[] เพื่อเก็บผลของหลายคู่ภาพ
  const [results, setResults] = useState<AnalysisResult[]>([]);

  // [เพิ่มใหม่]
  // เก็บรายการคู่ภาพที่วิเคราะห์ไม่สำเร็จ
  const [failedPairs, setFailedPairs] = useState<FailedPair[]>([]);

  // [เพิ่มใหม่]
  // ใช้ระบุว่าตอนนี้กำลังดูผลของคู่ภาพลำดับไหน
  const [selectedPairIndex, setSelectedPairIndex] = useState(0);

  // [เพิ่มใหม่]
  // ใช้ระบุว่าตอนนี้กำลังดู hotspot ตัวที่เท่าไรในภาพนั้น
  const [selectedDetectionIndex, setSelectedDetectionIndex] = useState(0);

  // [เพิ่มใหม่]
  // ใช้แสดง progress ว่าตอนนี้ batch analysis กำลังรันถึงคู่ที่เท่าไร
  const [activePairIndex, setActivePairIndex] = useState(0);
  const [activePairTotal, setActivePairTotal] = useState(0);
  const [activePairLabel, setActivePairLabel] = useState("");

  // [เพิ่มใหม่]
  // ใช้แยก layout ระหว่าง mobile กับ desktop
  const [isMobileViewport, setIsMobileViewport] = useState(false);

  // [เพิ่มใหม่]
  // คู่ภาพที่กำลังถูกเลือกดูอยู่
  const selectedPair = results[selectedPairIndex] ?? null;

  // [เพิ่มใหม่]
  // ป้องกัน index ของ hotspot ไม่ให้เกินจำนวนจริง
  const safeSelectedDetectionIndex =
    selectedPair && selectedPair.detections.length > 0
      ? Math.min(selectedDetectionIndex, selectedPair.detections.length - 1)
      : 0;

  // [เพิ่มใหม่]
  // hotspot ที่กำลังถูกเลือกดูอยู่จริง
  const selectedDetection = selectedPair?.detections[safeSelectedDetectionIndex] ?? null;

  // [เพิ่มใหม่]
  // สร้าง marker ทั้งหมดสำหรับ map จาก results ทุกตัว
  const hotspotMarkers = buildMapMarkers(results);

  // [เพิ่มใหม่]
  // ถ้าคู่ภาพที่เลือกมี GPS ก็ใช้คู่นั้น
  // ถ้าไม่มี ให้ fallback ไปหาคู่แรกที่มี GPS
  const selectedMapResult =
    selectedPair && selectedPair.latitude !== null && selectedPair.longitude !== null
      ? selectedPair
      : results.find((result) => result.latitude !== null && result.longitude !== null) ?? null;

  // [เพิ่มใหม่]
  // id ของ marker ที่ควรถูก highlight บนแผนที่
  const selectedMarkerId = selectedMapResult ? selectedMapResult.id : null;

  // [เพิ่มใหม่]
  // นับจำนวน hotspot ทั้งหมดของทุกภาพ
  const totalHotspots = results.reduce((count, result) => count + result.detections.length, 0);

  // [เพิ่มใหม่]
  // class สำหรับ layout ปุ่มเลือกภาพ
  const selectionRailClassName = `selectionRail ${isMobileViewport ? "selectionRailMobile" : "selectionRailDesktop"}`;
  const selectionChipClassName = `selectionChip ${isMobileViewport ? "selectionChipMobile" : "selectionChipDesktop"}`;

  // [เพิ่มใหม่]
  // helper กลางสำหรับตั้งข้อความ + โทนข้อความในครั้งเดียว
  function showMessage(nextMessage: string, tone: MessageTone = "default") {
    setMessage(nextMessage);
    setMessageTone(tone);
  }

  /*
    reset เฉพาะผลวิเคราะห์
    ไม่ลบไฟล์ที่ผู้ใช้เลือกไว้ เพื่อให้กด Analyze ใหม่ได้ทันที
  */
  function resetAnalysisState() {
    setResults([]);
    setFailedPairs([]);
    setSelectedPairIndex(0);
    setSelectedDetectionIndex(0);
    setRequestId("");
    setElapsedSeconds(0);
    setRunStartedAt(null);
    setActivePairIndex(0);
    setActivePairTotal(0);
    setActivePairLabel("");
  }

  /*
    polling progress จาก backend ทุก 1 วินาที
    ใช้ requestId ล่าสุดของคู่ภาพที่กำลังวิ่งอยู่
  */
  useEffect(() => {
    if (!loading || !requestId) {
      return;
    }

    let isActive = true;

    const pollProgress = async () => {
      try {
        const progressResponse = await fetch(`${backendBaseUrl}/progress/${requestId}`, {
          cache: "no-store",
        });
        const progressData = await progressResponse.json().catch(() => null);

        // [ปรับใหม่]
        // โค้ดใหม่เช็ก response แบบปลอดภัยขึ้น ด้วย isRecord
        if (!isActive || !isRecord(progressData) || progressData.success !== true) {
          return;
        }

        const backendStepMessage = describeBackendStep(
          typeof progressData.step === "string" ? progressData.step : null,
          isRecord(progressData.details) ? progressData.details : null,
        );

        if (backendStepMessage) {
          setProgressMessage(backendStepMessage);
        }
      } catch {
        // ignore intermittent progress failures while batch processing continues
      }
    };

    void pollProgress();
    const pollTimer = window.setInterval(() => {
      void pollProgress();
    }, 1000);

    return () => {
      isActive = false;
      window.clearInterval(pollTimer);
    };
  }, [loading, requestId]);

  /*
    นับเวลาระหว่างที่ batch ยังทำงาน
    เอาไว้แสดง elapsed time บนหน้า
  */
  useEffect(() => {
    if (!loading || runStartedAt === null) {
      return;
    }

    const updateElapsed = () => {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - runStartedAt) / 1000)));
    };

    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);

    return () => {
      window.clearInterval(timer);
    };
  }, [loading, runStartedAt]);

  /*
    [เพิ่มใหม่]
    desktop = 6 ช่องคงที่ต่อแถว
    mobile/tablet = แถบแนวนอนเลื่อนได้
  */
  useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 820px)");

    const syncViewport = () => {
      setIsMobileViewport(mediaQuery.matches);
    };

    syncViewport();
    mediaQuery.addEventListener("change", syncViewport);

    return () => {
      mediaQuery.removeEventListener("change", syncViewport);
    };
  }, []);

  /*
    ตอนนี้เลือกไฟล์ทีเดียวได้หลายภาพ
    แล้วใช้ helper จับคู่ thermal/rgb ตามเลขในชื่อไฟล์
  */
  async function handleBatchFilesChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFiles = Array.from(event.target.files ?? []).filter((file) => file.size > 0);
    event.target.value = "";

    setSelectedFiles(nextFiles);
    setSelectedFileLabel(nextFiles.length > 0 ? `${nextFiles.length} images selected` : "No images chosen");

    resetAnalysisState();
    setMatchedPairs([]);
    setPairingIssues([]);
    showMessage("");

    if (nextFiles.length === 0) {
      return;
    }

    const { pairs, issues } = await matchUploadPairs(nextFiles);
    setMatchedPairs(pairs);
    setPairingIssues(issues);

    if (pairs.length === 0) {
      showMessage("No thermal/RGB pairs could be matched from the selected files.", "warning");
      return;
    }

    if (issues.length > 0) {
      showMessage(`Matched ${pairs.length} pairs. Some files still need clearer thermal/RGB names.`, "warning");
      return;
    }

    showMessage(`Matched ${pairs.length} pairs and ready to analyze.`);
  }

  /*
    helper อัปโหลดทีละไฟล์เหมือนเดิม
    ยังใช้ flow เดิมคือ /upload-file ก่อน แล้วค่อย /analyze
  */
  async function uploadSingleFile(file: File, kind: "thermal" | "rgb", existingFileId?: string) {
    const params = new URLSearchParams({ kind });
    if (existingFileId) {
      params.set("file_id", existingFileId);
    }

    const uploadRequestId = createRequestId();

    const uploadResponse = await fetch(`${backendBaseUrl}/upload-file?${params.toString()}`, {
      method: "POST",
      headers: {
        // โค้ดใหม่ส่งไฟล์แบบ raw body พร้อม header metadata
        "Content-Type": file.type || "application/octet-stream",
        "x-file-name": file.name,
        "x-request-id": uploadRequestId,
      },
      body: file,
    });

    const responseData = await uploadResponse.json().catch(() => null);
    const headerRequestId = uploadResponse.headers.get("x-request-id") ?? "";
    const responseRequestId = getResponseRequestId(responseData, headerRequestId || uploadRequestId);

    if (!uploadResponse.ok || !isRecord(responseData) || responseData.success !== true) {
      const fallbackMessage = uploadResponse.ok
        ? `Failed to upload ${kind} image.`
        : `Backend returned HTTP ${uploadResponse.status} while uploading ${kind} image.`;

      throw {
        requestId: responseRequestId,
        message:
          isRecord(responseData) && typeof responseData.message === "string" && responseData.message.trim()
            ? responseData.message
            : fallbackMessage,
      };
    }

    return {
      fileId: typeof responseData.file_id === "string" ? responseData.file_id : existingFileId ?? "",
      requestId: responseRequestId,
    };
  }

  /*
    วิเคราะห์ภาพ 1 คู่
    ขั้นตอนเดิม:
    1) upload thermal
    2) upload rgb
    3) เรียก /analyze
  */
  async function analyzeMatchedPair(pair: MatchedPair) {
    const thermalUpload = await uploadSingleFile(pair.thermal, "thermal");
    setRequestId(thermalUpload.requestId);

    setProgressMessage("Uploading RGB image...");
    const rgbUpload = await uploadSingleFile(pair.rgb, "rgb", thermalUpload.fileId);
    setRequestId(rgbUpload.requestId);

    setProgressMessage("Running hotspot and equipment analysis...");
    const analyzeRequestId = createRequestId();
    setRequestId(analyzeRequestId);

    const analyzeResponse = await fetch(`${backendBaseUrl}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-request-id": analyzeRequestId,
      },
      body: JSON.stringify({ file_id: rgbUpload.fileId }),
    });

    const responseData = await analyzeResponse.json().catch(() => null);
    const headerRequestId = analyzeResponse.headers.get("x-request-id") ?? "";
    const responseRequestId = getResponseRequestId(responseData, headerRequestId || analyzeRequestId);

    if (!analyzeResponse.ok || !isRecord(responseData) || responseData.success !== true) {
      const fallbackMessage = analyzeResponse.ok
        ? "Analysis failed. Please try again."
        : `Backend returned HTTP ${analyzeResponse.status}.`;

      throw {
        requestId: responseRequestId,
        message:
          isRecord(responseData) && typeof responseData.message === "string" && responseData.message.trim()
            ? responseData.message
            : fallbackMessage,
      };
    }

    // [เพิ่มใหม่]
    // โค้ดใหม่แปลงผลลัพธ์ backend ให้เป็นรูปแบบ AnalysisResult ผ่าน helper กลาง
    return toAnalysisResult(pair, responseData, responseRequestId);
  }

  /*
    ใหม่: วนทีละคู่จากชุดไฟล์ที่เลือกมา
    แต่ backend pipeline ภายในยังทำงานแบบเดิมทุกคู่
  */
  async function handleUpload() {
    if (matchedPairs.length === 0) {
      showMessage("Please choose image files that can be matched into thermal/RGB pairs.", "warning");
      return;
    }

    showMessage("");
    resetAnalysisState();
    setLoading(true);
    setRunStartedAt(Date.now());
    setActivePairTotal(matchedPairs.length);
    setProgressMessage("Preparing batch analysis...");

    const nextFailedPairs: FailedPair[] = [];
    const nextResults: AnalysisResult[] = [];

    try {
      for (const [pairIndex, pair] of matchedPairs.entries()) {
        setActivePairIndex(pairIndex + 1);
        setActivePairLabel(pair.displayName);
        setProgressMessage("Uploading thermal image...");

        try {
          const result = await analyzeMatchedPair(pair);
          nextResults.push(result);
          setResults([...nextResults]);
          setSelectedPairIndex(nextResults.length - 1);
          setSelectedDetectionIndex(0);

          if (result.message.trim()) {
            showMessage(result.message, "warning");
          }
        } catch (error) {
          const failedRequestId =
            isRecord(error) && typeof error.requestId === "string" ? error.requestId : "";
          const failedMessage =
            isRecord(error) && typeof error.message === "string" && error.message.trim()
              ? error.message
              : "Cannot reach backend. The request did not complete.";

          if (failedRequestId) {
            setRequestId(failedRequestId);
          }

          nextFailedPairs.push({
            id: pair.id,
            displayName: pair.displayName,
            message:
              failedMessage === "Failed to fetch"
                ? "Backend connection dropped during upload or analysis."
                : failedMessage,
          });
          setFailedPairs([...nextFailedPairs]);
        }
      }

      const summaryParts = [`Analyzed ${nextResults.length}/${matchedPairs.length} matched pairs.`];

      if (pairingIssues.length > 0) {
        summaryParts.push(`${pairingIssues.length} groups still need clearer file names.`);
      }

      if (nextFailedPairs.length > 0) {
        summaryParts.push(`${nextFailedPairs.length} pairs failed.`);
      }

      showMessage(
        summaryParts.join(" "),
        nextFailedPairs.length > 0 || pairingIssues.length > 0 ? "warning" : "default",
      );
    } catch {
      showMessage("Batch analysis stopped unexpectedly.", "warning");
    } finally {
      // ปิดสถานะ loading และล้างข้อความ progress เมื่อจบงาน
      setProgressMessage("");
      setLoading(false);
    }
  }

  // [เพิ่มใหม่]
  // ปุ่ม Previous / Next ของรูปหลัก
  function selectPair(nextIndex: number) {
    if (results.length === 0) {
      return;
    }

    const normalizedIndex = (nextIndex + results.length) % results.length;
    setSelectedPairIndex(normalizedIndex);
    setSelectedDetectionIndex(0);
  }

  // [เพิ่มใหม่]
  // เลือก hotspot ที่กำลังดูอยู่
  function selectDetection(pairIndex: number, detectionIndex: number) {
    setSelectedPairIndex(pairIndex);
    setSelectedDetectionIndex(detectionIndex);
  }

  // [เพิ่มใหม่]
  // เวลา user คลิก marker บนแผนที่ ให้ sync กลับมาที่รูปนั้น
  function handleMarkerSelect(markerId: string) {
    const pairIndex = results.findIndex((result) => result.id === markerId);

    if (pairIndex < 0) {
      return;
    }

    setSelectedPairIndex(pairIndex);
    setSelectedDetectionIndex(0);
  }

  return (
    <main className="page">
      <section className="card">
        <header className="hero">
          {/* [แก้ข้อความ UI]
             เดิมในเวอร์ชันก่อนหน้านี้ใช้ "Thermal - RGB - GPS - Map"
             แต่ไฟล์ใหม่นี้เปลี่ยนกลับมาเป็น "Thermal - GPS - Map"
          */}
          <p className="eyebrow">Thermal - GPS - Map</p>

          {/* [แก้ชื่อระบบ]
             เดิมอีกไฟล์ใช้ชื่อ Thermal Hotspot Equipment Matcher
             แต่ไฟล์ใหม่นี้เปลี่ยนกลับมาใช้ชื่อ Thermal Image GPS Viewer
          */}
          <h1>Thermal Image GPS Viewer</h1>

          <p className="subtle">
            Upload the thermal image with GPS metadata and its matching RGB image to identify the hotspot equipment.
          </p>
        </header>

        {/* ส่วน upload ใหม่: เลือกหลายไฟล์จาก input เดียว */}
        <div className="uploadStack">
          <div className="uploadRow">
            <span className="uploadLabel">Thermal + RGB images</span>
            <input
              id="batch-files"
              className="fileInput"
              type="file"
              accept="image/*"
              multiple
              onChange={(event) => {
                void handleBatchFilesChange(event);
              }}
            />
            <label htmlFor="batch-files" className="fileButton">
              Choose images
            </label>
            <span className="fileName">{selectedFileLabel}</span>
          </div>

          {/* [เพิ่มใหม่]
             ปุ่มเริ่มวิเคราะห์แบบ batch
             disabled ถ้ายังไม่มีคู่ภาพที่จับได้ หรือกำลังประมวลผลอยู่
          */}
          <button
            className="analyzeButton"
            type="button"
            onClick={() => {
              void handleUpload();
            }}
            disabled={loading || matchedPairs.length === 0}
          >
            {loading ? "Analyzing..." : "Analyze All Pairs"}
          </button>
        </div>

        {/* [เพิ่มใหม่]
           แถบสรุปจำนวนไฟล์ จำนวนคู่ที่จับได้ และจำนวนกลุ่มที่ยังมีปัญหา
        */}
        {selectedFiles.length > 0 && (
          <div className="summaryBar">
            <span>{selectedFiles.length} uploaded images</span>
            <span>{matchedPairs.length} matched pairs</span>
            <span>{pairingIssues.length} groups need attention</span>
          </div>
        )}

        {/* [เพิ่มใหม่]
           แสดงรายการคู่ภาพที่ระบบจับคู่ได้
        */}
        {matchedPairs.length > 0 && (
          <div className="pairGrid">
            {matchedPairs.map((pair) => (
              <article key={pair.id} className="pairChip">
                <p className="pairChipTitle">{pair.displayName}</p>
                <p className="pairChipMeta">Thermal: {pair.thermal.name}</p>
                <p className="pairChipMeta">RGB: {pair.rgb.name}</p>
              </article>
            ))}
          </div>
        )}

        {/* [เพิ่มใหม่]
           แสดงไฟล์หรือกลุ่มที่จับคู่ไม่สำเร็จ
        */}
        {pairingIssues.length > 0 && (
          <div className="warningPanel">
            <p className="warningTitle">Files that could not be matched cleanly</p>
            <ul className="warningList">
              {pairingIssues.map((issue) => (
                <li key={issue.id}>
                  <strong>{issue.displayName}:</strong> {issue.message} ({issue.fileNames.join(", ")})
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* [เพิ่มใหม่]
           แสดงคู่ภาพที่วิเคราะห์ล้มเหลว
        */}
        {failedPairs.length > 0 && (
          <div className="warningPanel">
            <p className="warningTitle">Pairs that failed during analysis</p>
            <ul className="warningList">
              {failedPairs.map((failedPair) => (
                <li key={failedPair.id}>
                  <strong>{failedPair.displayName}:</strong> {failedPair.message}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* [เพิ่มใหม่]
           ระหว่างทำงาน แสดงว่าตอนนี้กำลังรันคู่ภาพลำดับไหน
        */}
        {loading && activePairTotal > 0 && (
          <p className="status">
            Pair {activePairIndex} of {activePairTotal}: {activePairLabel}
          </p>
        )}

        {/* [เพิ่มข้อความสถานะระหว่างทำงาน]
           เดิมมีแค่ Uploading...
           ใหม่มี:
           - ข้อความสถานะรวม
           - progress message ตาม step
           - elapsed time
           - request ID
        */}
        {loading && progressMessage && <p className="status progress">{progressMessage}</p>}
        {loading && <p className="status subtleStatus">Elapsed: {formatElapsedTime(elapsedSeconds)}</p>}
        {requestId && <p className="status subtleStatus">Request ID: {requestId}</p>}
        {message && <p className={`status ${messageTone === "warning" ? "warning" : ""}`}>{message}</p>}
      </section>

      {/* [เปลี่ยนสำคัญ]
         เดิมแสดงผลของภาพเดียว
         ใหม่แสดงผลของหลายภาพ และเลือกดูทีละคู่ได้
      */}
      {results.length > 0 && (
        <section className="card mapCard">
          <div className="sectionHeader">
            <div>
              {/* คืนชื่อ section กลับไปตามหน้าเดิม */}
              <h2 className="mapTitle">Thermal Hotspot Result</h2>
              <p className="subtle">
                {results.length} images analyzed with {totalHotspots} detected hotspots.
              </p>
            </div>

            {/* [เพิ่มใหม่]
               ปุ่มเปลี่ยนภาพก่อนหน้า / ถัดไป
            */}
            <div className="navigatorButtons">
              <button className="navButton" type="button" onClick={() => selectPair(selectedPairIndex - 1)}>
                Previous
              </button>
              <button className="navButton" type="button" onClick={() => selectPair(selectedPairIndex + 1)}>
                Next
              </button>
            </div>
          </div>

          {/* [เพิ่มใหม่]
             แถวปุ่มเลือกผลลัพธ์แต่ละภาพ
          */}
          <div className={selectionRailClassName}>
            {results.map((result, index) => (
              <button
                key={result.id}
                type="button"
                className={`${selectionChipClassName} ${index === selectedPairIndex ? "active" : ""}`}
                onClick={() => selectPair(index)}
              >
                <span className="selectionChipIndex">{index + 1}</span>
                <span>{result.displayName}</span>
              </button>
            ))}
          </div>

          {selectedPair && (
            <div className="resultGrid">
              <div className="annotatedPanel">
                {selectedPair.annotatedImage ? (
                  <Image
                    src={selectedPair.annotatedImage}
                    alt={`Annotated hotspot result for ${selectedPair.displayName}`}
                    width={1600}
                    height={900}
                    unoptimized
                    className="annotatedImage"
                  />
                ) : (
                  <div className="emptyState">Annotated image is unavailable for this pair.</div>
                )}

                {/* เหมือนเดิม: ถ้าไม่มี absolute temperature ให้แจ้งเตือน */}
                {selectedPair.thermalAvailable === false && (
                  <p className="status warning">
                    {selectedPair.thermalMode === "relative"
                      ? `Absolute temperature unavailable: ${
                          selectedPair.thermalError || "Relative hotspot points are shown only."
                        }`
                      : `Temperature extraction unavailable: ${
                          selectedPair.thermalError || "RawThermalImage metadata not found."
                        }`}
                  </p>
                )}

                {/* [เพิ่มใหม่]
                   แสดง reference temperature ถ้ามี
                */}
                {selectedPair.referenceTemperature !== null && (
                  <p className="subtle">
                    Reference temperature: {selectedPair.referenceTemperature.toFixed(1)} {DEGREE_C}
                  </p>
                )}
              </div>

              {/* [เพิ่มใหม่]
                 คอลัมน์รายละเอียดด้านขวา
                 แยกเป็นข้อมูลภาพ, รายการ hotspot, และรายละเอียด hotspot ที่เลือก
              */}
              <div className="detailColumn">
                <article className="detailCard">
                  <h3 className="detailTitle">
                    {selectedPair.displayName} ({selectedPairIndex + 1}/{results.length})
                  </h3>
                  <p className="detailLine">Thermal: {selectedPair.thermalFileName}</p>
                  <p className="detailLine">RGB: {selectedPair.rgbFileName}</p>
                  <p className="detailLine">Request ID: {selectedPair.requestId}</p>
                  <p className="detailLine">
                    GPS:{" "}
                    {selectedPair.latitude !== null && selectedPair.longitude !== null
                      ? `${selectedPair.latitude.toFixed(6)}, ${selectedPair.longitude.toFixed(6)}`
                      : "No GPS data"}
                  </p>
                  {selectedPair.message && <p className="detailLine">{selectedPair.message}</p>}
                </article>

                <article className="detailCard">
                  <h3 className="detailTitle">Hotspots in this image</h3>
                  {selectedPair.detections.length === 0 ? (
                    <p className="subtle">No hotspot detected by the model.</p>
                  ) : (
                    <div className="hotspotList">
                      {selectedPair.detections.map((detection, index) => (
                        <button
                          key={getMarkerId(selectedPairIndex, index)}
                          type="button"
                          className={`hotspotButton ${index === safeSelectedDetectionIndex ? "active" : ""}`}
                          onClick={() => selectDetection(selectedPairIndex, index)}
                        >
                          <span className="hotspotName">Hotspot {index + 1}</span>
                          <span className="hotspotMeta">{getHotspotSummary(detection)}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </article>

                <article className="detailCard">
                  <h3 className="detailTitle">Selected hotspot detail</h3>
                  {selectedDetection ? (
                    <div className="detailStack">
                      <p className="detailLine">Equipment: {getEquipmentLabel(selectedDetection)}</p>
                      <p className="detailLine">Temperature: {getTemperatureDetail(selectedDetection)}</p>

                      {/* [เพิ่มใหม่]
                         แสดงค่า reference เฉพาะ hotspot ที่กำลังเลือก
                      */}
                      {typeof selectedDetection.reference_temp === "number" && (
                        <p className="detailLine">
                          Reference: {selectedDetection.reference_temp.toFixed(1)} {DEGREE_C}
                        </p>
                      )}

                      {/* [เพิ่มใหม่]
                         แสดงค่า rise above reference
                      */}
                      {typeof selectedDetection.delta_above_reference === "number" && (
                        <p className="detailLine">
                          Rise above reference: {selectedDetection.delta_above_reference.toFixed(1)} {DEGREE_C}
                        </p>
                      )}

                      {/* [เพิ่มใหม่]
                         แสดงวิธี match และระยะ
                      */}
                      <p className="detailLine">
                        Match: {selectedDetection.match_method ?? "unknown"}
                        {typeof selectedDetection.match_distance === "number"
                          ? ` (${selectedDetection.match_distance.toFixed(1)} px)`
                          : ""}
                      </p>

                      {/* [เพิ่มใหม่]
                         แสดง priority และ action */}
                      <p className="detailLine">Priority: {selectedDetection.priority ?? "Not rated"}</p>
                      <p className="detailLine">
                        Action: {selectedDetection.action_required ?? "No action suggested"}
                      </p>

                      {/* [เพิ่มใหม่]
                         confidence ของ equipment model */}
                      {typeof selectedDetection.equipment_confidence === "number" && (
                        <p className="detailLine">
                          Equipment confidence: {selectedDetection.equipment_confidence.toFixed(2)}
                        </p>
                      )}
                    </div>
                  ) : (
                    <p className="subtle">Select a hotspot from the list or from the map.</p>
                  )}
                </article>
              </div>
            </div>
          )}
        </section>
      )}

      {/* [เปลี่ยนสำคัญ]
         เดิม MapView รับ lat/lon เพียงจุดเดียว
         ใหม่ MapView รับ markers หลายจุด และ sync กับผลวิเคราะห์ที่เลือก
      */}
      {hotspotMarkers.length > 0 && (
        <section className="card mapCard">
          <div className="sectionHeader">
            <div>
              <h2 className="mapTitle">Detected Location</h2>
              <p className="subtle">
                Each analyzed image is shown as one map point. Open the popup to see every hotspot found in that image.
              </p>
            </div>
          </div>

          <MapView
            markers={hotspotMarkers}
            selectedMarkerId={selectedMarkerId}
            onSelectMarker={handleMarkerSelect}
          />

          {/* [เพิ่มใหม่]
             กล่องสรุปข้อมูลของ marker / pair ที่กำลังเลือกบนแผนที่
          */}
          {selectedMapResult && (
            <div className="mapSummary">
              <p className="mapSummaryTitle">{selectedMapResult.displayName}</p>
              <p className="mapSummaryLine">
                GPS: {selectedMapResult.latitude?.toFixed(6)}, {selectedMapResult.longitude?.toFixed(6)}
              </p>
              <p className="mapSummaryLine">Hotspots: {selectedMapResult.detections.length}</p>

              <div className="mapSummaryList">
                {selectedMapResult.detections.map((detection, index) => (
                  <div key={`${selectedMapResult.id}-${index}`} className="mapSummaryHotspot">
                    <p className="mapSummaryLine">
                      <strong>Hotspot {index + 1}</strong>
                    </p>
                    <p className="mapSummaryLine">Equipment: {getEquipmentLabel(detection)}</p>
                    <p className="mapSummaryLine">Temperature: {getTemperatureDetail(detection)}</p>
                    <p className="mapSummaryLine">Priority: {detection.priority ?? "Not rated"}</p>
                    <p className="mapSummaryLine">
                      Action: {detection.action_required ?? "No action suggested"}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </main>
  );
}