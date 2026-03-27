"use client";

// เพิ่ม useEffect เพราะหน้าใหม่มี polling progress และจับเวลา
// เพิ่ม ChangeEvent เพื่อระบุ type ของ event ตอนเลือกไฟล์ให้ชัดเจน
import { useEffect, useState, type ChangeEvent } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import type { AnalysisResult, FailedPair, MatchedPair, PairingIssue } from "./uploadUtils";
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

const MapView = dynamic(() => import("./MapView"), { ssr: false });

type MessageTone = "default" | "warning";

const backendBaseUrl = (process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

// helper เหมือนเดิม: แปลงวินาที -> mm:ss
function formatElapsedTime(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export default function Home() {
  // ------------------------------
  // State กลุ่มข้อความ/สถานะหน้า
  // ------------------------------
  const [message, setMessage] = useState("");
  const [messageTone, setMessageTone] = useState<MessageTone>("default");
  const [progressMessage, setProgressMessage] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [requestId, setRequestId] = useState("");
  const [loading, setLoading] = useState(false);

  // ------------------------------
  // State กลุ่มไฟล์ที่ user เลือกมา
  // ------------------------------
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedFileLabel, setSelectedFileLabel] = useState("No images chosen");
  const [matchedPairs, setMatchedPairs] = useState<MatchedPair[]>([]);
  const [pairingIssues, setPairingIssues] = useState<PairingIssue[]>([]);

  // ------------------------------
  // State กลุ่มผลลัพธ์จาก backend
  // ------------------------------
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [failedPairs, setFailedPairs] = useState<FailedPair[]>([]);
  const [selectedPairIndex, setSelectedPairIndex] = useState(0);
  const [selectedDetectionIndex, setSelectedDetectionIndex] = useState(0);
  const [activePairIndex, setActivePairIndex] = useState(0);
  const [activePairTotal, setActivePairTotal] = useState(0);
  const [activePairLabel, setActivePairLabel] = useState("");
  const [isMobileViewport, setIsMobileViewport] = useState(false);

  const selectedPair = results[selectedPairIndex] ?? null;
  const safeSelectedDetectionIndex =
    selectedPair && selectedPair.detections.length > 0
      ? Math.min(selectedDetectionIndex, selectedPair.detections.length - 1)
      : 0;
  const selectedDetection = selectedPair?.detections[safeSelectedDetectionIndex] ?? null;
  const hotspotMarkers = buildMapMarkers(results);
  const selectedMapResult =
    selectedPair && selectedPair.latitude !== null && selectedPair.longitude !== null
      ? selectedPair
      : results.find((result) => result.latitude !== null && result.longitude !== null) ?? null;
  const selectedMarkerId = selectedMapResult ? selectedMapResult.id : null;
  const totalHotspots = results.reduce((count, result) => count + result.detections.length, 0);
  const selectionRailClassName = `selectionRail ${isMobileViewport ? "selectionRailMobile" : "selectionRailDesktop"}`;
  const selectionChipClassName = `selectionChip ${isMobileViewport ? "selectionChipMobile" : "selectionChipDesktop"}`;

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
      setProgressMessage("");
      setLoading(false);
    }
  }

  // ปุ่ม Previous / Next ของรูปหลัก
  function selectPair(nextIndex: number) {
    if (results.length === 0) {
      return;
    }

    const normalizedIndex = (nextIndex + results.length) % results.length;
    setSelectedPairIndex(normalizedIndex);
    setSelectedDetectionIndex(0);
  }

  // เลือก hotspot ที่กำลังดูอยู่
  function selectDetection(pairIndex: number, detectionIndex: number) {
    setSelectedPairIndex(pairIndex);
    setSelectedDetectionIndex(detectionIndex);
  }

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
          <p className="eyebrow">Thermal - GPS - Map</p>
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

        {selectedFiles.length > 0 && (
          <div className="summaryBar">
            <span>{selectedFiles.length} uploaded images</span>
            <span>{matchedPairs.length} matched pairs</span>
            <span>{pairingIssues.length} groups need attention</span>
          </div>
        )}

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

        {loading && activePairTotal > 0 && (
          <p className="status">
            Pair {activePairIndex} of {activePairTotal}: {activePairLabel}
          </p>
        )}
        {loading && progressMessage && <p className="status progress">{progressMessage}</p>}
        {loading && <p className="status subtleStatus">Elapsed: {formatElapsedTime(elapsedSeconds)}</p>}
        {requestId && <p className="status subtleStatus">Request ID: {requestId}</p>}
        {message && <p className={`status ${messageTone === "warning" ? "warning" : ""}`}>{message}</p>}
      </section>

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
            <div className="navigatorButtons">
              <button className="navButton" type="button" onClick={() => selectPair(selectedPairIndex - 1)}>
                Previous
              </button>
              <button className="navButton" type="button" onClick={() => selectPair(selectedPairIndex + 1)}>
                Next
              </button>
            </div>
          </div>

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

                {selectedPair.referenceTemperature !== null && (
                  <p className="subtle">
                    Reference temperature: {selectedPair.referenceTemperature.toFixed(1)} {DEGREE_C}
                  </p>
                )}
              </div>

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
                      {typeof selectedDetection.reference_temp === "number" && (
                        <p className="detailLine">
                          Reference: {selectedDetection.reference_temp.toFixed(1)} {DEGREE_C}
                        </p>
                      )}
                      {typeof selectedDetection.delta_above_reference === "number" && (
                        <p className="detailLine">
                          Rise above reference: {selectedDetection.delta_above_reference.toFixed(1)} {DEGREE_C}
                        </p>
                      )}
                      <p className="detailLine">
                        Match: {selectedDetection.match_method ?? "unknown"}
                        {typeof selectedDetection.match_distance === "number"
                          ? ` (${selectedDetection.match_distance.toFixed(1)} px)`
                          : ""}
                      </p>
                      <p className="detailLine">Priority: {selectedDetection.priority ?? "Not rated"}</p>
                      <p className="detailLine">
                        Action: {selectedDetection.action_required ?? "No action suggested"}
                      </p>
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
