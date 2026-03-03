"use client";

import { useEffect, useState, type ChangeEvent } from "react";
import dynamic from "next/dynamic";

const MapView = dynamic(() => import("./MapView"), { ssr: false });

type MatchMethod = "inside" | "nearest" | "unknown";

type Detection = {
  bbox: [number, number, number, number];
  thermal_bbox?: [number, number, number, number];
  hotspot_confidence?: number | null;
  hotspot_center?: [number, number] | null;
  max_temp: number | null;
  min_temp: number | null;
  avg_temp: number | null;
  max_raw?: number | null;
  min_raw?: number | null;
  avg_raw?: number | null;
  max_point?: [number, number] | null;
  min_point?: [number, number] | null;
  equipment_class?: string | null;
  equipment_confidence?: number | null;
  equipment_bbox?: [number, number, number, number] | null;
  match_method?: MatchMethod | null;
  match_distance?: number | null;
  reference_temp?: number | null;
  delta_above_reference?: number | null;
  priority?: string | null;
  action_required?: string | null;
};

type ThermalMode = "none" | "absolute" | "relative";

const backendBaseUrl = (process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

function formatNumber(value: number | null | undefined, digits = 1) {
  return typeof value === "number" ? value.toFixed(digits) : null;
}

function formatElapsedTime(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function createRequestId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  }
  return Math.random().toString(36).slice(2, 10);
}

function getResponseRequestId(responseData: unknown, headerRequestId: string) {
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

function describeBackendStep(step: string | null | undefined, details: Record<string, unknown> | null | undefined) {
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

export default function Home() {
  const [lat, setLat] = useState<number | null>(null);
  const [lon, setLon] = useState<number | null>(null);
  const [message, setMessage] = useState<string>("");
  const [progressMessage, setProgressMessage] = useState<string>("");
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [requestId, setRequestId] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [thermalFile, setThermalFile] = useState<File | null>(null);
  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [thermalFileName, setThermalFileName] = useState<string>("No thermal file chosen");
  const [rgbFileName, setRgbFileName] = useState<string>("No RGB file chosen");
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [thermalAvailable, setThermalAvailable] = useState<boolean | null>(null);
  const [thermalError, setThermalError] = useState<string>("");
  const [thermalMode, setThermalMode] = useState<ThermalMode | null>(null);
  const [referenceTemperature, setReferenceTemperature] = useState<number | null>(null);

  const degreeCelsius = "\u00B0C";

  function resetResults() {
    setAnnotatedImage(null);
    setDetections([]);
    setLat(null);
    setLon(null);
    setThermalAvailable(null);
    setThermalError("");
    setThermalMode(null);
    setReferenceTemperature(null);
    setRequestId("");
    setElapsedSeconds(0);
  }

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
        if (!isActive || !progressData?.success) {
          return;
        }

        if (typeof progressData.elapsed_seconds === "number") {
          setElapsedSeconds(Math.max(0, Math.round(progressData.elapsed_seconds)));
        }

        const backendStepMessage = describeBackendStep(
          typeof progressData.step === "string" ? progressData.step : null,
          typeof progressData.details === "object" && progressData.details !== null
            ? (progressData.details as Record<string, unknown>)
            : null,
        );
        if (backendStepMessage) {
          setProgressMessage(backendStepMessage);
        }
      } catch {
        // Keep the last known progress message while the active request is still running.
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

  function handleThermalFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setThermalFile(selectedFile);
    setThermalFileName(selectedFile?.name ?? "No thermal file chosen");
    setMessage("");
  }

  function handleRgbFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setRgbFile(selectedFile);
    setRgbFileName(selectedFile?.name ?? "No RGB file chosen");
    setMessage("");
  }

  async function handleUpload() {
    if (!thermalFile || !rgbFile) {
      setMessage("Please choose both thermal and RGB images.");
      return;
    }

    setMessage("");
    setProgressMessage("Uploading thermal image...");
    setLoading(true);
    resetResults();

    try {
      const uploadSingleFile = async (file: File, kind: "thermal" | "rgb", existingFileId?: string) => {
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

        if (!uploadResponse.ok || !responseData?.success) {
          const fallbackMessage = uploadResponse.ok
            ? `Failed to upload ${kind} image.`
            : `Backend returned HTTP ${uploadResponse.status} while uploading ${kind} image.`;
          throw {
            requestId: responseRequestId,
            message:
              typeof responseData?.message === "string" && responseData.message.trim()
                ? responseData.message
                : fallbackMessage,
          };
        }

        return {
          fileId: typeof responseData.file_id === "string" ? responseData.file_id : existingFileId ?? "",
          requestId: responseRequestId,
        };
      };

      const thermalUpload = await uploadSingleFile(thermalFile, "thermal");
      setRequestId(thermalUpload.requestId);

      setProgressMessage("Uploading RGB image...");
      const rgbUpload = await uploadSingleFile(rgbFile, "rgb", thermalUpload.fileId);
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
      if (!analyzeResponse.ok || !responseData?.success) {
        setRequestId(responseRequestId);
        if (analyzeResponse.status === 502) {
          setMessage(
            responseRequestId
              ? `Backend returned 502 while analyzing request ${responseRequestId}.`
              : "Backend returned 502 while analyzing the uploaded images.",
          );
        } else {
          const fallbackMessage = analyzeResponse.ok
            ? "Analysis failed. Please try again."
            : `Backend returned HTTP ${analyzeResponse.status}.`;
          setMessage(
            typeof responseData?.message === "string" && responseData.message.trim()
              ? responseData.message
              : fallbackMessage,
          );
        }
        return;
      }

      setMessage(typeof responseData.message === "string" ? responseData.message : "");
      setRequestId(responseRequestId);
      setLat(typeof responseData.latitude === "number" ? responseData.latitude : null);
      setLon(typeof responseData.longitude === "number" ? responseData.longitude : null);

      if (responseData.annotated_image) {
        const annotatedImageSource = String(responseData.annotated_image);
        if (annotatedImageSource.startsWith("data:")) {
          setAnnotatedImage(annotatedImageSource);
        } else {
          setAnnotatedImage(`${backendBaseUrl}${annotatedImageSource}`);
        }
      } else {
        setAnnotatedImage(null);
      }

      if (Array.isArray(responseData.detections)) {
        setDetections(responseData.detections as Detection[]);
      } else {
        setDetections([]);
      }

      if (typeof responseData.reference_temperature === "number") {
        setReferenceTemperature(responseData.reference_temperature);
      } else {
        setReferenceTemperature(null);
      }

      if (typeof responseData.thermal_available === "boolean") {
        setThermalAvailable(responseData.thermal_available);
      } else {
        setThermalAvailable(null);
      }

      if (
        responseData.thermal_mode === "none" ||
        responseData.thermal_mode === "absolute" ||
        responseData.thermal_mode === "relative"
      ) {
        setThermalMode(responseData.thermal_mode as ThermalMode);
      } else {
        setThermalMode(null);
      }

      if (typeof responseData.thermal_error === "string" && responseData.thermal_error.trim()) {
        setThermalError(responseData.thermal_error);
      } else {
        setThermalError("");
      }
    } catch (error) {
      if (error instanceof TypeError) {
        setMessage("Backend connection dropped during upload or analysis. Check Render backend logs and request ID.");
      } else if (typeof error === "object" && error !== null && "message" in error) {
        const requestIdFromError =
          "requestId" in error && typeof error.requestId === "string" ? error.requestId : "";
        setRequestId(requestIdFromError);
        const errorMessage =
          typeof error.message === "string" && error.message.trim() ? error.message : "Cannot reach backend. The request did not complete.";
        setMessage(errorMessage === "Failed to fetch" ? "Backend connection dropped during upload or analysis." : errorMessage);
      } else {
        setMessage("Cannot reach backend. The request did not complete.");
      }
    } finally {
      setProgressMessage("");
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="card">
        <header className="hero">
          <p className="eyebrow">Thermal - RGB - GPS - Map</p>
          <h1>Thermal Hotspot Equipment Matcher</h1>
          <p className="subtle">
            Upload the thermal image with GPS metadata and its matching RGB image to identify the hotspot equipment.
          </p>
        </header>

        <div className="uploadStack">
          <div className="uploadRow">
            <span className="uploadLabel">Thermal image</span>
            <input
              id="thermal-file"
              className="fileInput"
              type="file"
              accept="image/*"
              onChange={handleThermalFileChange}
            />
            <label htmlFor="thermal-file" className="fileButton">
              Choose file
            </label>
            <span className="fileName">{thermalFileName}</span>
          </div>

          <div className="uploadRow">
            <span className="uploadLabel">RGB image</span>
            <input
              id="rgb-file"
              className="fileInput"
              type="file"
              accept="image/*"
              onChange={handleRgbFileChange}
            />
            <label htmlFor="rgb-file" className="fileButton">
              Choose file
            </label>
            <span className="fileName">{rgbFileName}</span>
          </div>

          <button
            className="analyzeButton"
            type="button"
            onClick={handleUpload}
            disabled={loading || !thermalFile || !rgbFile}
          >
            {loading ? "Analyzing..." : "Analyze Pair"}
          </button>
        </div>

        {loading && <p className="status">Uploading files and analyzing the image pair...</p>}
        {loading && <p className="status subtleStatus">The status below is pulled from the backend in real time.</p>}
        {loading && progressMessage && <p className="status progress">{progressMessage}</p>}
        {loading && <p className="status subtleStatus">Elapsed: {formatElapsedTime(elapsedSeconds)}</p>}
        {requestId && <p className="status subtleStatus">Request ID: {requestId}</p>}
        {message && <p className="status warning">{message}</p>}
      </section>

      {annotatedImage && (
        <section className="card mapCard">
          <h2 className="mapTitle">Thermal Hotspot Result</h2>

          <img
            src={annotatedImage}
            alt="Annotated thermal hotspot result"
            style={{
              width: "100%",
              borderRadius: "12px",
              border: "1px solid rgba(0,0,0,0.1)",
              marginBottom: "16px",
            }}
          />

          {thermalAvailable === false && (
            <p className="status warning">
              {thermalMode === "relative"
                ? `Absolute temperature unavailable: ${thermalError || "Relative hotspot points are shown only."}`
                : `Temperature extraction unavailable: ${thermalError || "RawThermalImage metadata not found."}`}
            </p>
          )}

          {referenceTemperature !== null && (
            <p className="subtle">
              Reference temperature: {referenceTemperature.toFixed(1)} {degreeCelsius} (mean of pixels at or below 28
              {degreeCelsius})
            </p>
          )}

          <h3>Detected Hotspots</h3>

          {detections.length === 0 ? (
            <p className="subtle">No hotspot detected by model.</p>
          ) : (
            <ul>
              {detections.map((detection, index) => (
                <li key={index}>
                  <strong>Hotspot #{index + 1}</strong>
                  <br />
                  Equipment: {detection.equipment_class ?? "unknown"}
                  {typeof detection.equipment_confidence === "number"
                    ? ` (${detection.equipment_confidence.toFixed(2)})`
                    : ""}
                  <br />
                  Match: {detection.match_method ?? "unknown"}
                  {typeof detection.match_distance === "number" ? `, distance ${detection.match_distance.toFixed(1)} px` : ""}
                  <br />
                  {detection.max_temp !== null && detection.min_temp !== null && detection.avg_temp !== null ? (
                    <>
                      Max: {detection.max_temp.toFixed(1)} {degreeCelsius}
                      <br />
                      Min: {detection.min_temp.toFixed(1)} {degreeCelsius}
                      <br />
                      Avg: {detection.avg_temp.toFixed(1)} {degreeCelsius}
                      <br />
                      {typeof detection.reference_temp === "number" && (
                        <>
                          Reference: {detection.reference_temp.toFixed(1)} {degreeCelsius}
                          <br />
                        </>
                      )}
                      {typeof detection.delta_above_reference === "number" && (
                        <>
                          Max rise above reference: {detection.delta_above_reference.toFixed(1)} {degreeCelsius}
                          <br />
                        </>
                      )}
                    </>
                  ) : detection.max_raw !== null &&
                    detection.max_raw !== undefined &&
                    detection.min_raw !== null &&
                    detection.min_raw !== undefined &&
                    detection.avg_raw !== null &&
                    detection.avg_raw !== undefined ? (
                    <>
                      Max raw: {formatNumber(detection.max_raw)}
                      <br />
                      Min raw: {formatNumber(detection.min_raw)}
                      <br />
                      Avg raw: {formatNumber(detection.avg_raw)}
                      <br />
                    </>
                  ) : (
                    <em>Temperature data unavailable</em>
                  )}
                  {detection.priority && (
                    <>
                      Priority: {detection.priority}
                      <br />
                    </>
                  )}
                  {detection.action_required && <>Action required: {detection.action_required}</>}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {lat !== null && lon !== null && (
        <section className="card mapCard">
          <h2 className="mapTitle">Detected Location</h2>
          <MapView lat={lat} lon={lon} />
        </section>
      )}
    </main>
  );
}
