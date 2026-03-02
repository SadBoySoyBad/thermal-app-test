"use client";

import { useState, type ChangeEvent } from "react";
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

export default function Home() {
  const [lat, setLat] = useState<number | null>(null);
  const [lon, setLon] = useState<number | null>(null);
  const [message, setMessage] = useState<string>("");
  const [progressMessage, setProgressMessage] = useState<string>("");
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
  }

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
    setProgressMessage("Uploading files to backend...");
    setLoading(true);
    resetResults();

    const uploadFormData = new FormData();
    uploadFormData.append("thermal_file", thermalFile);
    uploadFormData.append("rgb_file", rgbFile);

    const progressSteps = [
      "Uploading files to backend...",
      "Waiting for backend to accept the upload...",
      "Backend is still processing the request...",
      "Still waiting for backend response. Check Render logs if this takes too long...",
      "Processing is taking longer than usual...",
    ];
    let progressIndex = 0;
    setProgressMessage(progressSteps[progressIndex]);
    const progressTimer = window.setInterval(() => {
      progressIndex = Math.min(progressIndex + 1, progressSteps.length - 1);
      setProgressMessage(progressSteps[progressIndex]);
    }, 3500);

    try {
      const uploadResponse = await fetch(`${backendBaseUrl}/upload`, {
        method: "POST",
        body: uploadFormData,
      });

      const responseData = await uploadResponse.json().catch(() => null);
      const headerRequestId = uploadResponse.headers.get("x-request-id") ?? "";
      if (!uploadResponse.ok || !responseData?.success) {
        const responseRequestId =
          typeof responseData?.request_id === "string" && responseData.request_id.trim()
            ? responseData.request_id
            : headerRequestId;
        setRequestId(responseRequestId);

        if (uploadResponse.status === 502) {
          setMessage(
            responseRequestId
              ? `Backend returned 502 while processing the upload. Check backend logs for request ${responseRequestId}.`
              : "Backend returned 502 while processing the upload. Check Render backend logs.",
          );
        } else {
          const fallbackMessage = uploadResponse.ok
            ? "Upload failed. Please try again."
            : `Backend returned HTTP ${uploadResponse.status}.`;
          setMessage(responseData?.detail ?? responseData?.message ?? fallbackMessage);
        }
        return;
      }

      setMessage(typeof responseData.message === "string" ? responseData.message : "");
      setRequestId(
        typeof responseData.request_id === "string" && responseData.request_id.trim()
          ? responseData.request_id
          : headerRequestId,
      );
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
    } catch {
      setMessage("Cannot reach backend. The request did not complete.");
      resetResults();
    } finally {
      window.clearInterval(progressTimer);
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

        {loading && <p className="status">Analyzing thermal and RGB images...</p>}
        {loading && <p className="status subtleStatus">The progress text below is approximate until the backend responds.</p>}
        {loading && progressMessage && <p className="status progress">{progressMessage}</p>}
        {message && <p className="status warning">{message}</p>}
        {!loading && requestId && <p className="status subtleStatus">Request ID: {requestId}</p>}
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
