"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

// โหลด MapView เฉพาะฝั่ง browser เพื่อลดปัญหา SSR กับ Leaflet
const MapView = dynamic(() => import("./MapView"), { ssr: false });

type Detection = {
  bbox: [number, number, number, number];
  max_temp: number | null;
  min_temp: number | null;
  avg_temp: number | null;
  max_raw?: number | null;
  min_raw?: number | null;
  avg_raw?: number | null;
  max_point?: [number, number] | null;
  min_point?: [number, number] | null;
};

type ThermalMode = "none" | "absolute" | "relative";

const backendBaseUrl = (process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

export default function Home() {
  // พิกัด GPS จากภาพ
  const [lat, setLat] = useState<number | null>(null);
  const [lon, setLon] = useState<number | null>(null);

  // ข้อความสถานะที่แสดงบนหน้าเว็บ
  const [message, setMessage] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [fileName, setFileName] = useState<string>("No file chosen");

  // ผลลัพธ์จาก backend
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [thermalAvailable, setThermalAvailable] = useState<boolean | null>(null);
  const [thermalError, setThermalError] = useState<string>("");
  const [thermalMode, setThermalMode] = useState<ThermalMode | null>(null);

  const degreeCelsius = "\u00B0C";

  async function handleFileUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    setFileName(selectedFile.name);
    setMessage("");
    setLoading(true);

    // รีเซ็ตผลเก่าก่อนเริ่มอัปโหลดใหม่
    setAnnotatedImage(null);
    setDetections([]);
    setLat(null);
    setLon(null);
    setThermalAvailable(null);
    setThermalError("");
    setThermalMode(null);

    const uploadFormData = new FormData();
    uploadFormData.append("file", selectedFile);

    try {
      const uploadResponse = await fetch(`${backendBaseUrl}/upload`, {
        method: "POST",
        body: uploadFormData,
      });

      if (!uploadResponse.ok) {
        setMessage("Upload failed. Please try again.");
        return;
      }

      const responseData = await uploadResponse.json();

      if (!responseData?.success) {
        setMessage(responseData?.message ?? "No GPS data found in image.");
        setLat(null);
        setLon(null);
        setThermalAvailable(null);
        setThermalError("");
        setThermalMode(null);
        return;
      }

      setLat(responseData.latitude);
      setLon(responseData.longitude);

      if (responseData.annotated_image) {
        const annotatedImageSource = String(responseData.annotated_image);
        if (annotatedImageSource.startsWith("data:")) {
          setAnnotatedImage(annotatedImageSource);
        } else {
          setAnnotatedImage(`${backendBaseUrl}${annotatedImageSource}`);
        }
      }

      if (Array.isArray(responseData.detections)) {
        setDetections(responseData.detections as Detection[]);
      } else {
        setDetections([]);
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
      setMessage("Cannot reach backend. Is it running on port 8000?");
      setLat(null);
      setLon(null);
      setThermalAvailable(null);
      setThermalError("");
      setThermalMode(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="card">
        <header className="hero">
          <p className="eyebrow">Thermal - GPS - Map</p>
          <h1>Thermal Image GPS Viewer</h1>
          <p className="subtle">
            Upload a thermal image with GPS metadata to plot its location.
          </p>
        </header>

        <div className="uploadRow">
          <input
            id="image-file"
            className="fileInput"
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
          />

          <label htmlFor="image-file" className="fileButton">
            Choose file
          </label>

          <span className="fileName">{fileName}</span>
        </div>

        {loading && <p className="status">Uploading...</p>}
        {message && <p className="status warning">{message}</p>}
      </section>

      {lat !== null && lon !== null && (
        <section className="card mapCard">
          <h2 className="mapTitle">Detected Location</h2>
          <MapView lat={lat} lon={lon} />
        </section>
      )}

      {annotatedImage && (
        <section className="card mapCard">
          <h2 className="mapTitle">Hotspot Detection Result</h2>

          <img
            src={annotatedImage}
            alt="Annotated thermal"
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

          <h3>Detected Hotspots</h3>

          {detections.length === 0 ? (
            <p className="subtle">No hotspot detected by model.</p>
          ) : (
            <ul>
              {detections.map((detection, index) => (
                <li key={index}>
                  <strong>Hotspot #{index + 1}</strong>
                  <br />
                  {detection.max_temp !== null &&
                  detection.min_temp !== null &&
                  detection.avg_temp !== null ? (
                    <>
                      Max: {detection.max_temp.toFixed(1)} {degreeCelsius}
                      <br />
                      Min: {detection.min_temp.toFixed(1)} {degreeCelsius}
                      <br />
                      Avg: {detection.avg_temp.toFixed(1)} {degreeCelsius}
                    </>
                  ) : (
                    <em>Temperature data unavailable</em>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </main>
  );
}
