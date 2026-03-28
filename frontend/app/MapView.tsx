"use client";

// หน้า map นี้รันเฉพาะฝั่ง browser
// เพราะ react-leaflet ต้องใช้ window / document
import { DivIcon, type Marker as LeafletMarker } from "leaflet";
import { MapContainer, Marker, Popup, TileLayer, useMap } from "react-leaflet";
import { useEffect, useRef } from "react";

// marker 1 จุด = 1 รูปบนแผนที่
export type MapMarkerItem = {
  id: string;
  lat: number;
  lon: number;
  pairLabel: string;
  priority: string | null;
  hotspots: Array<{
    hotspotLabel: string;
    equipmentLabel: string;
    temperatureLabel: string;
    priority: string | null;
    actionRequired: string | null;
  }>;
};

type Props = {
  markers: MapMarkerItem[];
  selectedMarkerId: string | null;
  onSelectMarker?: (markerId: string) => void;
};

// map priority -> สี marker
function getPriorityClass(priority: string | null) {
  const normalized = (priority ?? "").toLowerCase();
  if (normalized.includes("priority 1")) {
    return "mapMarkerP1";
  }
  if (normalized.includes("priority 2")) {
    return "mapMarkerP2";
  }
  if (normalized.includes("priority 3")) {
    return "mapMarkerP3";
  }
  return "mapMarkerDefault";
}

function getPopupPriorityClass(priority: string | null) {
  const normalized = (priority ?? "").toLowerCase();
  if (normalized.includes("priority 1")) {
    return "mapPopupPriority mapPopupPriorityP1";
  }
  if (normalized.includes("priority 2")) {
    return "mapPopupPriority mapPopupPriorityP2";
  }
  if (normalized.includes("priority 3")) {
    return "mapPopupPriority mapPopupPriorityP3";
  }
  return "mapPopupPriority mapPopupPriorityDefault";
}

// ใช้ DivIcon เพื่อให้ปรับสี/สถานะ selected ผ่าน CSS ได้ง่าย
function createMarkerIcon(priority: string | null, selected: boolean) {
  const markerClass = getPriorityClass(priority);
  return new DivIcon({
    className: "mapMarkerShell",
    html: `<span class="mapMarker ${markerClass} ${selected ? "selected" : ""}"></span>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    popupAnchor: [0, -28],
  });
}

// sync มุมมองแผนที่ตาม marker ที่กำลังเลือกอยู่
function FocusMap({
  markers,
  selectedMarkerId,
}: {
  markers: MapMarkerItem[];
  selectedMarkerId: string | null;
}) {
  const map = useMap();

  useEffect(() => {
    if (markers.length === 0) {
      return;
    }

    const selectedMarker = markers.find((marker) => marker.id === selectedMarkerId) ?? null;

    if (selectedMarker) {
      map.setView([selectedMarker.lat, selectedMarker.lon], Math.max(map.getZoom(), 18), {
        animate: true,
      });
      return;
    }

    if (markers.length === 1) {
      map.setView([markers[0].lat, markers[0].lon], 18);
      return;
    }

    map.fitBounds(
      markers.map((marker) => [marker.lat, marker.lon] as [number, number]),
      { padding: [36, 36] },
    );
  }, [map, markers, selectedMarkerId]);

  return null;
}

export default function MapView({ markers, selectedMarkerId, onSelectMarker }: Props) {
  // ถ้ายังไม่มี hotspot ที่มี GPS ก็ไม่ต้อง render Leaflet
  if (markers.length === 0) {
    return <div className="mapPlaceholder">No hotspot with GPS is available for the map yet.</div>;
  }

  const selectedMarker = markers.find((marker) => marker.id === selectedMarkerId) ?? markers[0];
  const markerRefs = useRef<Record<string, LeafletMarker | null>>({});

  useEffect(() => {
    for (const marker of markers) {
      const markerInstance = markerRefs.current[marker.id];
      if (!markerInstance) {
        continue;
      }

      if (marker.id === selectedMarker.id) {
        markerInstance.openPopup();
      } else {
        markerInstance.closePopup();
      }
    }
  }, [markers, selectedMarker.id]);

  return (
    // ใช้ marker 1 ตัวแทนภาพ 1 รูป
    <MapContainer center={[selectedMarker.lat, selectedMarker.lon]} zoom={18} className="map">
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <FocusMap markers={markers} selectedMarkerId={selectedMarkerId} />

      {markers.map((marker) => (
        <Marker
          key={marker.id}
          position={[marker.lat, marker.lon]}
          icon={createMarkerIcon(marker.priority, marker.id === selectedMarkerId)}
          ref={(markerInstance) => {
            markerRefs.current[marker.id] = markerInstance;
          }}
          eventHandlers={{
            click: () => {
              onSelectMarker?.(marker.id);
            },
          }}
        >
          <Popup
            className="mapPopup"
            maxWidth={296}
            minWidth={248}
            keepInView
            autoPanPaddingTopLeft={[24, 24]}
            autoPanPaddingBottomRight={[24, 24]}
          >
            <div className="mapPopupBody">
              <div className="mapPopupHeader">
                <h3 className="mapPopupTitle">{marker.pairLabel}</h3>
                <span className="mapPopupCount">{marker.hotspots.length} hotspots</span>
              </div>
              <div className="mapPopupHotspotList">
                {marker.hotspots.map((hotspot) => (
                  <section key={`${marker.id}-${hotspot.hotspotLabel}`} className="mapPopupHotspotCard">
                    <div className="mapPopupHotspotHeader">
                      <p className="mapPopupHotspotName">{hotspot.hotspotLabel}</p>
                      <span className={getPopupPriorityClass(hotspot.priority)}>
                        {hotspot.priority ?? "Not rated"}
                      </span>
                    </div>
                    <div className="mapPopupFactList">
                      <p className="mapPopupFactRow">
                        <span className="mapPopupFactInlineLabel">Equipment:</span>
                        <span className="mapPopupFactInlineValue">{hotspot.equipmentLabel}</span>
                      </p>
                      <p className="mapPopupFactRow">
                        <span className="mapPopupFactInlineLabel">Temperature:</span>
                        <span className="mapPopupFactInlineValue">{hotspot.temperatureLabel}</span>
                      </p>
                      <p className="mapPopupFactRow">
                        <span className="mapPopupFactInlineLabel">Action:</span>
                        <span className="mapPopupFactInlineValue">
                          {hotspot.actionRequired ?? "No action suggested"}
                        </span>
                      </p>
                    </div>
                  </section>
                ))}
              </div>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
