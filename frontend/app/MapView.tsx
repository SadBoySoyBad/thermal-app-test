"use client";

import * as Leaflet from "leaflet";
import { MapContainer, Marker, Popup, TileLayer, useMap } from "react-leaflet";
import { useEffect, useRef, type ComponentProps } from "react";

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

type PopupCapableMarker = {
  openPopup: () => void;
  closePopup: () => void;
};

type DivIconFactory = {
  DivIcon: new (options: {
    className: string;
    html: string;
    iconSize: [number, number];
    iconAnchor: [number, number];
    popupAnchor: [number, number];
  }) => unknown;
};

type MapController = {
  setView: (center: [number, number], zoom: number, options?: { animate?: boolean }) => void;
  getZoom: () => number;
  fitBounds: (bounds: [number, number][], options?: { padding?: [number, number] }) => void;
};

type MarkerIconLike = NonNullable<ComponentProps<typeof Marker>["icon"]>;
type PopupPropsLike = ComponentProps<typeof Popup>;

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

function createMarkerIcon(priority: string | null, selected: boolean): MarkerIconLike {
  const markerClass = getPriorityClass(priority);
  const leafletRuntime = Leaflet as unknown as DivIconFactory;

  return new leafletRuntime.DivIcon({
    className: "mapMarkerShell",
    html: `<span class="mapMarker ${markerClass} ${selected ? "selected" : ""}"></span>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    popupAnchor: [0, -28],
  }) as MarkerIconLike;
}

function FocusMap({
  markers,
  selectedMarkerId,
}: {
  markers: MapMarkerItem[];
  selectedMarkerId: string | null;
}) {
  const map = useMap() as unknown as MapController;

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
  const selectedMarker = markers.find((marker) => marker.id === selectedMarkerId) ?? markers[0] ?? null;
  const markerRefs = useRef<Record<string, PopupCapableMarker | null>>({});
  const popupProps = {
    className: "mapPopup",
    maxWidth: 296,
    minWidth: 248,
    keepInView: true,
    autoPanPaddingTopLeft: [24, 24],
    autoPanPaddingBottomRight: [24, 24],
  } as unknown as PopupPropsLike;

  useEffect(() => {
    if (!selectedMarker) {
      return;
    }

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
  }, [markers, selectedMarker]);

  if (markers.length === 0 || !selectedMarker) {
    return <div className="mapPlaceholder">No hotspot with GPS is available for the map yet.</div>;
  }

  return (
    <MapContainer center={[selectedMarker.lat, selectedMarker.lon]} zoom={18} className="map">
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <FocusMap markers={markers} selectedMarkerId={selectedMarkerId} />

      {markers.map((marker) => (
        <Marker
          key={marker.id}
          position={[marker.lat, marker.lon]}
          icon={createMarkerIcon(marker.priority, marker.id === selectedMarkerId)}
          ref={(markerInstance) => {
            markerRefs.current[marker.id] = markerInstance as PopupCapableMarker | null;
          }}
          eventHandlers={{
            click: () => {
              onSelectMarker?.(marker.id);
            },
          }}
        >
          <Popup {...popupProps}>
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
