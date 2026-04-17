"use client";
// บอก Next.js ว่าไฟล์นี้ต้องรันฝั่ง Browser (ไม่ใช่ฝั่ง Server)
// เพราะ Leaflet ใช้ window / document ซึ่ง Server ใช้ไม่ได้

// =============================
// ส่วนที่ 1: import library ที่จำเป็น
// ส่วนนี้คือการดึงเครื่องมือที่ไฟล์นี้ต้องใช้เข้ามา
// ถ้าพูดแบบภาษาคนทั่วไป ส่วนนี้คือการ "หยิบอุปกรณ์เข้ากล่องทำงาน" ก่อนเริ่มใช้งานจริง
// =============================

// นำทุกอย่างจาก leaflet มาใช้ในชื่อ Leaflet
// ใช้ในไฟล์นี้เพื่อสร้าง DivIcon แบบ custom สำหรับหมุด
import * as Leaflet from "leaflet";

// นำ component ที่จำเป็นมาจาก react-leaflet
// MapContainer = กล่องแผนที่
// TileLayer = ชั้นแผนที่ (พื้นหลัง)
// Marker = หมุดปักตำแหน่ง
// Popup = กล่องข้อมูลที่เด้งขึ้นมาเมื่อกดหรือเลือกหมุด
// useMap = ใช้เข้าถึงตัวแผนที่โดยตรง เพื่อสั่งเลื่อนหรือซูมแผนที่ได้
import { MapContainer, Marker, Popup, TileLayer, useMap } from "react-leaflet";

// นำ hook และ type จาก React มาใช้
// useEffect = ใช้ทำงานบางอย่างตอน component render หรือเมื่อค่ามีการเปลี่ยน
// useState = ใช้เก็บสถานะภายใน component
// ComponentProps = ใช้ดึง type ของ props จาก component อื่นมาใช้อ้างอิง
import { useEffect, useState, type ComponentProps } from "react";

// =============================
// ส่วนที่ 2: รูปแบบข้อมูลที่ component นี้ใช้
// ส่วนนี้คือการกำหนดว่า "ข้อมูลหน้าตาแบบไหนถึงจะส่งเข้ามาได้"
// =============================

// ข้อมูลของหมุด 1 จุดบนแผนที่
// 1 หมุดจะเก็บตำแหน่ง ชื่อคู่ภาพ ระดับความสำคัญ
// และรายการ hotspot ที่อยู่ในจุดนั้น
export type MapMarkerItem = {
  id: string; // id ของหมุด ใช้แยกว่าหมุดไหนคือหมุดไหน
  lat: number; // ละติจูด
  lon: number; // ลองจิจูด
  pairLabel: string; // ชื่อที่ใช้แสดงของภาพคู่นั้น
  priority: string | null; // ระดับความสำคัญรวมของหมุด เช่น Priority 1
  hotspots: Array<{
    hotspotLabel: string; // ชื่อ hotspot
    equipmentLabel: string; // ชื่ออุปกรณ์ที่เกี่ยวข้อง
    temperatureLabel: string; // ข้อความอุณหภูมิ เช่น 52.1 °C
    priority: string | null; // ระดับความสำคัญของ hotspot นี้
    actionRequired: string | null; // สิ่งที่ควรทำ เช่น Immediate repair
  }>;
};

// กำหนดชนิดของข้อมูล (Props) ที่ component นี้ต้องรับเข้ามา
// ภาษาคนทั่วไป: ข้อมูลหลักที่หน้าจอแผนที่ต้องใช้ในการทำงาน
// - markers = รายการหมุดทั้งหมด
// - selectedMarkerId = หมุดที่กำลังถูกเลือก
// - onSelectMarker = ตัวส่งสัญญาณกลับเมื่อผู้ใช้กดเลือกหมุด
 type Props = {
  markers: MapMarkerItem[]; // รายการหมุดทั้งหมดที่จะเอาไปแสดงบนแผนที่
  selectedMarkerId: string | null; // id ของหมุดที่กำลังถูกเลือกอยู่
  onSelectMarker?: (markerId: string) => void; // function ที่เรียกกลับเมื่อผู้ใช้เลือกหมุด
};

// type นี้ใช้บอกรูปแบบของ DivIcon จาก Leaflet
// เพราะเราจะสร้างหมุดแบบกำหนดหน้าตาเอง
type DivIconFactory = {
  DivIcon: new (options: {
    className: string;
    html: string;
    iconSize: [number, number];
    iconAnchor: [number, number];
    popupAnchor: [number, number];
  }) => unknown;
};

// type นี้อธิบาย object แผนที่ที่เราจะใช้จริงแค่บางเมธอด
// setView = สั่งย้ายตำแหน่งแผนที่
// getZoom = ดูระดับซูมปัจจุบัน
type MapController = {
  setView: (center: [number, number], zoom: number, options?: { animate?: boolean }) => void;
  getZoom: () => number;
};

// ดึง type ของ icon ที่ Marker ของ react-leaflet รับได้
// เอาไว้ใช้เป็นชนิดข้อมูลของหมุด custom icon
type MarkerIconLike = NonNullable<ComponentProps<typeof Marker>["icon"]>;

// =============================
// ส่วนที่ 3: ฟังก์ชันช่วยเรื่องสีและหน้าตาของหมุด
// ส่วนนี้มีหน้าที่แปลค่า priority ให้กลายเป็น class CSS
// พูดง่าย ๆ คือดูว่าหมุดสำคัญระดับไหน แล้วเลือกสี/หน้าตาให้ถูก
// =============================

// ดูว่า priority เป็นระดับไหน
// แล้วคืนชื่อ class CSS สำหรับตกแต่งสีของหมุดบนแผนที่
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

  // ถ้าไม่มี priority ชัดเจน ใช้ style ปกติ
  return "mapMarkerDefault";
}

// คล้ายกับ getPriorityClass
// แต่ฟังก์ชันนี้ใช้กับ badge หรือข้อความ priority ใน popup
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

// สร้างรูปหมุด (Marker Icon)
// ของเดิมใช้รูปมาตรฐานจากไฟล์ภาพใน public
// แต่โค้ดใหม่นี้เปลี่ยนเป็นสร้างหมุดจาก HTML + CSS แทน
// ข้อดีคือเปลี่ยนสี/สถานะ selected ได้ง่ายกว่า
function createMarkerIcon(priority: string | null, selected: boolean): MarkerIconLike {
  const markerClass = getPriorityClass(priority);
  const leafletRuntime = Leaflet as unknown as DivIconFactory;

  return new leafletRuntime.DivIcon({
    className: "mapMarkerShell",
    html: `<span class="mapMarker ${markerClass} ${selected ? "selected" : ""}"></span>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    popupAnchor: [0, -12],
  }) as MarkerIconLike;
}

// =============================
// ส่วนที่ 4: component ย่อยสำหรับเลื่อนแผนที่ไปยังหมุดที่เลือก
// พูดง่าย ๆ คือถ้าเลือกหมุดใหม่ กล้องของแผนที่จะขยับตามไปหาจุดนั้น
// =============================

// component ย่อยตัวนี้มีหน้าที่พาแผนที่เลื่อนไปหาหมุดที่ถูกเลือก
// เวลาผู้ใช้เลือก marker ใหม่ แผนที่จะขยับไปหาตำแหน่งนั้นอัตโนมัติ
function FocusMap({
  selectedMarker,
}: {
  selectedMarker: MapMarkerItem;
}) {
  const map = useMap() as unknown as MapController;

  useEffect(() => {
    map.setView([selectedMarker.lat, selectedMarker.lon], Math.max(map.getZoom(), 18), {
      animate: true,
    });
  }, [map, selectedMarker.lat, selectedMarker.lon]);

  return null;
}

// =============================
// ส่วนที่ 5: component หลักของแผนที่
// ส่วนนี้คือหัวใจของไฟล์
// ทำหน้าที่รับข้อมูลหมุดทั้งหมด แล้วแสดงเป็นแผนที่พร้อม popup
// =============================

// สร้าง component ชื่อ MapView
// รับข้อมูลหมุดทั้งหมด หมุดที่เลือกอยู่ และ function สำหรับเปลี่ยนหมุดที่เลือก
export default function MapView({ markers, selectedMarkerId, onSelectMarker }: Props) {
  // หา marker ที่กำลังถูกเลือก
  // ถ้าไม่เจอ จะใช้ marker ตัวแรกแทน
  // ถ้าไม่มี marker เลย จะได้ null
  const selectedMarker = markers.find((marker) => marker.id === selectedMarkerId) ?? markers[0] ?? null;

  // เก็บ id ของ marker ที่ถูกเลือกจริง ๆ
  const effectiveSelectedMarkerId = selectedMarker?.id ?? null;

  // เก็บ id ของ marker ที่ผู้ใช้กดปิด popup ไปแล้ว
  // เพื่อไม่ให้ popup เด้งกลับมาทันที
  const [dismissedMarkerId, setDismissedMarkerId] = useState<string | null>(null);

  // ใช้เช็กว่าหน้าจอเล็กไหม
  // ถ้าหน้าจอเล็ก จะเปิด popup แบบ compact
  const [isCompactPopup, setIsCompactPopup] = useState(false);

  // [เพิ่มใหม่ล่าสุด] เดิมเก็บแค่เลขลำดับ hotspot อย่างเดียว
  // แต่เวอร์ชันล่าสุดเปลี่ยนมาเก็บทั้ง markerId และ hotspotIndex
  // เพื่อให้ระบบรู้ว่า "ตอนนี้ผู้ใช้กำลังเปิด hotspot ของหมุดไหนอยู่"
  // ภาษาคนทั่วไป: กันอาการข้อมูลสลับกันหรือจำค่าผิดตอนผู้ใช้กดเปลี่ยนหมุดไปมา
  const [compactSelection, setCompactSelection] = useState<{ markerId: string | null; hotspotIndex: number }>({
    markerId: null,
    hotspotIndex: 0,
  });

  // เช็กขนาดหน้าจอ
  // ถ้าหน้าจอเล็กกว่า 640px จะใช้ popup แบบ compact
  useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 640px)");

    const syncPopupMode = () => {
      setIsCompactPopup(mediaQuery.matches);
    };

    syncPopupMode();

    // browser ใหม่ใช้ addEventListener
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", syncPopupMode);
      return () => {
        mediaQuery.removeEventListener("change", syncPopupMode);
      };
    }

    // browser เก่าใช้ addListener
    mediaQuery.addListener(syncPopupMode);
    return () => {
      mediaQuery.removeListener(syncPopupMode);
    };
  }, []);

  // ถ้าไม่มีหมุดเลย หรือไม่มี marker ที่เลือก
  // หรือ marker ที่เลือกไม่มี hotspot
  // ก็ไม่ render แผนที่ และแสดงข้อความแทน
  if (markers.length === 0 || !selectedMarker || selectedMarker.hotspots.length === 0) {
    return <div className="mapPlaceholder">No hotspot with GPS is available for the map yet.</div>;
  }

  // [เพิ่มใหม่ล่าสุด] เดิม logic นี้ดูจากตัวเลข hotspot อย่างเดียว
  // เวอร์ชันล่าสุดจะเช็กก่อนว่า hotspot ที่จำไว้นั้นเป็นของ marker ตัวที่กำลังเปิดอยู่จริงไหม
  // ถ้าใช่ ค่อยใช้ index เดิมต่อ
  // ถ้าไม่ใช่ ให้เริ่มกลับไปที่ hotspot แรกของ marker ใหม่ทันที
  // ภาษาคนทั่วไป: เวลาเปลี่ยนไปกดอีกหมุดหนึ่ง ระบบจะไม่เผลอเอาตำแหน่ง hotspot ของหมุดเก่ามาปนกับหมุดใหม่
  // ป้องกันไม่ให้ index หลุดเกินจำนวน hotspot จริง
  const activeCompactHotspotIndex =
    compactSelection.markerId === selectedMarker.id && isCompactPopup
      ? Math.min(compactSelection.hotspotIndex, selectedMarker.hotspots.length - 1)
      : 0;

  // ถ้าเป็น compact popup จะแสดงทีละ hotspot
  // ถ้าไม่ compact จะแสดง hotspot ทุกตัวใน popup เดียว
  const visibleHotspots = isCompactPopup
    ? [
        {
          hotspot: selectedMarker.hotspots[activeCompactHotspotIndex],
          index: activeCompactHotspotIndex,
        },
      ]
    : selectedMarker.hotspots.map((hotspot, index) => ({ hotspot, index }));

  // รวม props ของ Popup ไว้ตรงนี้
  // เพื่อไม่ให้ JSX ยาวเกินไป
  const popupProps = {
    className: isCompactPopup ? "mapPopup mapPopupCompact" : "mapPopup",
    closeButton: true,
    autoPan: true,
    keepInView: true,
    minWidth: isCompactPopup ? 260 : 312,
    maxWidth: isCompactPopup ? 292 : 356,
    autoPanPaddingTopLeft: (isCompactPopup ? [64, 16] : [88, 24]) as [number, number],
    autoPanPaddingBottomRight: (isCompactPopup ? [16, 16] : [24, 24]) as [number, number],
    offset: [0, -12] as [number, number],
    eventHandlers: {
      remove: () => {
        // ถ้าผู้ใช้ปิด popup ให้จำไว้ว่า popup ของ marker นี้ถูกปิดแล้ว
        setDismissedMarkerId(selectedMarker.id);
      },
    },
  } as ComponentProps<typeof Popup>;

  // ส่วนที่แสดงผลบนหน้าจอ
  return (
    // div ครอบภายนอกของแผนที่
    <div className="mapViewport">
      {/* MapContainer คือกรอบแผนที่ทั้งหมด */}
      <MapContainer
        center={[selectedMarker.lat, selectedMarker.lon]} // ตำแหน่งศูนย์กลางแผนที่
        zoom={18} // ระดับการซูม (ยิ่งมาก ยิ่งใกล้)
        className="map" // class สำหรับตกแต่งด้วย CSS
      >
        {/* TileLayer คือพื้นแผนที่ */}
        {/* ใช้แผนที่ฟรีจาก OpenStreetMap */}
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        {/* component นี้ช่วยสั่งให้แผนที่เลื่อนไปยังหมุดที่เลือก */}
        <FocusMap selectedMarker={selectedMarker} />

        {/* วนสร้าง Marker สำหรับทุกหมุดในข้อมูล */}
        {markers.map((marker) => (
          <Marker
            key={marker.id}
            position={[marker.lat, marker.lon]} // position ใช้ lat, lon ของ marker นั้น
            icon={createMarkerIcon(marker.priority, marker.id === effectiveSelectedMarkerId)} // ใช้หมุด custom ที่เปลี่ยนสีตาม priority ได้
            eventHandlers={{
              click: () => {
                // เมื่อกดหมุด ให้เปิด popup ใหม่และแจ้ง parent ว่าเลือก marker ตัวนี้
                setDismissedMarkerId(null);
                onSelectMarker?.(marker.id);
              },
            }}
          />
        ))}

        {/* แสดง popup เฉพาะตอนที่ popup ของ marker นี้ยังไม่ถูกปิด */}
        {dismissedMarkerId !== selectedMarker.id && (
          <Popup
            key={`popup-${selectedMarker.id}`}
            position={[selectedMarker.lat, selectedMarker.lon]}
            {...popupProps}
          >
            {/* เนื้อหาทั้งหมดใน popup */}
            <div className="mapPopupBody">
              {/* ส่วนหัวของ popup */}
              <header className="mapPopupIntro">
                <h3 className="mapPopupTitle">{selectedMarker.pairLabel}</h3>
                <p className="mapPopupSubtitle">
                  {selectedMarker.hotspots.length} hotspot{selectedMarker.hotspots.length === 1 ? "" : "s"} in this
                  image
                </p>
              </header>

              {/* ถ้าเป็นหน้าจอเล็ก จะแสดงปุ่มเลือก hotspot เป็นแถบด้านบน */}
              {isCompactPopup && (
                <div className="mapPopupCompactRail" role="tablist" aria-label="Hotspot selector">
                  {selectedMarker.hotspots.map((hotspot, hotspotIndex) => (
                    <button
                      key={`${selectedMarker.id}-chip-${hotspotIndex}`}
                      type="button"
                      role="tab"
                      aria-selected={hotspotIndex === activeCompactHotspotIndex}
                      className={`mapPopupCompactRailButton ${hotspotIndex === activeCompactHotspotIndex ? "selected" : ""}`}
                      // [เพิ่มใหม่ล่าสุด] เดิมตอนกดเลือก hotspot จะจำแค่เลขลำดับ
                      // ตอนนี้เปลี่ยนเป็นจำทั้งหมุดที่เปิดอยู่และลำดับ hotspot ที่เลือก
                      // ผลคือ popup แบบ mobile/compact จะจำข้อมูลได้ถูกต้องว่าเลือกของหมุดไหนอยู่
                      onClick={() => setCompactSelection({ markerId: selectedMarker.id, hotspotIndex })}
                    >
                      {hotspot.hotspotLabel || `Hotspot ${hotspotIndex + 1}`}
                    </button>
                  ))}
                </div>
              )}

              {/* พื้นที่เนื้อหาใน popup ที่เลื่อนได้ */}
              <div key={`scroll-${selectedMarker.id}`} className="mapPopupScroll">
                {visibleHotspots.map(({ hotspot, index }) => (
                  <section key={`${selectedMarker.id}-${hotspot.hotspotLabel}-${index}`} className="mapPopupHotspotCard">
                    {/* ส่วนหัวของการ์ด hotspot แต่ละตัว */}
                    <div className="mapPopupHotspotHeader">
                      <p className="mapPopupHotspotName">{hotspot.hotspotLabel || `Hotspot ${index + 1}`}</p>
                      <span className={getPopupPriorityClass(hotspot.priority)}>
                        {hotspot.priority ?? "Not rated"}
                      </span>
                    </div>

                    {/* ส่วนข้อมูลสรุปของ hotspot */}
                    <div className="mapPopupMetricGrid">
                      <section className="mapPopupMetricCard">
                        <p className="mapPopupMetricLabel">Equipment</p>
                        <p className="mapPopupMetricValue">{hotspot.equipmentLabel}</p>
                      </section>

                      <section className="mapPopupMetricCard">
                        <p className="mapPopupMetricLabel">Temperature</p>
                        <p className="mapPopupMetricValue">{hotspot.temperatureLabel}</p>
                      </section>

                      <section className="mapPopupMetricCard mapPopupMetricCardWide">
                        <p className="mapPopupMetricLabel">Action</p>
                        <p className="mapPopupMetricValue">{hotspot.actionRequired ?? "No action suggested"}</p>
                      </section>
                    </div>
                  </section>
                ))}
              </div>
            </div>
          </Popup>
        )}
      </MapContainer>
    </div>
  );
}
