// Minimal Leaflet type stubs to satisfy react-leaflet props in this project.
// If you add @types/leaflet later, you can remove this file.
declare module "leaflet" {
  export type LatLngTuple = [number, number] | [number, number, number];
  export type PointExpression = [number, number];
  export type LatLngExpression =
    | LatLngTuple
    | { lat: number; lng: number }
    | { lat: number; lon: number };
  export type LatLngBoundsExpression =
    | [LatLngExpression, LatLngExpression]
    | LatLngExpression[];

  export interface FitBoundsOptions {
    padding?: unknown;
    paddingTopLeft?: unknown;
    paddingBottomRight?: unknown;
    maxZoom?: number;
  }

  export interface MapOptions {
    center?: LatLngExpression;
    zoom?: number;
    minZoom?: number;
    maxZoom?: number;
    [key: string]: unknown;
  }

  export interface IconOptions {
    iconUrl?: string;
    iconRetinaUrl?: string;
    shadowUrl?: string;
    iconSize?: PointExpression;
    iconAnchor?: PointExpression;
    popupAnchor?: PointExpression;
    shadowSize?: PointExpression;
  }

  export class Icon<T extends IconOptions = IconOptions> {
    constructor(options?: T);
  }

  export interface MarkerOptions {
    icon?: Icon;
    opacity?: number;
    zIndexOffset?: number;
    draggable?: boolean;
    riseOnHover?: boolean;
    riseOffset?: number;
  }

  export class Map {}
}
