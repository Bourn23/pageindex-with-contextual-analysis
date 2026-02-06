
export enum GeminiModel {
  FLASH_3 = 'gemini-3-flash-preview',
  PRO_3 = 'gemini-3-pro-preview',
  FLASH_2_5 = 'gemini-2.5-flash-latest',
  PRO_2_5 = 'gemini-2.5-pro-latest'
}

export interface DataPoint {
  label: string;
  xValue: number | string;
  yValue: number | string;
}

export interface FigureExtractionResult {
  title: string;
  xAxis: {
    label: string;
    unit?: string;
  };
  yAxis: {
    label: string;
    unit?: string;
  };
  dataPoints: DataPoint[];
  summary: string;
}

export interface ComparisonData {
  label: string;
  extracted: number;
  groundTruth: number;
}

export interface GridConfig {
  enabled: boolean;
  rows: number;
  cols: number;
}

export interface BoundingBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
}

export type ActiveTab = 'parser' | 'cropper';
