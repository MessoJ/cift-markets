/**
 * Drawing Tools Type Definitions
 * 
 * Types for chart drawings (trendlines, shapes, annotations).
 * All drawings persist to database.
 */

export type DrawingType =
  | 'trendline'
  | 'horizontal_line'
  | 'vertical_line'
  | 'fibonacci'
  | 'rectangle'
  | 'text'
  | 'arrow'
  | 'ruler';

export type DrawingStyle = {
  color: string;
  lineWidth: number;
  lineType: 'solid' | 'dashed' | 'dotted';
  fillColor?: string;
  fillOpacity?: number;
};

export interface DrawingPoint {
  timestamp: number; // X coordinate (time)
  price: number; // Y coordinate (price)
}

export interface BaseDrawing {
  id: string;
  type: DrawingType;
  symbol: string;
  timeframe: string;
  userId?: string;
  createdAt: string;
  updatedAt: string;
  style: DrawingStyle;
  locked: boolean;
  visible: boolean;
}

export interface TrendlineDrawing extends BaseDrawing {
  type: 'trendline';
  points: [DrawingPoint, DrawingPoint]; // Start and end
  extended: boolean; // Extend line infinitely
}

export interface HorizontalLineDrawing extends BaseDrawing {
  type: 'horizontal_line';
  price: number;
  label?: string;
}

export interface VerticalLineDrawing extends BaseDrawing {
  type: 'vertical_line';
  timestamp: number;
  label?: string;
}

export interface FibonacciDrawing extends BaseDrawing {
  type: 'fibonacci';
  points: [DrawingPoint, DrawingPoint]; // High and low points
  levels: number[]; // [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
  showLabels: boolean;
}

export interface RectangleDrawing extends BaseDrawing {
  type: 'rectangle';
  points: [DrawingPoint, DrawingPoint]; // Top-left and bottom-right
}

export interface TextDrawing extends BaseDrawing {
  type: 'text';
  point: DrawingPoint;
  text: string;
  fontSize: number;
  fontFamily: string;
  backgroundColor?: string;
}

export interface ArrowDrawing extends BaseDrawing {
  type: 'arrow';
  points: [DrawingPoint, DrawingPoint];
  arrowType: 'single' | 'double';
}

export interface RulerDrawing extends BaseDrawing {
  type: 'ruler';
  points: [DrawingPoint, DrawingPoint];
}

export type Drawing =
  | TrendlineDrawing
  | HorizontalLineDrawing
  | VerticalLineDrawing
  | FibonacciDrawing
  | RectangleDrawing
  | TextDrawing
  | ArrowDrawing
  | RulerDrawing;

export interface DrawingState {
  drawings: Drawing[];
  selectedDrawingId: string | null;
  activeTool: DrawingType | null;
  isDrawing: boolean;
}

// Default styles
export const DEFAULT_DRAWING_STYLES: Record<DrawingType, DrawingStyle> = {
  trendline: {
    color: '#3b82f6',
    lineWidth: 2,
    lineType: 'solid',
  },
  horizontal_line: {
    color: '#10b981',
    lineWidth: 1,
    lineType: 'dashed',
  },
  vertical_line: {
    color: '#f59e0b',
    lineWidth: 1,
    lineType: 'dashed',
  },
  fibonacci: {
    color: '#ec4899',
    lineWidth: 1,
    lineType: 'solid',
    fillColor: '#ec489920',
    fillOpacity: 0.1,
  },
  rectangle: {
    color: '#8b5cf6',
    lineWidth: 1,
    lineType: 'solid',
    fillColor: '#8b5cf620',
    fillOpacity: 0.1,
  },
  text: {
    color: '#ffffff',
    lineWidth: 1,
    lineType: 'solid',
    fillColor: '#00000080',
    fillOpacity: 0.5,
  },
  arrow: {
    color: '#f97316',
    lineWidth: 2,
    lineType: 'solid',
  },
  ruler: {
    color: '#22d3ee',
    lineWidth: 2,
    lineType: 'dashed',
  },
};
