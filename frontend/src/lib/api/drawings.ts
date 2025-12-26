/**
 * Chart Drawings API Client
 * 
 * Functions for saving/loading drawings to PostgreSQL database.
 * Uses apiClient for proper authentication handling and token refresh.
 */

import { apiClient } from './client';
import type { Drawing } from '~/types/drawing.types';

const DRAWINGS_PATH = '/chart-drawings';

export interface DrawingCreateDTO {
  symbol: string;
  timeframe: string;
  drawing_type: string;
  drawing_data: any;
  style: {
    color: string;
    lineWidth: number;
    lineType: string;
    fillColor?: string;
    fillOpacity?: number;
  };
  locked?: boolean;
  visible?: boolean;
}

export interface DrawingResponse {
  id: string;
  user_id: string;
  symbol: string;
  timeframe: string;
  drawing_type: string;
  drawing_data: any;
  style: any;
  locked: boolean;
  visible: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Transform backend response to frontend Drawing format
 */
function transformToDrawing(d: DrawingResponse): Drawing {
  return {
    id: d.id,
    type: d.drawing_type as any,
    symbol: d.symbol,
    timeframe: d.timeframe,
    userId: d.user_id,
    createdAt: d.created_at,
    updatedAt: d.updated_at,
    style: d.style,
    locked: d.locked,
    visible: d.visible,
    ...d.drawing_data, // Merge drawing-specific data (points, price, etc.)
  };
}

/**
 * Get all drawings for a symbol and timeframe
 */
export async function getDrawings(
  symbol: string,
  timeframe: string
): Promise<Drawing[]> {
  try {
    const data: DrawingResponse[] = await apiClient.get(
      `${DRAWINGS_PATH}?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}`
    );
    
    return data.map(transformToDrawing);
  } catch (error: any) {
    // Silently return empty for auth errors or not found
    if (error?.status === 401 || error?.status === 404) {
      console.warn('Drawings not available:', error?.message || 'Auth or not found');
      return [];
    }
    console.error('Failed to load drawings:', error);
    return [];
  }
}

/**
 * Create a new drawing
 */
export async function createDrawing(drawing: Partial<Drawing>): Promise<Drawing | null> {
  try {
    // Extract drawing-specific data
    const { id, userId, createdAt, updatedAt, type, symbol, timeframe, style, locked, visible, ...drawingData } = drawing as any;
    
    const payload: DrawingCreateDTO = {
      symbol: symbol!,
      timeframe: timeframe!,
      drawing_type: type!,
      drawing_data: drawingData,
      style: style || {
        color: '#3b82f6',
        lineWidth: 2,
        lineType: 'solid',
      },
      locked: locked || false,
      visible: visible !== false,
    };

    const data: DrawingResponse = await apiClient.post(DRAWINGS_PATH, payload);
    return transformToDrawing(data);
  } catch (error) {
    console.error('Failed to save drawing:', error);
    return null;
  }
}

/**
 * Update an existing drawing
 */
export async function updateDrawing(id: string, updates: Partial<Drawing>): Promise<Drawing | null> {
  try {
    const { type, symbol, timeframe, userId, createdAt, updatedAt, style, locked, visible, ...drawingData } = updates as any;
    
    const payload: any = {};
    if (Object.keys(drawingData).length > 0) {
      payload.drawing_data = drawingData;
    }
    if (style) payload.style = style;
    if (locked !== undefined) payload.locked = locked;
    if (visible !== undefined) payload.visible = visible;

    const data: DrawingResponse = await apiClient.patch(`${DRAWINGS_PATH}/${id}`, payload);
    return transformToDrawing(data);
  } catch (error) {
    console.error('Failed to update drawing:', error);
    return null;
  }
}

/**
 * Delete a single drawing
 */
export async function deleteDrawing(id: string): Promise<boolean> {
  try {
    await apiClient.delete(`${DRAWINGS_PATH}/${id}`);
    return true;
  } catch (error) {
    console.error('Failed to delete drawing:', error);
    return false;
  }
}

/**
 * Delete all drawings for a symbol/timeframe
 */
export async function deleteAllDrawings(
  symbol: string,
  timeframe?: string
): Promise<number> {
  try {
    const url = timeframe
      ? `${DRAWINGS_PATH}/symbol/${encodeURIComponent(symbol)}?timeframe=${encodeURIComponent(timeframe)}`
      : `${DRAWINGS_PATH}/symbol/${encodeURIComponent(symbol)}`;

    const data = await apiClient.delete(url);
    return data.deleted_count || 0;
  } catch (error) {
    console.error('Failed to delete all drawings:', error);
    return 0;
  }
}
