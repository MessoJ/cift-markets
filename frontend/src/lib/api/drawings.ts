/**
 * Chart Drawings API Client
 * 
 * Functions for saving/loading drawings to PostgreSQL database.
 */

import { apiClient } from './client';
import type { Drawing } from '~/types/drawing.types';

const BASE_URL = '/api/v1/chart-drawings';

// Get auth token from localStorage
function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem('access_token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

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
 * Get all drawings for a symbol and timeframe
 */
export async function getDrawings(
  symbol: string,
  timeframe: string
): Promise<Drawing[]> {
  try {
    const response = await fetch(
      `${BASE_URL}?symbol=${symbol}&timeframe=${timeframe}`,
      {
        headers: getAuthHeaders(),
      }
    );

    if (!response.ok) {
      if (response.status === 404 || response.status === 401) {
        return []; // No drawings found or not authenticated, return empty array
      }
      throw new Error(`Failed to load drawings: ${response.statusText}`);
    }

    const data: DrawingResponse[] = await response.json();
    
    // Transform backend format to frontend format
    return data.map(d => ({
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
    }));
  } catch (error) {
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
    const { id, userId, createdAt, updatedAt, ...drawingData } = drawing as any;
    
    const payload: DrawingCreateDTO = {
      symbol: drawing.symbol!,
      timeframe: drawing.timeframe!,
      drawing_type: drawing.type!,
      drawing_data: drawingData,
      style: drawing.style || {
        color: '#3b82f6',
        lineWidth: 2,
        lineType: 'solid',
      },
      locked: drawing.locked || false,
      visible: drawing.visible !== false,
    };

    const response = await fetch(BASE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Failed to save drawing: ${response.statusText}`);
    }

    const data: DrawingResponse = await response.json();
    
    // Transform to frontend format
    return {
      id: data.id,
      type: data.drawing_type as any,
      symbol: data.symbol,
      timeframe: data.timeframe,
      userId: data.user_id,
      createdAt: data.created_at,
      updatedAt: data.updated_at,
      style: data.style,
      locked: data.locked,
      visible: data.visible,
      ...data.drawing_data,
    };
  } catch (error) {
    console.error('Failed to save drawing:', error);
    return null;
  }
}

/**
 * Delete a single drawing
 */
export async function deleteDrawing(id: string): Promise<boolean> {
  try {
    const response = await fetch(`${BASE_URL}/${id}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to delete drawing: ${response.statusText}`);
    }

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
      ? `${BASE_URL}/symbol/${symbol}?timeframe=${timeframe}`
      : `${BASE_URL}/symbol/${symbol}`;

    const response = await fetch(url, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to delete drawings: ${response.statusText}`);
    }

    const data = await response.json();
    return data.deleted_count || 0;
  } catch (error) {
    console.error('Failed to delete all drawings:', error);
    return 0;
  }
}
