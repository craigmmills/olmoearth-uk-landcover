import { API_BASE_URL } from '@/constants';
import type { SessionSummary, IterationSummary, IterationDetail } from '@/types';

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    throw new ApiError(res.status, `${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    await fetchJson<{ status: string }>('/api/health');
    return true;
  } catch {
    return false;
  }
}

export async function listSessions(): Promise<SessionSummary[]> {
  return fetchJson<SessionSummary[]>('/api/sessions');
}

export async function getSession(sessionId: string): Promise<SessionSummary> {
  return fetchJson<SessionSummary>(`/api/sessions/${sessionId}`);
}

export async function listIterations(
  sessionId: string,
  signal?: AbortSignal,
): Promise<IterationSummary[]> {
  return fetchJson<IterationSummary[]>(
    `/api/sessions/${sessionId}/iterations`,
    { signal },
  );
}

/**
 * Fetch full detail for a single iteration.
 * Used by IterationCard on expand.
 */
export async function getIterationDetail(
  sessionId: string,
  iterationNum: number,
  signal?: AbortSignal,
): Promise<IterationDetail> {
  return fetchJson<IterationDetail>(
    `/api/sessions/${sessionId}/iterations/${iterationNum}`,
    { signal },
  );
}

/**
 * Build the tile URL template for a year-based classification overlay.
 * Returns a URL with {z}/{x}/{y} placeholders for MapLibre.
 */
export function buildTileUrl(
  sessionId: string,
  iterationNum: number,
  year: '2021' | '2023',
): string {
  return `${API_BASE_URL}/api/sessions/${sessionId}/iterations/${iterationNum}/tiles/${year}/{z}/{x}/{y}.png`;
}

/**
 * Build the tile URL template for the change map overlay using the
 * direct titiler /cog/tiles/ endpoint. The COG path is an absolute
 * filesystem path on the server.
 */
export function buildChangeMapTileUrl(cogPath: string): string {
  return `${API_BASE_URL}/cog/tiles/{z}/{x}/{y}.png?url=file://${encodeURIComponent(cogPath)}`;
}

/**
 * Derive the expected COG path for the change map in a given
 * session/iteration directory. The backend's ensure_cog() creates
 * files with _cog suffix alongside the original.
 *
 * Path pattern: experiments/{session_id}/iteration_{NNN}/change_map_cog.tif
 */
export function deriveChangeMapCogPath(
  sessionId: string,
  iterationNum: number,
): string {
  const iterDir = `iteration_${String(iterationNum).padStart(3, '0')}`;
  return `experiments/${sessionId}/${iterDir}/change_map_cog.tif`;
}

/**
 * Build the tile URL template for WorldCover overlay tiles.
 * Returns a URL with {z}/{x}/{y} placeholders for MapLibre.
 */
export function buildWorldCoverTileUrl(): string {
  return `${API_BASE_URL}/api/worldcover/tiles/{z}/{x}/{y}.png`;
}
