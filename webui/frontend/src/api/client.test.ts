import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildTileUrl, buildChangeMapTileUrl, buildWorldCoverTileUrl, checkHealth, listSessions } from './client';

describe('buildTileUrl', () => {
  it('constructs correct URL for 2021 tiles', () => {
    const url = buildTileUrl('session_20260404_093509', 1, '2021');
    expect(url).toBe(
      'http://localhost:8000/api/sessions/session_20260404_093509/iterations/1/tiles/2021/{z}/{x}/{y}.png',
    );
  });

  it('constructs correct URL for 2023 tiles', () => {
    const url = buildTileUrl('session_20260404_093509', 3, '2023');
    expect(url).toContain('/tiles/2023/{z}/{x}/{y}.png');
  });
});

describe('buildChangeMapTileUrl', () => {
  it('constructs correct titiler URL', () => {
    const url = buildChangeMapTileUrl('experiments/session_1/iteration_001/change_map_cog.tif');
    expect(url).toContain('/cog/tiles/{z}/{x}/{y}.png?url=file://');
    expect(url).toContain('change_map_cog.tif');
  });
});

describe('buildWorldCoverTileUrl', () => {
  it('constructs correct WorldCover tile URL', () => {
    const url = buildWorldCoverTileUrl();
    expect(url).toBe('http://localhost:8000/api/worldcover/tiles/{z}/{x}/{y}.png');
  });
});

describe('checkHealth', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('returns true when backend is healthy', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ status: 'ok' }), { status: 200 }),
    );
    expect(await checkHealth()).toBe(true);
  });

  it('returns false when backend is unreachable', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('ECONNREFUSED'));
    expect(await checkHealth()).toBe(false);
  });

  it('returns false when backend returns 500', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('Internal Server Error', { status: 500, statusText: 'Internal Server Error' }),
    );
    expect(await checkHealth()).toBe(false);
  });
});

describe('listSessions', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('parses session list from backend', async () => {
    const mockSessions = [
      {
        session_id: 'session_20260404_093509',
        start_time: '2026-04-04T09:35:09',
        end_time: null,
        stop_reason: null,
        best_iteration: 1,
        final_score: 0.85,
        n_iterations: 3,
      },
    ];
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(mockSessions), { status: 200 }),
    );
    const result = await listSessions();
    expect(result).toHaveLength(1);
    expect(result[0].session_id).toBe('session_20260404_093509');
  });
});
