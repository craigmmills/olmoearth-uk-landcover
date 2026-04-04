import { useState, useEffect, useCallback } from 'react';
import {
  listSessions,
  listIterations,
  buildTileUrl,
  buildChangeMapTileUrl,
  deriveChangeMapCogPath,
} from '@/api/client';
import type { SessionSummary, LayerState } from '@/types';
import { DEFAULT_LAYERS } from '@/constants';

interface SessionStateInternal {
  loading: boolean;
  error: string | null;
  sessionId: string | null;
  iterationNum: number | null;
  layers: LayerState[];
}

export interface SessionState extends SessionStateInternal {
  /** Call to update the displayed iteration (e.g., on SSE event) */
  refreshIteration: (sessionId: string, iterationNum: number) => void;
}

/**
 * On mount, fetches the latest session and its best iteration,
 * then resolves tile URL templates for each overlay layer.
 */
export function useLatestSession(): SessionState {
  const [state, setState] = useState<SessionStateInternal>({
    loading: true,
    error: null,
    sessionId: null,
    iterationNum: null,
    layers: DEFAULT_LAYERS.map((l) => ({
      id: l.id,
      label: l.label,
      visible: l.visible,
      opacity: l.opacity,
      tileUrlTemplate: null,
    })),
  });

  useEffect(() => {
    let cancelled = false;

    async function resolve() {
      try {
        const sessions = await listSessions();
        if (sessions.length === 0) {
          if (!cancelled) {
            setState((s) => ({
              ...s,
              loading: false,
              error: 'No sessions found. Run the pipeline first.',
            }));
          }
          return;
        }

        // Pick the most recent session by ISO 8601 start_time string comparison
        const latest = sessions.reduce((a: SessionSummary, b: SessionSummary) =>
          a.start_time > b.start_time ? a : b,
        );

        // Get best iteration, or fall back to latest by number
        let iterNum = latest.best_iteration;
        if (iterNum == null) {
          const iterations = await listIterations(latest.session_id);
          if (iterations.length === 0) {
            // Session exists but has no iterations -- cannot serve tiles
            if (!cancelled) {
              setState((s) => ({
                ...s,
                loading: false,
                error: 'Session has no iterations yet.',
                sessionId: latest.session_id,
                iterationNum: null,
              }));
            }
            return;
          }
          // Sort by iteration number to pick the last one
          const sorted = [...iterations].sort((a, b) => a.iteration - b.iteration);
          iterNum = sorted[sorted.length - 1].iteration;
        }

        // Build tile URLs for each layer
        const layers: LayerState[] = DEFAULT_LAYERS.map((def) => {
          let tileUrlTemplate: string | null = null;

          if (def.year === '2021' || def.year === '2023') {
            tileUrlTemplate = buildTileUrl(latest.session_id, iterNum!, def.year);
          } else if (def.year === 'change') {
            const cogPath = deriveChangeMapCogPath(latest.session_id, iterNum!);
            tileUrlTemplate = buildChangeMapTileUrl(cogPath);
          }

          return {
            id: def.id,
            label: def.label,
            visible: def.visible,
            opacity: def.opacity,
            tileUrlTemplate,
          };
        });

        if (!cancelled) {
          setState({
            loading: false,
            error: null,
            sessionId: latest.session_id,
            iterationNum: iterNum,
            layers,
          });
        }
      } catch (e) {
        if (!cancelled) {
          setState((s) => ({
            ...s,
            loading: false,
            error: e instanceof Error ? e.message : 'Failed to load session data',
          }));
        }
      }
    }

    resolve();
    return () => { cancelled = true; };
  }, []);

  const refreshIteration = useCallback(
    (newSessionId: string, newIterationNum: number) => {
      const layers: LayerState[] = DEFAULT_LAYERS.map((def) => {
        let tileUrlTemplate: string | null = null;
        if (def.year === '2021' || def.year === '2023') {
          tileUrlTemplate = buildTileUrl(newSessionId, newIterationNum, def.year);
        } else if (def.year === 'change') {
          const cogPath = deriveChangeMapCogPath(newSessionId, newIterationNum);
          tileUrlTemplate = buildChangeMapTileUrl(cogPath);
        }
        return {
          id: def.id,
          label: def.label,
          visible: def.visible,
          opacity: def.opacity,
          tileUrlTemplate,
        };
      });
      setState({
        loading: false,
        error: null,
        sessionId: newSessionId,
        iterationNum: newIterationNum,
        layers,
      });
    },
    [],
  );

  return { ...state, refreshIteration };
}
