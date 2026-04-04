/** Mirrors backend Pydantic SessionSummary model */
export interface SessionSummary {
  session_id: string;
  start_time: string;   // ISO 8601
  end_time: string | null;
  stop_reason: string | null;
  best_iteration: number | null;
  final_score: number | null;
  n_iterations: number;
}

/** Mirrors backend Pydantic SessionDetail model */
export interface SessionDetail extends SessionSummary {
  initial_config: Record<string, unknown> | null;
  parameter_count: number | null;
}

/** Mirrors backend Pydantic IterationSummary model */
export interface IterationSummary {
  iteration: number;
  timestamp: string;    // ISO 8601
  status: string;
  overall_accuracy: number | null;
}

/** Mirrors backend Pydantic IterationDetail model */
export interface IterationDetail {
  metadata: Record<string, unknown>;
  metrics: Record<string, unknown>;
  config: Record<string, unknown>;
  hypothesis: Record<string, unknown> | null;
  evaluations: Record<string, Record<string, unknown> | null>;
  images: Record<string, string | null>;
  config_diff: Record<string, unknown> | null;
  metrics_diff: Record<string, unknown> | null;
}

/** Represents a toggleable overlay layer in the UI */
export interface LayerState {
  id: string;                       // e.g. "landcover-2021"
  label: string;                    // e.g. "Landcover 2021"
  visible: boolean;
  opacity: number;                  // 0.0 - 1.0
  tileUrlTemplate: string | null;   // null if not yet resolved
}

export type BasemapType = 'osm' | 'satellite';
