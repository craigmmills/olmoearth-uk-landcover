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

export type ComparisonMode =
  | 'satellite-vs-classification'
  | '2021-vs-2023'
  | 'classification-vs-worldcover';

// --- Dashboard types (Issue #23) ---

/** Typed per-class metrics from metrics.json */
export interface PerClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

/** Typed metrics structure from metrics.json */
export interface Metrics {
  overall_accuracy: number;
  evaluation_year?: string;
  per_class: Record<string, PerClassMetrics>;
  weighted_avg: { precision: number; recall: number; f1: number };
  training_accuracy?: number;
  n_training_samples?: number;
}

/** Gemini evaluation per-class entry */
export interface EvalPerClass {
  class_name: string;
  score: number;
  notes: string;
}

/** Gemini evaluation error region */
export interface ErrorRegion {
  location: string;
  expected: string;
  predicted: string;
  severity: 'high' | 'medium' | 'low';
}

/** Gemini evaluation data (nested under evaluation key) */
export interface EvaluationData {
  overall_score: number;
  per_class: EvalPerClass[];
  error_regions: ErrorRegion[];
  spatial_quality: string;
  confidence: number;
  recommendations: string[];
}

/** Full evaluation JSON structure (evaluation_YYYY.json) */
export interface Evaluation {
  year: string;
  timestamp: string;
  model: string;
  evaluation: EvaluationData;
  summary: {
    overall_score: number;
    confidence: number;
    num_error_regions: number;
    num_recommendations: number;
  };
}

/** Hypothesis JSON structure */
export interface Hypothesis {
  hypothesis: string;
  component: string;
  parameter_changes: Record<string, unknown>;
  expected_impact: string;
  risk?: string;
  tier?: number;
  confidence?: number;
  reasoning?: string;
  source?: string;
}

/** Config diff entry: { a: old_val, b: new_val } */
export interface ConfigDiffEntry {
  a: unknown;
  b: unknown;
}

/** Metrics diff structure */
export interface MetricsDiff {
  overall_accuracy: { a: number; b: number; delta: number };
  per_class: Record<string, { f1: { a: number; b: number; delta: number } }>;
}

// --- SSE Event Types (Issue #24) ---

/** SSE new_iteration event data -- mirrors backend SSEEvent */
export interface SSENewIterationEvent {
  session_id: string;
  iteration: number;
  timestamp: string;
}

/** SSE session_complete event data -- mirrors backend SSESessionCompleteEvent */
export interface SSESessionCompleteEvent {
  session_id: string;
  end_time: string;
  stop_reason: string;
  best_iteration: number | null;
  final_score: number | null;
  n_iterations: number;
}

/** Loop status for the status indicator in ControlPanel */
export type LoopStatus =
  | { state: 'connecting' }
  | { state: 'running'; sessionId: string; currentIteration: number }
  | { state: 'complete'; sessionId: string; nIterations: number; bestScore: number | null; stopReason: string }
  | { state: 'idle' }
  | { state: 'error'; message: string };
