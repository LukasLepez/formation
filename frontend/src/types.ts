export type RunStatus = 'queued' | 'running' | 'success' | 'failed'
export type LayerName = 'all' | 'bronze' | 'silver' | 'gold'
export type GraphSourceLayer = 'bronze' | 'silver'

export type RunInfo = {
  run_id: string
  status: RunStatus
  layer: LayerName
  persist_db: boolean
  auto_start_docker: boolean
  log_level: string
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  rows?: number | null
  columns?: number | null
  error?: string | null
}

export type GraphRunInfo = {
  run_id: string
  status: RunStatus
  source_layer: GraphSourceLayer
  created_at: string
  graph_count?: number | null
  incident_rows?: number | null
  telemetry_rows?: number | null
  error?: string | null
}

export type TablePreview = {
  layer: string
  table: string
  columns: string[]
  rows: Record<string, unknown>[]
  total_rows: number
  limit: number
  offset: number
}

export type GoldCsvInfo = {
  run_name: string
  csv_path: string
  created_at?: string | null
  rows?: number | null
  columns?: number | null
  size_bytes: number
}

export type MaintenanceMlRunInfo = {
  run_id: string
  status: RunStatus
  label_column: string
  gold_run_name?: string | null
  random_forest_balanced: boolean
  created_at: string
  rows?: number | null
  features?: number | null
  best_model?: string | null
  error?: string | null
}

export type MaintenanceMlReport = {
  run_id: string
  status: string
  gold_run_name: string
  gold_csv_path: string
  label_column: string
  rows: number
  features: number
  class_balance: Record<string, number>
  scale_pos_weight: number
  random_forest_balanced: boolean
  mlflow_tracking_uri: string
  results: MaintenanceMlResult[]
  best_model: string
  conclusion: string
}

export type MaintenanceMlResult = {
  model: string
  pr_auc_validation: number
  roc_auc_validation: number
  pr_auc_test: number
  roc_auc_test: number
  threshold: number
  cv_pr_auc_mean: number
  cv_pr_auc_std: number
  cv_roc_auc_mean: number
  cv_roc_auc_std: number
  validation_confusion_matrix: ConfusionMatrix
  test_confusion_matrix: ConfusionMatrix
  model_path: string
}

export type ConfusionMatrix = {
  tn: number
  fp: number
  fn: number
  tp: number
}
