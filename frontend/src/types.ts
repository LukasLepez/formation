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

export type VisionDatasetSplit = {
  name: string
  label: string
  count: number
  sample_paths: string[]
}

export type VisionDatasetPreparation = {
  version_id: string
  dataset_hash: string
  created_at: string
  manifest_path: string
  target_size: number
  validation_ratio: number
  random_seed: number
  split_counts: Record<string, number>
  class_counts: Record<string, number>
  channel_mean: number[]
  channel_std: number[]
  resize_strategy: string
  pixel_scaling: string
  leakage_free: boolean
  augmentation_scope: string
  vertical_flip: boolean
}

export type VisionDatasetInfo = {
  name: string
  root_path: string
  archive_path?: string | null
  archive_size_bytes?: number | null
  image_size: string
  total_images: number
  train_good: VisionDatasetSplit
  test_good: VisionDatasetSplit
  test_defects: VisionDatasetSplit[]
  ground_truth_masks: VisionDatasetSplit[]
  validation_hint: string
  preparation?: VisionDatasetPreparation | null
}

export type VisionModelRunInfo = {
  run_id: string
  status: RunStatus
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  dataset_version: string
  model_type?: 'autoencoder' | 'patchcore'
  epochs: number
  batch_size: number
  loss_name: 'mse' | 'ssim' | string
  threshold_percentile: number
  error?: string | null
  report_path?: string | null
}

export type VisionModelReport = {
  run_id: string
  status: string
  created_at: string
  dataset_version: string
  dataset_hash: string
  model_type?: 'autoencoder' | 'patchcore'
  config: {
    epochs: number
    batch_size: number
    learning_rate: number
    loss_name: string
    latent_filters: number
    threshold_percentile: number
  }
  architecture: {
    input_shape: number[]
    latent_shape: number[]
    input_values: number
    latent_values: number
    compression_ratio: number
    parameter_count: number
    comment: string
    summary_artifact: string
  }
  training: {
    train_normal_images: number
    validation_normal_images: number
    test_images: number
    epochs_completed: number
    best_epoch: number
    best_validation_loss: number
    augmentation: string
  }
  threshold: {
    method: string
    percentile: number
    value: number
    calibration_images: number
  }
  metrics: {
    image: VisionEvaluationMetrics
    pixel: { auroc?: number | null; average_precision?: number | null }
  }
  artifacts: Record<string, string>
  samples: Array<{
    path: string
    label: string
    score: number
    predicted_anomaly: boolean
    heatmap_artifact: string
  }>
  critical_analysis: string
  carbon?: { available?: boolean; duration_seconds?: number; energy_kwh?: number | null; emissions_gco2eq?: number | null; reason?: string }
  b7_artifacts?: Record<string, string>
  model_card_path?: string | null
  mlflow_tracking_uri: string
  mlflow_run_id?: string | null
}

export type VisionEvaluationMetrics = {
  auroc?: number | null
  average_precision?: number | null
  precision: number
  recall: number
  f1: number
  confusion_matrix: ConfusionMatrix
}

export type MaintenanceMlRunInfo = {
  run_id: string
  status: RunStatus
  label_column: string
  gold_run_name?: string | null
  random_forest_balanced: boolean
  selected_models?: string[]
  decision_tree_max_depth?: number
  decision_tree_min_samples_leaf?: number
  random_forest_n_estimators?: number
  random_forest_max_depth?: number
  random_forest_min_samples_leaf?: number
  random_forest_min_samples_split?: number
  random_forest_max_features?: string | null
  random_forest_bootstrap?: boolean
  xgboost_n_estimators?: number
  xgboost_max_depth?: number
  xgboost_learning_rate?: number
  xgboost_scale_pos_weight_auto?: boolean
  xgboost_scale_pos_weight?: number | null
  threshold_strategy?: ThresholdStrategy
  target_recall?: number
  false_negative_cost?: number
  false_positive_cost?: number
  experiment_hypothesis?: string
  random_state?: number
  tune?: boolean
  tune_n_trials?: number
  tune_timeout_seconds?: number
  tune_mode?: 'frugal' | 'heavy' | string
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
  pr_auc_random_baseline?: number | null
  scale_pos_weight: number
  xgboost_effective_scale_pos_weight?: number | null
  random_forest_balanced: boolean
  selected_models?: string[]
  decision_tree_max_depth?: number
  decision_tree_min_samples_leaf?: number
  random_forest_n_estimators?: number
  random_forest_max_depth?: number
  random_forest_min_samples_leaf?: number
  random_forest_min_samples_split?: number
  random_forest_max_features?: string | null
  random_forest_bootstrap?: boolean
  xgboost_n_estimators?: number
  xgboost_max_depth?: number
  xgboost_learning_rate?: number
  xgboost_scale_pos_weight_auto?: boolean
  xgboost_scale_pos_weight?: number | null
  threshold_strategy?: ThresholdStrategy
  target_recall?: number
  false_negative_cost?: number
  false_positive_cost?: number
  experiment_hypothesis?: string
  random_state?: number
  mlflow_tracking_uri: string
  tuning?: Record<string, unknown>
  carbon?: { available?: boolean; duration_seconds?: number; energy_kwh?: number | null; emissions_gco2eq?: number | null; reason?: string }
  b7_artifacts?: Record<string, string>
  reproducibility?: Record<string, string>
  event_log_path?: string | null
  model_card_path?: string | null
  results: MaintenanceMlResult[]
  best_model: string
  conclusion: string
}

export type MaintenanceMlResult = {
  model: string
  pr_auc_train?: number
  roc_auc_train?: number
  pr_auc_validation: number
  roc_auc_validation: number
  pr_auc_test: number
  roc_auc_test: number
  precision_validation?: number
  recall_validation?: number
  f1_validation?: number
  precision_test?: number
  recall_test?: number
  f1_test?: number
  threshold: number
  business_cost_test?: number
  training_seconds?: number
  inference_ms_per_row?: number
  model_size_bytes?: number
  test_metrics_at_05?: MaintenanceMetricSnapshot
  cv_pr_auc_mean: number
  cv_pr_auc_std: number
  cv_roc_auc_mean: number
  cv_roc_auc_std: number
  validation_confusion_matrix: ConfusionMatrix
  test_confusion_matrix: ConfusionMatrix
  top_features?: FeatureInsight[]
  shap_explanations?: ShapExplanations
  artifacts?: Record<string, string>
  mlflow_run_id?: string | null
  model_path: string
}

export type ShapExplanations = {
  available: boolean
  reason?: string
  base_value?: number
  top_features?: ShapFeature[]
  sample?: ShapLocalSample
}

export type ShapFeature = {
  feature: string
  mean_abs_shap: number
}

export type ShapLocalSample = {
  row_number: number
  score: number
  threshold: number
  is_alert: boolean
  factors: ShapFactor[]
}

export type ShapFactor = {
  feature: string
  value: number
  contribution: number
  direction: 'augmente' | 'réduit' | string
}

export type ConfusionMatrix = {
  tn: number
  fp: number
  fn: number
  tp: number
}

export type MaintenanceMetricSnapshot = {
  pr_auc: number
  roc_auc: number
  precision: number
  recall: number
  f1: number
  business_cost?: number
  confusion_matrix: ConfusionMatrix
}

export type FeatureInsight = {
  feature: string
  importance: number
  coefficient?: number
  direction?: 'augmente' | 'diminue'
  explanation?: string
}

export type ThresholdStrategy = 'balanced' | 'recall' | 'precision' | 'target_recall'

export type MaintenanceMlEvent = {
  ts: string
  step: string
  status: 'running' | 'success' | 'failed' | string
  message: string
  details?: Record<string, unknown>
}

export type MlflowTrackingSummary = {
  experiment_name: string
  tracking_uri: string
  runs: MlflowRunSummary[]
}

export type MlflowRunSummary = {
  run_id: string
  app_run_id?: string | null
  run_name?: string | null
  status?: string | null
  start_time?: string | null
  model?: string | null
  threshold?: number | null
  threshold_strategy?: string | null
  dataset_hash?: string | null
  validation_pr_auc?: number | null
  validation_recall?: number | null
  test_pr_auc?: number | null
  test_recall?: number | null
  test_f1?: number | null
  test_business_cost?: number | null
}

export type MlflowUiStatus = {
  running: boolean
  url: string
}

export type ModelPromotionResponse = {
  registered_model_name: string
  version: string
  stage: string
  mlflow_run_id: string
  model_uri: string
  readme_path?: string | null
}

export type ModelCandidateTest = {
  name: string
  passed: boolean
  detail: string
}

export type ModelCandidateTestReport = {
  registered_model_name: string
  stage: string
  model_uri: string
  passed: boolean
  tests: ModelCandidateTest[]
}
