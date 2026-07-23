import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Collapse,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  Switch,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import type { SelectChangeEvent } from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import ScienceIcon from '@mui/icons-material/Science'
import TerminalIcon from '@mui/icons-material/Terminal'
import { api, apiText, messageFrom } from '../lib/api'
import { formatNumber, formatPercent } from '../lib/format'
import { statusLabel, statusValueColor } from '../lib/status'
import type { GoldCsvInfo, MaintenanceMlEvent, MaintenanceMlReport, MaintenanceMlResult, MaintenanceMlRunInfo, MlflowTrackingSummary, MlflowUiStatus, ModelCandidateTestReport, ModelPromotionResponse, ThresholdStrategy } from '../types'
import { MetricCard } from '../components/MetricCard'
import { SectionHeader } from '../components/SectionHeader'
import { StatusChip } from '../components/StatusChip'
import { InfoTooltip } from '../components/InfoTooltip'

export function MaintenanceMlPage() {
  const [maintenanceRuns, setMaintenanceRuns] = useState<MaintenanceMlRunInfo[]>([])
  const [goldCsvs, setGoldCsvs] = useState<GoldCsvInfo[]>([])
  const [selectedMaintenanceRunId, setSelectedMaintenanceRunId] = useState('')
  const [mlGoldRun, setMlGoldRun] = useState('')
  const [mlLabel, setMlLabel] = useState('label_failure_next_24h')
  const [selectedModels, setSelectedModels] = useState<string[]>(['logistic_regression', 'decision_tree', 'random_forest', 'random_forest_balanced', 'xgboost'])
  const [decisionTreeMaxDepth, setDecisionTreeMaxDepth] = useState(6)
  const [decisionTreeMinSamplesLeaf, setDecisionTreeMinSamplesLeaf] = useState(10)
  const [randomForestEstimators, setRandomForestEstimators] = useState(60)
  const [randomForestMaxDepth, setRandomForestMaxDepth] = useState(12)
  const [randomForestMinSamplesLeaf, setRandomForestMinSamplesLeaf] = useState(2)
  const [randomForestMinSamplesSplit, setRandomForestMinSamplesSplit] = useState(10)
  const [randomForestMaxFeatures, setRandomForestMaxFeatures] = useState('sqrt')
  const [randomForestBootstrap, setRandomForestBootstrap] = useState(true)
  const [xgboostEstimators, setXgboostEstimators] = useState(100)
  const [xgboostMaxDepth, setXgboostMaxDepth] = useState(6)
  const [xgboostLearningRate, setXgboostLearningRate] = useState(0.1)
  const [xgboostScalePosWeightAuto, setXgboostScalePosWeightAuto] = useState(true)
  const [xgboostScalePosWeight, setXgboostScalePosWeight] = useState(10)
  const [thresholdStrategy, setThresholdStrategy] = useState<ThresholdStrategy>('balanced')
  const [targetRecall, setTargetRecall] = useState(0.8)
  const [falseNegativeCost, setFalseNegativeCost] = useState(20)
  const [falsePositiveCost, setFalsePositiveCost] = useState(1)
  const [experimentHypothesis, setExperimentHypothesis] = useState('')
  const [randomState, setRandomState] = useState(42)
  const [tune, setTune] = useState(false)
  const [tuneTrials, setTuneTrials] = useState(15)
  const [tuneMode, setTuneMode] = useState<'frugal' | 'heavy'>('frugal')
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [maintenanceLogs, setMaintenanceLogs] = useState('')
  const [maintenanceEvents, setMaintenanceEvents] = useState<MaintenanceMlEvent[]>([])
  const [maintenanceReport, setMaintenanceReport] = useState<MaintenanceMlReport | null>(null)
  const [mlflowTracking, setMlflowTracking] = useState<MlflowTrackingSummary | null>(null)
  const [mlflowUiStatus, setMlflowUiStatus] = useState<MlflowUiStatus | null>(null)
  const [promotionResult, setPromotionResult] = useState<ModelPromotionResponse | null>(null)
  const [candidateTestReport, setCandidateTestReport] = useState<ModelCandidateTestReport | null>(null)
  const [mlflowLoading, setMlflowLoading] = useState(false)
  const [comparisonRunIds, setComparisonRunIds] = useState<string[]>([])
  const [comparisonReports, setComparisonReports] = useState<Record<string, MaintenanceMlReport>>({})
  const [comparisonLoading, setComparisonLoading] = useState(false)
  const [submittedRunSettings, setSubmittedRunSettings] = useState<Record<string, Partial<MaintenanceMlRunInfo>>>({})
  const [activeTab, setActiveTab] = useState(0)
  const [reportTab, setReportTab] = useState(0)
  const [runToDelete, setRunToDelete] = useState<MaintenanceMlRunInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [launchingMaintenanceRun, setLaunchingMaintenanceRun] = useState(false)
  const [error, setError] = useState('')

  const displayedMaintenanceRuns = maintenanceRuns.map((run) => ({ ...(submittedRunSettings[run.run_id] ?? {}), ...run }))
  const latestMaintenanceRun = displayedMaintenanceRuns[0]
  const selectedMaintenanceRun = displayedMaintenanceRuns.find((run) => run.run_id === selectedMaintenanceRunId)
  const maintenanceRunIsActive = selectedMaintenanceRun?.status === 'queued' || selectedMaintenanceRun?.status === 'running'
  const maintenanceButtonLoading = launchingMaintenanceRun || maintenanceRunIsActive
  const canLaunchMaintenanceRun = Boolean(goldCsvs.length && selectedModels.length)
  const selectedComparisonReports = comparisonRunIds.map((runId) => comparisonReports[runId]).filter(Boolean)

  function toggleModel(model: string, checked: boolean) {
    setSelectedModels((current) => {
      if (checked) return current.includes(model) ? current : [...current, model]
      return current.filter((item) => item !== model)
    })
  }

  function toggleComparisonRun(runId: string, checked: boolean) {
    setComparisonRunIds((current) => {
      if (checked) return current.includes(runId) ? current : [...current, runId]
      return current.filter((item) => item !== runId)
    })
  }

  async function openReportFromMlflow(appRunId?: string | null) {
    if (!appRunId) return
    setSelectedMaintenanceRunId(appRunId)
    setActiveTab(1)
    setReportTab(0)
    await loadMaintenanceReport(appRunId)
  }

  async function refresh() {
    setError('')
    try {
      const [csvData, mlRunData] = await Promise.all([
        api<GoldCsvInfo[]>('/gold-csvs'),
        api<MaintenanceMlRunInfo[]>('/maintenance-ml-runs'),
      ])
      setGoldCsvs(csvData)
      setMaintenanceRuns(mlRunData)
      if (!mlGoldRun && csvData[0]) setMlGoldRun(csvData[0].run_name)
      if (!selectedMaintenanceRunId && mlRunData[0]) setSelectedMaintenanceRunId(mlRunData[0].run_id)
      setComparisonRunIds((current) => {
        if (current.length) return current.filter((runId) => mlRunData.some((run) => run.run_id === runId && run.status === 'success'))
        return current
      })
    } catch (refreshError) {
      setError(messageFrom(refreshError))
    }
  }

  async function launchMaintenanceRun() {
    setBusy(true)
    setLaunchingMaintenanceRun(true)
    setError('')
    setMaintenanceReport(null)
    try {
      const payload = {
        label_column: mlLabel,
        gold_run_name: mlGoldRun || null,
        random_forest_balanced: selectedModels.includes('random_forest_balanced'),
        selected_models: selectedModels,
        decision_tree_max_depth: decisionTreeMaxDepth,
        decision_tree_min_samples_leaf: decisionTreeMinSamplesLeaf,
        random_forest_n_estimators: randomForestEstimators,
        random_forest_max_depth: randomForestMaxDepth,
        random_forest_min_samples_leaf: randomForestMinSamplesLeaf,
        random_forest_min_samples_split: randomForestMinSamplesSplit,
        random_forest_max_features: randomForestMaxFeatures,
        random_forest_bootstrap: randomForestBootstrap,
        xgboost_n_estimators: xgboostEstimators,
        xgboost_max_depth: xgboostMaxDepth,
        xgboost_learning_rate: xgboostLearningRate,
        xgboost_scale_pos_weight_auto: xgboostScalePosWeightAuto,
        xgboost_scale_pos_weight: xgboostScalePosWeightAuto ? null : xgboostScalePosWeight,
        threshold_strategy: thresholdStrategy,
        target_recall: targetRecall,
        false_negative_cost: falseNegativeCost,
        false_positive_cost: falsePositiveCost,
        experiment_hypothesis: experimentHypothesis.trim(),
        random_state: randomState,
        tune,
        tune_n_trials: tuneTrials,
        tune_timeout_seconds: tuneMode === 'frugal' ? 600 : 1800,
        tune_mode: tuneMode,
      }
      const run = await api<MaintenanceMlRunInfo>('/maintenance-ml-runs', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      setSubmittedRunSettings((current) => ({ ...current, [run.run_id]: payload }))
      setSelectedMaintenanceRunId(run.run_id)
      setActiveTab(1)
      await refresh()
    } catch (mlError) {
      setError(messageFrom(mlError))
    } finally {
      setBusy(false)
      setLaunchingMaintenanceRun(false)
    }
  }

  async function loadMaintenanceReport(runId = selectedMaintenanceRunId) {
    if (!runId) return
    setError('')
    try {
      const report = await api<MaintenanceMlReport>(`/maintenance-ml-runs/${runId}/report`)
      setMaintenanceReport(report)
    } catch (reportError) {
      setMaintenanceReport(null)
      setError(messageFrom(reportError))
    }
  }

  async function loadMaintenanceEvents(runId = selectedMaintenanceRunId) {
    if (!runId) return
    try {
      const events = await api<MaintenanceMlEvent[]>(`/maintenance-ml-runs/${runId}/events`)
      setMaintenanceEvents(events)
    } catch {
      setMaintenanceEvents([])
    }
  }

  async function loadMlflowTracking() {
    setError('')
    setMlflowLoading(true)
    try {
      const [tracking, uiStatus] = await Promise.all([
        api<MlflowTrackingSummary>('/maintenance-mlflow/runs'),
        api<MlflowUiStatus>('/maintenance-mlflow/ui'),
      ])
      setMlflowTracking(tracking)
      setMlflowUiStatus(uiStatus)
    } catch (trackingError) {
      setError(messageFrom(trackingError))
    } finally {
      setMlflowLoading(false)
    }
  }

  async function startMlflowUi() {
    setError('')
    setMlflowLoading(true)
    try {
      setMlflowUiStatus(await api<MlflowUiStatus>('/maintenance-mlflow/ui/start', { method: 'POST' }))
    } catch (uiError) {
      setError(messageFrom(uiError))
    } finally {
      setMlflowLoading(false)
    }
  }

  async function promoteSelectedModel() {
    if (!maintenanceReport) return
    setError('')
    setMlflowLoading(true)
    try {
      const result = await api<ModelPromotionResponse>(`/maintenance-ml-runs/${maintenanceReport.run_id}/promote-staging`, { method: 'POST' })
      setPromotionResult(result)
      await loadMlflowTracking()
    } catch (promotionError) {
      setError(messageFrom(promotionError))
    } finally {
      setMlflowLoading(false)
    }
  }

  async function testModelCandidate() {
    setError('')
    setMlflowLoading(true)
    try {
      setCandidateTestReport(await api<ModelCandidateTestReport>('/maintenance-mlflow/model-candidate/test', { method: 'POST' }))
    } catch (testError) {
      setError(messageFrom(testError))
    } finally {
      setMlflowLoading(false)
    }
  }

  async function deleteMaintenanceRun() {
    if (!runToDelete) return
    setBusy(true)
    setError('')
    try {
      await api<{ status: string; run_id: string }>(`/maintenance-ml-runs/${runToDelete.run_id}`, { method: 'DELETE' })
      const remainingRuns = maintenanceRuns.filter((run) => run.run_id !== runToDelete.run_id)
      setMaintenanceRuns(remainingRuns)
      if (selectedMaintenanceRunId === runToDelete.run_id) {
        setSelectedMaintenanceRunId(remainingRuns[0]?.run_id ?? '')
        setMaintenanceLogs('')
        setMaintenanceEvents([])
      }
      setComparisonRunIds((current) => current.filter((runId) => runId !== runToDelete.run_id))
      setSubmittedRunSettings((current) => {
        const next = { ...current }
        delete next[runToDelete.run_id]
        return next
      })
      if (maintenanceReport?.run_id === runToDelete.run_id) setMaintenanceReport(null)
      setRunToDelete(null)
      await refresh()
    } catch (deleteError) {
      setError(messageFrom(deleteError))
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    void refresh()
    void loadMlflowTracking()
  }, [])

  useEffect(() => {
    if (reportTab > 2) setReportTab(0)
  }, [reportTab])

  useEffect(() => {
    if (!comparisonRunIds.length) return
    let active = true
    async function loadComparisonReports() {
      setComparisonLoading(true)
      try {
        const reports = await Promise.all(
          comparisonRunIds.map(async (runId) => [runId, await api<MaintenanceMlReport>(`/maintenance-ml-runs/${runId}/report`)] as const),
        )
        if (active) setComparisonReports(Object.fromEntries(reports))
      } catch (compareError) {
        if (active) setError(messageFrom(compareError))
      } finally {
        if (active) setComparisonLoading(false)
      }
    }
    void loadComparisonReports()
    return () => {
      active = false
    }
  }, [comparisonRunIds])

  useEffect(() => {
    if (!selectedMaintenanceRunId) return
    const selected = displayedMaintenanceRuns.find((run) => run.run_id === selectedMaintenanceRunId)
    if (selected?.status === 'success') {
      void loadMaintenanceReport(selectedMaintenanceRunId)
    } else if (selected) {
      setMaintenanceReport(null)
    }
  }, [selectedMaintenanceRunId, selectedMaintenanceRun?.status])

  useEffect(() => {
    if (!selectedMaintenanceRunId) return
    const selectedStatus = selectedMaintenanceRun?.status
    let active = true
    async function loadOnce() {
      try {
        const [text] = await Promise.all([
          apiText(`/maintenance-ml-runs/${selectedMaintenanceRunId}/logs/raw`),
          loadMaintenanceEvents(selectedMaintenanceRunId),
        ])
        if (active) setMaintenanceLogs(text)
      } catch {
        if (active) setMaintenanceLogs('')
        if (active) setMaintenanceEvents([])
      }
    }
    async function poll() {
      try {
        const [text] = await Promise.all([
          apiText(`/maintenance-ml-runs/${selectedMaintenanceRunId}/logs/raw`),
          loadMaintenanceEvents(selectedMaintenanceRunId),
        ])
        if (active) setMaintenanceLogs(text)
        const current = await api<MaintenanceMlRunInfo>(`/maintenance-ml-runs/${selectedMaintenanceRunId}`)
        if (active) {
          setMaintenanceRuns((runs) => runs.map((run) => (run.run_id === current.run_id ? current : run)))
        }
        if (current?.status === 'success' || current?.status === 'failed') {
          await refresh()
          if (current.status === 'success') await loadMaintenanceReport(selectedMaintenanceRunId)
        }
      } catch {
        if (active) setMaintenanceLogs('')
        if (active) setMaintenanceEvents([])
      }
    }
    if (selectedStatus !== 'queued' && selectedStatus !== 'running') {
      void loadOnce()
      return () => {
        active = false
      }
    }
    void poll()
    const interval = window.setInterval(() => void poll(), 3000)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [selectedMaintenanceRunId, selectedMaintenanceRun?.status])

  return (
    <Container maxWidth={false} className="pageShell maintenancePage" id="maintenance">
      <Stack spacing={2}>
        <Box className="topbar">
          <Box>
            <Typography variant="caption" color="text.secondary">InduSense</Typography>
            <Typography variant="h4">Maintenance ML</Typography>
            <Typography color="text.secondary">Classification binaire sur Gold dataset, split temporel, déséquilibre et suivi MLflow.</Typography>
          </Box>
        </Box>

        {(busy || maintenanceRunIsActive) && <LinearProgress />}
        {error && <Alert severity="error">{error}</Alert>}

        <Box className="summaryGrid">
          <MetricCard
            label="Dernier run ML"
            value={latestMaintenanceRun ? statusLabel[latestMaintenanceRun.status] : '-'}
            helper={latestMaintenanceRun?.run_id ?? 'Aucun run'}
            valueColor={latestMaintenanceRun ? statusValueColor[latestMaintenanceRun.status] : undefined}
            info="Indique l'état du dernier entraînement ML lancé : succès, échec, en cours ou en attente."
          />
          <MetricCard
            label="Meilleur modèle"
            value={latestMaintenanceRun?.best_model ? modelDisplayName(latestMaintenanceRun.best_model) : maintenanceReport?.best_model ? modelDisplayName(maintenanceReport.best_model) : '-'}
            helper="sélection par PR-AUC validation"
            info="Modèle retenu parmi la régression logistique, l'arbre de décision, le Random Forest et XGBoost. Il est choisi sur la PR-AUC de validation, adaptée aux pannes rares."
          />
          <MetricCard
            label="Lignes Gold"
            value={maintenanceReport?.rows.toLocaleString('fr-FR') ?? latestMaintenanceRun?.rows?.toLocaleString('fr-FR') ?? '-'}
            helper={`${maintenanceReport?.features ?? latestMaintenanceRun?.features ?? '-'} features`}
            info="Nombre de lignes utilisées depuis le Gold dataset et nombre de variables exploitées par les modèles après exclusion des colonnes de fuite."
          />
          <MetricCard
            label="Taux panne train"
            value={formatPercent(maintenanceReport?.class_balance.train_positive_rate)}
            helper="accuracy volontairement ignorée"
            info="Part de lignes positives dans le train. Si ce taux est faible, les pannes sont rares : l'accuracy devient trompeuse et PR-AUC est plus pertinente."
          />
        </Box>

        <Paper className="tabsPanel">
          <Tabs
            value={activeTab}
            onChange={(_, value) => setActiveTab(value)}
            variant="scrollable"
            scrollButtons="auto"
            aria-label="Sections Maintenance ML"
          >
            <Tab label="Entraînement" />
            <Tab label="Rapports" />
            <Tab label="Historique" />
            <Tab label="Comparatif" />
            <Tab label="MLflow" />
          </Tabs>
        </Paper>

        {activeTab === 0 && (
          <Stack spacing={2}>
            <Paper className="panel">
              <SectionHeader
                title="Réglages d'entraînement"
                icon={<ScienceIcon fontSize="small" />}
                action={
                  <Button
                    variant="contained"
                    startIcon={<PlayArrowIcon />}
                    onClick={() => void launchMaintenanceRun()}
                    loading={maintenanceButtonLoading}
                    disabled={!canLaunchMaintenanceRun}
                  >
                    Entraîner
                  </Button>
                }
              />
              <Stack spacing={2.5}>
                <Box className="settingsSection">
                  <Typography variant="subtitle2">Données utilisées</Typography>
                  <Box className="settingsGrid twoColumns">
                    <FormControl size="small">
                      <InputLabel>Gold dataset source</InputLabel>
                      <Select value={mlGoldRun} label="Gold dataset source" onChange={(event: SelectChangeEvent) => setMlGoldRun(event.target.value)}>
                        {goldCsvs.map((csv) => <MenuItem key={csv.run_name} value={csv.run_name}>{csv.run_name}</MenuItem>)}
                      </Select>
                    </FormControl>
                    <FormControl size="small">
                      <InputLabel>Cible à prédire</InputLabel>
                      <Select value={mlLabel} label="Cible à prédire" onChange={(event: SelectChangeEvent) => setMlLabel(event.target.value)}>
                        <MenuItem value="label_failure_next_6h">Panne dans les 6 prochaines heures</MenuItem>
                        <MenuItem value="label_failure_next_12h">Panne dans les 12 prochaines heures</MenuItem>
                        <MenuItem value="label_failure_next_24h">Panne dans les 24 prochaines heures</MenuItem>
                        <MenuItem value="label_failure_next_48h">Panne dans les 48 prochaines heures</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                </Box>

                <Box className="settingsSection">
                  <Box className="settingsSectionHeader">
                    <Box>
                      <Typography variant="subtitle2">Modèles à entraîner</Typography>
                      <Typography variant="body2" color="text.secondary">Chaque modèle sélectionné sera entraîné puis comparé dans le rapport.</Typography>
                    </Box>
                  </Box>
                  <Box className="modelChoiceGrid">
                    <ModelSwitch label="Régression logistique" checked={selectedModels.includes('logistic_regression')} onChange={(checked) => toggleModel('logistic_regression', checked)} info="Modèle simple qui transforme les variables en probabilité de panne. Il sert de baseline interprétable." />
                    <ModelSwitch label="Arbre de décision" checked={selectedModels.includes('decision_tree')} onChange={(checked) => toggleModel('decision_tree', checked)} info="Modèle lisible sous forme de règles. Il aide à comprendre les séparations, mais peut surapprendre si on le laisse trop profond." />
                    <ModelSwitch label="Random Forest" checked={selectedModels.includes('random_forest')} onChange={(checked) => toggleModel('random_forest', checked)} info="Version standard sans compensation explicite des pannes rares. Utile comme baseline robuste." />
                    <ModelSwitch label="RF rééquilibré" checked={selectedModels.includes('random_forest_balanced')} onChange={(checked) => toggleModel('random_forest_balanced', checked)} info="Même Random Forest, mais avec class_weight='balanced' pour donner plus de poids aux pannes rares." />
                    <ModelSwitch label="XGBoost" checked={selectedModels.includes('xgboost')} onChange={(checked) => toggleModel('xgboost', checked)} info="Modèle de boosting souvent performant. Il est ignoré automatiquement si la dépendance XGBoost n'est pas disponible." />
                  </Box>
                  {!selectedModels.length && <Alert severity="warning">Sélectionne au moins un modèle pour lancer l'entraînement.</Alert>}
                </Box>

                <Box className="settingsSection">
                  <Button
                    className="advancedToggle"
                    variant="text"
                    endIcon={<ExpandMoreIcon className={advancedOpen ? 'advancedToggleIconOpen' : 'advancedToggleIcon'} />}
                    onClick={() => setAdvancedOpen((open) => !open)}
                    aria-expanded={advancedOpen}
                  >
                    Réglages avancés
                  </Button>
                  <Collapse in={advancedOpen} timeout="auto" unmountOnExit>
                    <Box className="trainingAdvancedGrid">
                      <Box className="thresholdGrid">
                        <Box className="thresholdHeader">
                          <Box>
                            <Typography variant="body2" sx={{ fontWeight: 700 }}>Objectif du seuil d'alerte</Typography>
                            <Typography variant="caption" color="text.secondary">Choisis le comportement attendu des alertes.</Typography>
                          </Box>
                          <InfoTooltip title="Le seuil convertit la probabilité de panne en décision : alerte ou pas d'alerte." />
                        </Box>
                        <FormControl size="small">
                          <InputLabel>Stratégie de seuil</InputLabel>
                          <Select value={thresholdStrategy} label="Stratégie de seuil" onChange={(event: SelectChangeEvent) => setThresholdStrategy(event.target.value as ThresholdStrategy)}>
                            <MenuItem value="balanced">Équilibré F1</MenuItem>
                            <MenuItem value="recall">Détecter plus de pannes</MenuItem>
                            <MenuItem value="precision">Limiter les fausses alertes</MenuItem>
                            <MenuItem value="target_recall">Atteindre un recall cible</MenuItem>
                          </Select>
                        </FormControl>
                        {thresholdStrategy === 'target_recall' && (
                          <TextField size="small" type="number" label="Objectif recall" value={targetRecall} onChange={(event) => setTargetRecall(clampFloat(event.target.value, 0.05, 1))} helperText="Ex. 0,80 = détecter au moins 80 % des pannes si possible." />
                        )}
                        <ThresholdExplanation strategy={thresholdStrategy} targetRecall={targetRecall} falseNegativeCost={falseNegativeCost} falsePositiveCost={falsePositiveCost} />
                        <Typography variant="caption" color="text.secondary">Coûts métier</Typography>
                        <Box className="costSettingsGrid">
                          <TextField size="small" type="number" label="Panne manquée" value={falseNegativeCost} onChange={(event) => setFalseNegativeCost(clampFloat(event.target.value, 1, 100000))} />
                          <TextField size="small" type="number" label="Fausse alerte" value={falsePositiveCost} onChange={(event) => setFalsePositiveCost(clampFloat(event.target.value, 1, 100000))} />
                        </Box>
                      </Box>
                      <Box className="advancedSettingsGrid">
                        <Box className="advancedSettingsGroup">
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>Arbre de décision</Typography>
                          <TextField size="small" type="number" label="Profondeur maximale" value={decisionTreeMaxDepth} onChange={(event) => setDecisionTreeMaxDepth(clampInteger(event.target.value, 1, 30))} helperText="Nombre maximal de questions successives." />
                          <TextField size="small" type="number" label="Lignes minimum par feuille" value={decisionTreeMinSamplesLeaf} onChange={(event) => setDecisionTreeMinSamplesLeaf(clampInteger(event.target.value, 1, 500))} helperText="Évite les règles trop spécifiques." />
                        </Box>
                        <Box className="advancedSettingsGroup">
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>Random Forest</Typography>
                          <TextField size="small" type="number" label="Nombre d'arbres" value={randomForestEstimators} onChange={(event) => setRandomForestEstimators(clampInteger(event.target.value, 10, 500))} helperText="Plus stable, mais plus lent." />
                          <TextField size="small" type="number" label="Profondeur maximale" value={randomForestMaxDepth} onChange={(event) => setRandomForestMaxDepth(clampInteger(event.target.value, 1, 40))} helperText="Complexité de chaque arbre." />
                          <TextField size="small" type="number" label="Lignes minimum par feuille" value={randomForestMinSamplesLeaf} onChange={(event) => setRandomForestMinSamplesLeaf(clampInteger(event.target.value, 1, 500))} helperText="Limite le surapprentissage." />
                          <TextField size="small" type="number" label="Lignes minimum pour split" value={randomForestMinSamplesSplit} onChange={(event) => setRandomForestMinSamplesSplit(clampInteger(event.target.value, 2, 1000))} helperText="Minimum requis pour couper un noeud." />
                          <FormControl size="small">
                            <InputLabel>Variables par split</InputLabel>
                            <Select value={randomForestMaxFeatures} label="Variables par split" onChange={(event: SelectChangeEvent) => setRandomForestMaxFeatures(event.target.value)}>
                              <MenuItem value="sqrt">Racine du nombre de variables</MenuItem>
                              <MenuItem value="log2">Log2 du nombre de variables</MenuItem>
                              <MenuItem value="all">Toutes les variables</MenuItem>
                            </Select>
                          </FormControl>
                          <Box className="switchField">
                            <Typography variant="caption" className="switchFieldLabel">Échantillonnage Bootstrap</Typography>
                            <FormControlLabel
                              control={<Switch checked={randomForestBootstrap} onChange={(_, checked) => setRandomForestBootstrap(checked)} />}
                              label={randomForestBootstrap ? 'Activé' : 'Désactivé'}
                            />
                            <Typography variant="caption" color="text.secondary">Chaque arbre apprend sur un tirage avec remise.</Typography>
                          </Box>
                        </Box>
                        <Box className="advancedSettingsGroup">
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>XGBoost</Typography>
                          <TextField size="small" type="number" label="Nombre d'arbres" value={xgboostEstimators} onChange={(event) => setXgboostEstimators(clampInteger(event.target.value, 10, 500))} helperText="Nombre d'étapes de boosting." />
                          <TextField size="small" type="number" label="Profondeur maximale" value={xgboostMaxDepth} onChange={(event) => setXgboostMaxDepth(clampInteger(event.target.value, 1, 20))} helperText="Complexité de chaque arbre XGBoost." />
                          <TextField size="small" type="number" label="Learning rate" value={xgboostLearningRate} onChange={(event) => setXgboostLearningRate(clampFloat(event.target.value, 0.001, 1))} helperText="Plus bas = apprentissage plus prudent." />
                          <FormControlLabel
                            control={<Switch checked={xgboostScalePosWeightAuto} onChange={(_, checked) => setXgboostScalePosWeightAuto(checked)} />}
                            label={
                              <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center' }}>
                                <span>Poids pannes auto</span>
                                <InfoTooltip title="Auto : scale_pos_weight = nombre de lignes OK / nombre de pannes dans le train. Manuel : tu forces le poids." />
                              </Stack>
                            }
                          />
                          <TextField size="small" type="number" label="Poids XGBoost manuel" value={xgboostScalePosWeight} disabled={xgboostScalePosWeightAuto} onChange={(event) => setXgboostScalePosWeight(clampFloat(event.target.value, 1, 10000))} helperText="Utilisé seulement si le mode auto est désactivé." />
                        </Box>
                        <Box className="advancedSettingsGroup">
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>Traçabilité du run</Typography>
                          <FormControlLabel control={<Switch checked={tune} onChange={(event) => setTune(event.target.checked)} />} label="Optimiser XGBoost avec Optuna" />
                          {tune && <><FormControl size="small"><InputLabel>Étude</InputLabel><Select label="Étude" value={tuneMode} onChange={(event) => setTuneMode(event.target.value as 'frugal' | 'heavy')}><MenuItem value="frugal">Frugale · pruning · budget réduit</MenuItem><MenuItem value="heavy">Large · sans pruning</MenuItem></Select></FormControl><TextField size="small" type="number" label="Essais Optuna" value={tuneTrials} onChange={(event) => setTuneTrials(clampInteger(event.target.value, 1, 100))} helperText="PR-AUC moyenne en CV temporelle." /></>}
                          <TextField size="small" type="number" label="Seed" value={randomState} onChange={(event) => setRandomState(clampInteger(event.target.value, 0, 999999))} helperText="Reproductibilité du run." />
                          <TextField
                            size="small"
                            label="Note du run"
                            value={experimentHypothesis}
                            onChange={(event) => setExperimentHypothesis(event.target.value)}
                            helperText="Optionnel : sert à retrouver pourquoi ce test a été lancé."
                            placeholder="Ex. Comparer XGBoost avec un seuil plus strict."
                            multiline
                            minRows={2}
                          />
                        </Box>
                      </Box>
                    </Box>
                  </Collapse>
                </Box>

              </Stack>
            </Paper>
            <Paper className="panel">
              <SectionHeader title="Suivi du run ML" icon={<TerminalIcon fontSize="small" />} />
              <TrainingTimeline events={maintenanceEvents} />
              <Typography variant="caption" color="text.secondary">
                Logs techniques : {maintenanceLogs.split('\n').filter(Boolean).length.toLocaleString('fr-FR')} lignes
              </Typography>
              <Box component="pre" className="logBox">
                {maintenanceLogs || 'Aucun log pour ce run.'}
              </Box>
            </Paper>
          </Stack>
        )}

        {activeTab === 1 && (
          <Stack spacing={2}>
            <Paper className="panel">
              <SectionHeader
                title="Rapports ML"
                icon={<ScienceIcon fontSize="small" />}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Choisis un run, lis son efficacité en synthèse, puis ouvre les détails seulement si nécessaire.
              </Typography>
              <ReportRunPicker
                runs={displayedMaintenanceRuns}
                selected={selectedMaintenanceRunId}
                onSelect={setSelectedMaintenanceRunId}
                onDelete={setRunToDelete}
              />
            </Paper>
            <Paper className="tabsPanel reportTabsPanel">
              <Tabs
                value={reportTab}
                onChange={(_, value) => setReportTab(value)}
                variant="scrollable"
                scrollButtons="auto"
                aria-label="Vues du rapport ML"
              >
                <Tab label="Synthèse" />
                <Tab label="Détails modèles" />
                <Tab label="Artefacts & explications" />
              </Tabs>
            </Paper>
            {maintenanceReport ? (
              <>
                {reportTab === 0 && (
                  <>
                    <Paper className="panel">
                      <RunEffectivenessSummary report={maintenanceReport} />
                    </Paper>
                    <B7StudySummary report={maintenanceReport} />
                    <Paper className="panel">
                      <SectionHeader title="Matrice de confusion" />
                      <ConfusionSummary result={bestResult(maintenanceReport)} />
                    </Paper>
                    <Alert severity="success">{maintenanceReport.conclusion}</Alert>
                  </>
                )}
                {reportTab === 1 && (
                <Paper className="panel">
                    <SectionHeader title="Comparatif modèles" icon={<ScienceIcon fontSize="small" />} />
                    <MaintenanceComparison report={maintenanceReport} />
                  </Paper>
                )}
                {reportTab === 2 && (
                  <Box className="reportDetailGrid">
                    <Box className="panelLite">
                      <SectionHeader title="Variables importantes" />
                      <FeatureInsightPanel result={bestResult(maintenanceReport)} />
                    </Box>
                    <Box className="panelLite">
                      <SectionHeader title="Explicabilité SHAP" />
                      <ShapExplanationPanel report={maintenanceReport} result={bestResult(maintenanceReport)} />
                    </Box>
                  </Box>
                )}
              </>
            ) : (
              <Alert severity="info">Aucun rapport chargé pour le moment. Lance un entraînement ou choisis un run réussi.</Alert>
            )}
          </Stack>
        )}

        {activeTab === 2 && (
          <Paper className="panel">
            <SectionHeader title="Historique global des runs" icon={<ScienceIcon fontSize="small" />} />
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Tous les entraînements lancés depuis l'application. Clique sur une ligne pour ouvrir son rapport.
            </Typography>
            <MaintenanceRunTable
              runs={displayedMaintenanceRuns}
              selected={selectedMaintenanceRunId}
              onSelect={(runId) => {
                setSelectedMaintenanceRunId(runId)
                setActiveTab(1)
                setReportTab(0)
              }}
              onDelete={setRunToDelete}
            />
          </Paper>
        )}

        {activeTab === 3 && (
          <Stack spacing={2}>
            <Paper className="panel">
              <SectionHeader title="Comparatif des entraînements" icon={<ScienceIcon fontSize="small" />} />
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Sélectionne librement autant de runs réussis que tu veux. Ce comparatif est indépendant du rapport ouvert.
              </Typography>
              <ComparisonRunSelector
                runs={displayedMaintenanceRuns}
                selectedRunIds={comparisonRunIds}
                onToggle={toggleComparisonRun}
                onSelectAll={(ids) => setComparisonRunIds(ids)}
                onClear={() => setComparisonRunIds([])}
                onOpenReport={(runId) => {
                  setSelectedMaintenanceRunId(runId)
                  setActiveTab(1)
                  setReportTab(0)
                }}
              />
            </Paper>
            <Paper className="panel">
              {comparisonLoading ? (
                <LinearProgress sx={{ mt: 1 }} />
              ) : (
                <MultiRunComparison reports={selectedComparisonReports} selectedCount={comparisonRunIds.length} />
              )}
            </Paper>
          </Stack>
        )}

        {activeTab === 4 && (
          <Stack spacing={2}>
            <Paper className="panel">
              <SectionHeader
                title="Tracking MLflow"
                icon={<ScienceIcon fontSize="small" />}
                action={
                  <Button variant="outlined" onClick={() => void loadMlflowTracking()} disabled={mlflowLoading}>
                    Actualiser
                  </Button>
                }
              />
              {mlflowLoading && <LinearProgress sx={{ mb: 1.5 }} />}
              <MlflowTrackingPanel
                tracking={mlflowTracking}
                uiStatus={mlflowUiStatus}
                report={maintenanceReport}
                promotionResult={promotionResult}
                candidateTestReport={candidateTestReport}
                onStartUi={() => void startMlflowUi()}
                onPromote={() => void promoteSelectedModel()}
                onTestCandidate={() => void testModelCandidate()}
                onOpenReport={(runId) => void openReportFromMlflow(runId)}
              />
            </Paper>
          </Stack>
        )}
      </Stack>
      <Dialog open={Boolean(runToDelete)} onClose={() => setRunToDelete(null)}>
        <DialogTitle>Supprimer ce rapport ML ?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Cette action supprimera le rapport, les métriques et les logs associés à ce run. Elle ne supprime pas le Gold dataset utilisé.
          </DialogContentText>
          {runToDelete && (
            <Typography sx={{ mt: 2, fontFamily: 'monospace', wordBreak: 'break-all' }}>
              {runToDelete.run_id}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRunToDelete(null)}>Annuler</Button>
          <Button color="error" variant="contained" onClick={() => void deleteMaintenanceRun()} disabled={busy}>
            Supprimer
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  )
}

function ReportRunPicker({
  runs,
  selected,
  onSelect,
  onDelete,
}: {
  runs: MaintenanceMlRunInfo[]
  selected: string
  onSelect: (id: string) => void
  onDelete: (run: MaintenanceMlRunInfo) => void
}) {
  const selectedRun = runs.find((run) => run.run_id === selected)
  const successfulRuns = runs.filter((run) => run.status === 'success')
  return (
    <Stack spacing={1.5}>
      <Box className="reportPickerGrid">
        <FormControl size="small">
          <InputLabel>Run à analyser</InputLabel>
          <Select value={selected} label="Run à analyser" onChange={(event: SelectChangeEvent) => onSelect(event.target.value)}>
            {runs.map((run) => (
              <MenuItem key={run.run_id} value={run.run_id}>
                {shortRunId(run.run_id)} · {statusLabel[run.status]} · {run.best_model ? modelDisplayName(run.best_model) : 'en attente'}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Box className="reportPickerActions">
          {selectedRun && (
            <Tooltip title={selectedRun.status === 'queued' || selectedRun.status === 'running' ? 'Suppression possible une fois le run terminé.' : 'Supprimer ce rapport'}>
              <span>
                <IconButton
                  color="error"
                  size="small"
                  disabled={selectedRun.status === 'queued' || selectedRun.status === 'running'}
                  aria-label={`Supprimer ${selectedRun.run_id}`}
                  onClick={() => onDelete(selectedRun)}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          )}
        </Box>
      </Box>
      <Box className="reportPickerMeta">
        <DefinitionItem label="Run sélectionné" detail={selectedRun ? shortRunId(selectedRun.run_id) : 'Aucun'} />
        <DefinitionItem label="Statut" detail={selectedRun ? statusLabel[selectedRun.status] : '-'} />
        <DefinitionItem label="Meilleur modèle" detail={selectedRun?.best_model ? modelDisplayName(selectedRun.best_model) : '-'} />
        <DefinitionItem label="Runs comparables" detail={`${successfulRuns.length} réussis dans l'onglet Comparatif`} />
      </Box>
    </Stack>
  )
}

function ComparisonRunSelector({
  runs,
  selectedRunIds,
  onToggle,
  onSelectAll,
  onClear,
  onOpenReport,
}: {
  runs: MaintenanceMlRunInfo[]
  selectedRunIds: string[]
  onToggle: (id: string, checked: boolean) => void
  onSelectAll: (ids: string[]) => void
  onClear: () => void
  onOpenReport: (id: string) => void
}) {
  const comparableRuns = runs.filter((run) => run.status === 'success')
  const comparableIds = comparableRuns.map((run) => run.run_id)
  if (!comparableRuns.length) {
    return <Alert severity="info">Aucun run réussi disponible pour le comparatif.</Alert>
  }

  return (
    <Stack spacing={1.5}>
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={1}
        sx={{ alignItems: { xs: 'stretch', sm: 'center' }, justifyContent: 'space-between' }}
      >
        <Typography variant="caption" color="text.secondary">
          {selectedRunIds.length} run{selectedRunIds.length > 1 ? 's' : ''} sélectionné{selectedRunIds.length > 1 ? 's' : ''} sur {comparableRuns.length}
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button size="small" variant="outlined" onClick={() => onSelectAll(comparableIds)}>
            Tout sélectionner
          </Button>
          <Button size="small" variant="text" onClick={onClear} disabled={!selectedRunIds.length}>
            Vider
          </Button>
        </Stack>
      </Stack>
      <TableContainer className="tableBlock comparisonTable">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Comparer</TableCell>
            <TableCell>Run</TableCell>
            <TableCell>Gold</TableCell>
            <TableCell>Cible</TableCell>
            <TableCell>Meilleur modèle</TableCell>
            <TableCell>Seuil</TableCell>
            <TableCell align="right">Rapport</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {comparableRuns.map((run) => (
            <TableRow key={run.run_id} hover selected={selectedRunIds.includes(run.run_id)}>
              <TableCell>
                <Checkbox
                  size="small"
                  checked={selectedRunIds.includes(run.run_id)}
                  slotProps={{ input: { 'aria-label': `Comparer ${run.run_id}` } }}
                  onChange={(_, checked) => onToggle(run.run_id, checked)}
                />
              </TableCell>
              <TableCell>{shortRunId(run.run_id)}</TableCell>
              <TableCell>{run.gold_run_name ?? '-'}</TableCell>
              <TableCell>{targetWindowLabel(run.label_column)}</TableCell>
              <TableCell>{run.best_model ? modelDisplayName(run.best_model) : '-'}</TableCell>
              <TableCell>{thresholdStrategyLabel(run.threshold_strategy)}</TableCell>
              <TableCell align="right">
                <Button size="small" variant="outlined" onClick={() => onOpenReport(run.run_id)}>
                  Ouvrir
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      </TableContainer>
    </Stack>
  )
}

function MaintenanceRunTable({
  runs,
  selected,
  onSelect,
  onDelete,
}: {
  runs: MaintenanceMlRunInfo[]
  selected: string
  onSelect: (id: string) => void
  onDelete: (run: MaintenanceMlRunInfo) => void
}) {
  return (
    <TableContainer sx={{ maxHeight: 430 }}>
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Run</TableCell>
            <TableCell>Statut</TableCell>
            <TableCell>Gold</TableCell>
            <TableCell>Cible</TableCell>
            <TableCell>Modèles</TableCell>
            <TableCell>Seuil</TableCell>
            <TableCell>Arbre</TableCell>
            <TableCell>Random Forest</TableCell>
            <TableCell>Meilleur</TableCell>
            <TableCell align="right">Action</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {runs.map((run) => {
            const isRunning = run.status === 'queued' || run.status === 'running'
            return (
              <TableRow key={run.run_id} hover selected={run.run_id === selected} onClick={() => onSelect(run.run_id)} sx={{ cursor: 'pointer' }}>
                <TableCell>{run.run_id}</TableCell>
                <TableCell><StatusChip status={run.status} /></TableCell>
                <TableCell>{run.gold_run_name ?? '-'}</TableCell>
                <TableCell>{run.label_column}</TableCell>
                <TableCell>
                  <RunSettingText value={formatRunModels(run)} detail={formatRunModelsDetail(run)} />
                </TableCell>
                <TableCell>
                  <RunSettingText value={thresholdStrategyLabel(run.threshold_strategy)} detail={thresholdStrategyDetail(run.threshold_strategy)} />
                </TableCell>
                <TableCell>
                  <RunSettingText value={formatTreeSettings(run)} detail="Profondeur maximale et lignes minimum par feuille pour l'arbre de décision." />
                </TableCell>
                <TableCell>
                  <RunSettingText value={formatForestSettings(run)} detail="Nombre d'arbres, profondeur maximale, lignes minimum par feuille et rééquilibrage de la Random Forest." />
                </TableCell>
                <TableCell>{run.best_model ? modelDisplayName(run.best_model) : '-'}</TableCell>
                <TableCell align="right">
                  <Tooltip title={isRunning ? 'Suppression possible une fois le run terminé.' : 'Supprimer ce rapport'}>
                    <span>
                      <IconButton
                        color="error"
                        size="small"
                        disabled={isRunning}
                        aria-label={`Supprimer ${run.run_id}`}
                        onClick={(event) => {
                          event.stopPropagation()
                          onDelete(run)
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

function MultiRunComparison({ reports, selectedCount }: { reports: MaintenanceMlReport[]; selectedCount: number }) {
  if (selectedCount < 2) {
    return <Alert severity="info">Coche au moins deux runs réussis pour comparer leurs métriques métier.</Alert>
  }
  if (reports.length < 2) {
    return <Alert severity="info">Les rapports sélectionnés sont en cours de chargement ou indisponibles.</Alert>
  }

  const rows = reports.map((report) => {
    const best = bestResult(report)
    const matrix = best.test_confusion_matrix
    const businessCost = best.business_cost_test ?? (matrix.fn * (report.false_negative_cost ?? 20) + matrix.fp * (report.false_positive_cost ?? 1))
    return { report, best, matrix, businessCost }
  })
  const maxRecall = Math.max(...rows.map((row) => row.best.recall_test ?? 0), 0.01)
  const maxF1 = Math.max(...rows.map((row) => row.best.f1_test ?? 0), 0.01)
  const maxPrecision = Math.max(...rows.map((row) => row.best.precision_test ?? 0), 0.01)
  const maxErrors = Math.max(...rows.map((row) => row.matrix.fn + row.matrix.fp), 1)
  const maxCost = Math.max(...rows.map((row) => row.businessCost), 1)
  const bestRecall = rows.reduce((best, row) => ((row.best.recall_test ?? 0) > (best.best.recall_test ?? 0) ? row : best), rows[0])
  const lowestCost = rows.reduce((best, row) => (row.businessCost < best.businessCost ? row : best), rows[0])

  return (
    <Stack spacing={2}>
      <Box className="runCompareSummary">
        <MetricCard
          label="Meilleur rappel"
          value={formatPercent(bestRecall.best.recall_test)}
          helper={`${shortRunId(bestRecall.report.run_id)} · ${modelDisplayName(bestRecall.best.model)}`}
          info="Le rappel est prioritaire dans le POC InduSense : il mesure les pannes réellement détectées."
        />
        <MetricCard
          label="Coût relatif min."
          value={lowestCost.businessCost.toLocaleString('fr-FR')}
          helper={`${shortRunId(lowestCost.report.run_id)} · FN x${formatNumber(lowestCost.report.false_negative_cost ?? 20)} + FP x${formatNumber(lowestCost.report.false_positive_cost ?? 1)}`}
          info="Score relatif pour arbitrer avec les coûts configurés pour le run : FN x coût panne manquée + FP x coût fausse alerte."
        />
        <MetricCard
          label="Runs comparés"
          value={reports.length.toLocaleString('fr-FR')}
          helper="basé sur le meilleur modèle de chaque run"
          info="Chaque run est résumé par son meilleur modèle selon la PR-AUC validation."
        />
      </Box>

      <Box className="runCompareHelp">
        <Typography variant="body2" sx={{ fontWeight: 750 }}>Lecture des métriques</Typography>
        <Box className="runCompareHelpGrid">
          <DefinitionItem label="Recall" detail="TP / (TP + FN). Parmi les vraies pannes, part détectée par le modèle. Prioritaire ici car un FN est une panne manquée." />
          <DefinitionItem label="Precision" detail="TP / (TP + FP). Parmi les alertes déclenchées, part qui correspond vraiment à une panne." />
          <DefinitionItem label="F1" detail="2 x Precision x Recall / (Precision + Recall). Compromis entre détecter les pannes et limiter les fausses alertes." />
          <DefinitionItem label="FN" detail="Faux négatifs : pannes réelles non détectées. C'est le risque métier principal en maintenance prédictive." />
          <DefinitionItem label="FP" detail="Faux positifs : alertes déclenchées alors qu'aucune panne n'arrive. Coût de vérification ou intervention inutile." />
          <DefinitionItem label="Coût" detail="Score relatif de comparaison : FN x coût panne manquée + FP x coût fausse alerte. Il illustre qu'une panne manquée pèse beaucoup plus qu'une fausse alarme." />
        </Box>
      </Box>

      <Box className="runCompareGrid">
        {rows.map((row) => (
          <Box className="runCompareCard" key={row.report.run_id}>
            <Stack spacing={0.75}>
              <Typography variant="body2" sx={{ fontWeight: 750 }}>{shortRunId(row.report.run_id)}</Typography>
              <RunConfigSummary report={row.report} result={row.best} />
            </Stack>
            <Box className="metricBars">
              <MetricBar label="Recall" value={row.best.recall_test ?? 0} max={maxRecall} tone="good" />
              <MetricBar label="F1" value={row.best.f1_test ?? 0} max={maxF1} tone="neutral" />
              <MetricBar label="Precision" value={row.best.precision_test ?? 0} max={maxPrecision} tone="neutral" />
            </Box>
            <Box className="errorBars">
              <ErrorBar label="FN" value={row.matrix.fn} max={maxErrors} tone="danger" />
              <ErrorBar label="FP" value={row.matrix.fp} max={maxErrors} tone="warning" />
              <ErrorBar label="Coût" value={row.businessCost} max={maxCost} tone="danger" />
            </Box>
          </Box>
        ))}
      </Box>
    </Stack>
  )
}

function DefinitionItem({ label, detail }: { label: string; detail: string }) {
  return (
    <Box className="definitionItem">
      <Typography variant="caption" sx={{ fontWeight: 750 }}>{label}</Typography>
      <Typography variant="caption" color="text.secondary">{detail}</Typography>
    </Box>
  )
}

function TrainingTimeline({ events }: { events: MaintenanceMlEvent[] }) {
  if (!events.length) {
    return (
      <Alert severity="info" sx={{ mb: 1.5 }}>
        Les nouveaux runs afficheront ici une timeline structurée : dataset, split, entraînement, seuils, MLflow et artefacts.
      </Alert>
    )
  }

  return (
    <Box className="trainingTimeline">
      {events.map((event, index) => (
        <Box key={`${event.ts}-${event.step}-${index}`} className={`timelineItem timelineItem-${event.status}`}>
          <Box className="timelineDot" />
          <Box className="timelineBody">
            <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline', justifyContent: 'space-between', gap: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 750 }}>{eventLabel(event.step)}</Typography>
              <Typography variant="caption" color="text.secondary">{event.status}</Typography>
            </Stack>
            <Typography variant="body2">{event.message}</Typography>
            <EventDetails details={event.details ?? {}} />
          </Box>
        </Box>
      ))}
    </Box>
  )
}

function EventDetails({ details }: { details: Record<string, unknown> }) {
  const entries = Object.entries(details).filter(([, value]) => value !== undefined && value !== null && value !== '')
  if (!entries.length) return null
  return (
    <Box className="eventDetails">
      {entries.slice(0, 5).map(([key, value]) => (
        <Typography key={key} variant="caption" color="text.secondary">
          {eventDetailLabel(key)} {formatEventValue(value)}
        </Typography>
      ))}
    </Box>
  )
}

function MlflowTrackingPanel({
  tracking,
  uiStatus,
  report,
  promotionResult,
  candidateTestReport,
  onStartUi,
  onPromote,
  onTestCandidate,
  onOpenReport,
}: {
  tracking: MlflowTrackingSummary | null
  uiStatus: MlflowUiStatus | null
  report: MaintenanceMlReport | null
  promotionResult: ModelPromotionResponse | null
  candidateTestReport: ModelCandidateTestReport | null
  onStartUi: () => void
  onPromote: () => void
  onTestCandidate: () => void
  onOpenReport: (runId?: string | null) => void
}) {
  const runs = tracking?.runs ?? []
  const bestRecall = bestMlflowRun(runs, 'test_recall')
  const bestF1 = bestMlflowRun(runs, 'test_f1')
  const bestCost = bestMlflowRun(runs, 'test_business_cost', false)
  const latestRun = latestMlflowRun(runs)
  return (
    <Stack spacing={2}>
      <Box className="modelMetaGrid">
        <ModelMetaItem label="Experiment" value={tracking?.experiment_name ?? 'maintenance_predictive_b5'} helper="Groupe logique MLflow qui rassemble les entraînements de maintenance prédictive." />
        <ModelMetaItem label="Tracking URI" value={tracking?.tracking_uri ?? '-'} helper="Base locale utilisée par MLflow pour stocker runs, métriques et artefacts." />
        <ModelMetaItem label="MLflow UI" value={uiStatus?.running ? 'Démarrée' : 'Arrêtée'} helper={uiStatus?.url ?? 'Interface web MLflow locale.'} />
        <ModelMetaItem label="Runs MLflow" value={`${runs.length}`} helper="Nombre de runs MLflow retrouvés via mlflow.search_runs()." />
        <ModelMetaItem label="Dernier run" value={latestRun?.start_time ? formatDateTime(latestRun.start_time) : '-'} helper="Date et heure de démarrage du run MLflow le plus récent." />
        <ModelMetaItem label="Meilleur coût" value={bestCost ? `${modelDisplayName(bestCost.model ?? '')} · ${formatNumber(bestCost.test_business_cost)}` : '-'} helper="Run avec le coût métier test le plus bas parmi les runs MLflow." />
      </Box>

      <Box className="mlflowActionBar">
        <Button variant="outlined" onClick={onStartUi}>{uiStatus?.running ? 'MLflow UI démarrée' : 'Lancer MLflow UI'}</Button>
        <Button variant="outlined" href={uiStatus?.url ?? 'http://127.0.0.1:5000'} target="_blank" rel="noreferrer" disabled={!uiStatus?.running}>Ouvrir MLflow UI</Button>
        <Button variant="contained" onClick={onPromote} disabled={!report}>Promouvoir meilleur modèle en Staging</Button>
        <Button variant="outlined" onClick={onTestCandidate}>Tester modèle Staging</Button>
      </Box>

      <Box className="runCompareSummary">
        <MetricCard label="Meilleur recall" value={bestRecall ? formatPercent(bestRecall.test_recall) : '-'} helper={bestRecall?.model ? modelDisplayName(bestRecall.model) : 'Aucun run'} info="Run MLflow qui retrouve la plus grande part de pannes sur le test." />
        <MetricCard label="Meilleur F1" value={bestF1 ? formatNumber(bestF1.test_f1) : '-'} helper={bestF1?.model ? modelDisplayName(bestF1.model) : 'Aucun run'} info="Run MLflow avec le meilleur compromis précision / recall." />
        <MetricCard label="Coût min" value={bestCost ? formatNumber(bestCost.test_business_cost) : '-'} helper={bestCost?.model ? modelDisplayName(bestCost.model) : 'Aucun run'} info="Run MLflow qui minimise FN x coût panne manquée + FP x coût fausse alerte." />
      </Box>

      <MlflowChecklist report={report} />
      <ModelRegistryPanel promotion={promotionResult} candidateTestReport={candidateTestReport} />
      <ReproducibilityPanel report={report} />
      <MlflowRunsTable runs={runs} report={report} onOpenReport={onOpenReport} />
    </Stack>
  )
}

function MlflowChecklist({ report }: { report: MaintenanceMlReport | null }) {
  const best = report ? bestResult(report) : undefined
  const items = [
    { label: 'Params', ok: Boolean(report), detail: 'Hyperparamètres, seuil, coûts métier, seed et cible.' },
    { label: 'Metrics', ok: Boolean(best), detail: 'PR-AUC, ROC-AUC, précision, recall, F1 et coût métier.' },
    { label: 'Artifacts', ok: Boolean(best?.artifacts && Object.keys(best.artifacts).length), detail: 'Courbes, matrices, importances et modèle sérialisé.' },
    { label: 'Tags', ok: Boolean(report?.reproducibility?.dataset_hash), detail: 'Hash dataset et versions Python / scikit-learn.' },
    { label: 'Model', ok: Boolean(best?.mlflow_run_id || best?.model_path), detail: 'Pipeline sklearn loggé dans MLflow et pickle local.' },
  ]
  return (
    <Box className="trackingChecklist">
      {items.map((item) => (
        <Box key={item.label} className={`trackingChecklistItem ${item.ok ? 'trackingChecklistItem-ok' : ''}`}>
          <Typography variant="body2" sx={{ fontWeight: 750 }}>{item.label}</Typography>
          <Typography variant="caption" color="text.secondary">{item.detail}</Typography>
        </Box>
      ))}
    </Box>
  )
}

function ModelRegistryPanel({
  promotion,
  candidateTestReport,
}: {
  promotion: ModelPromotionResponse | null
  candidateTestReport: ModelCandidateTestReport | null
}) {
  return (
    <Box className="runCompareHelp">
      <Typography variant="body2" sx={{ fontWeight: 750 }}>Model Registry</Typography>
      <Box className="runCompareHelpGrid">
        <DefinitionItem label="Modèle enregistré" detail={promotion?.registered_model_name ?? 'InduSense_PanneDetection'} />
        <DefinitionItem label="Stage" detail={promotion?.stage ?? candidateTestReport?.stage ?? 'Aucun modèle Staging testé'} />
        <DefinitionItem label="Version" detail={promotion?.version ?? '-'} />
        <DefinitionItem label="Model URI" detail={promotion?.model_uri ?? candidateTestReport?.model_uri ?? 'models:/InduSense_PanneDetection/Staging'} />
        <DefinitionItem label="Run MLflow" detail={promotion?.mlflow_run_id ? shortMlflowRunId(promotion.mlflow_run_id) : '-'} />
        <DefinitionItem label="Fiche modèle" detail={promotion?.readme_path ?? '-'} />
      </Box>
      {candidateTestReport && (
        <Box className="candidateTests">
          {candidateTestReport.tests.map((test) => (
            <Box key={test.name} className={`candidateTestItem ${test.passed ? 'candidateTestItem-ok' : 'candidateTestItem-failed'}`}>
              <Typography variant="caption" sx={{ fontWeight: 750 }}>{test.name}</Typography>
              <Typography variant="caption" color="text.secondary">{test.detail}</Typography>
            </Box>
          ))}
        </Box>
      )}
    </Box>
  )
}

function ReproducibilityPanel({ report }: { report: MaintenanceMlReport | null }) {
  if (!report) {
    return <Alert severity="info">Sélectionne un rapport réussi pour afficher la reproductibilité du run applicatif.</Alert>
  }
  const reproducibility = report.reproducibility ?? {}
  return (
    <Box className="runCompareHelp">
      <Typography variant="body2" sx={{ fontWeight: 750 }}>Reproductibilité</Typography>
      <Box className="runCompareHelpGrid">
        <DefinitionItem label="Dataset hash" detail={reproducibility.dataset_hash ?? '-'} />
        <DefinitionItem label="Seed" detail={`${report.random_state ?? 42}`} />
        <DefinitionItem label="CSV source" detail={report.gold_run_name} />
        <DefinitionItem label="Python" detail={reproducibility.python_version ?? '-'} />
        <DefinitionItem label="scikit-learn" detail={reproducibility.sklearn_version ?? '-'} />
        <DefinitionItem label="Journal structuré" detail={report.event_log_path ?? 'training_events.jsonl'} />
      </Box>
    </Box>
  )
}

function MlflowRunsTable({
  runs,
  report,
  onOpenReport,
}: {
  runs: MlflowTrackingSummary['runs']
  report: MaintenanceMlReport | null
  onOpenReport: (runId?: string | null) => void
}) {
  if (!runs.length) {
    return <Alert severity="info">Aucun run MLflow trouvé pour le moment. Lance un nouvel entraînement pour alimenter le tracking.</Alert>
  }
  const reportRunIds = new Set((report?.results ?? []).map((result) => result.mlflow_run_id).filter(Boolean))
  const sortedRuns = sortMlflowRunsByDateDesc(runs)
  return (
    <TableContainer className="tableBlock comparisonTable">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Run MLflow</TableCell>
            <TableCell>Date</TableCell>
            <TableCell>Modèle</TableCell>
            <TableCell>Statut</TableCell>
            <TableCell>Seuil</TableCell>
            <TableCell>PR-AUC val.</TableCell>
            <TableCell>Recall test</TableCell>
            <TableCell>F1 test</TableCell>
            <TableCell>Coût</TableCell>
            <TableCell>Dataset hash</TableCell>
            <TableCell align="right">Rapport</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sortedRuns.map((run) => (
            <TableRow key={run.run_id} selected={reportRunIds.has(run.run_id)}>
              <TableCell>
                <RunSettingText value={shortMlflowRunId(run.run_id)} detail={run.run_id} />
              </TableCell>
              <TableCell>{run.start_time ? formatDateTime(run.start_time) : '-'}</TableCell>
              <TableCell>{run.model ? modelDisplayName(run.model) : '-'}</TableCell>
              <TableCell>{run.status ?? '-'}</TableCell>
              <TableCell>{formatNumber(run.threshold)}</TableCell>
              <TableCell>{formatNumber(run.validation_pr_auc)}</TableCell>
              <TableCell>{formatPercent(run.test_recall)}</TableCell>
              <TableCell>{formatNumber(run.test_f1)}</TableCell>
              <TableCell>{formatNumber(run.test_business_cost)}</TableCell>
              <TableCell>{run.dataset_hash ?? '-'}</TableCell>
              <TableCell align="right">
                <Button
                  size="small"
                  variant="outlined"
                  disabled={!run.app_run_id}
                  onClick={() => onOpenReport(run.app_run_id)}
                >
                  Rapport
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

function RunConfigSummary({ report, result }: { report: MaintenanceMlReport; result: MaintenanceMlResult }) {
  return (
    <Box className="runConfigSummary">
      <ConfigHint label={targetWindowLabel(report.label_column)} detail={targetWindowDetail(report.label_column)} />
      <ConfigHint label={modelDisplayName(result.model)} detail={modelDetail(result.model)} />
      <ConfigHint label={thresholdStrategyLabel(report.threshold_strategy)} detail={thresholdStrategyDetail(report.threshold_strategy)} />
      <ConfigHint label={`seuil ${formatNumber(result.threshold)}`} detail="Seuil calculé sur validation pour ce modèle : score supérieur ou égal à cette valeur signifie alerte." />
      <ConfigHint label={`recall cible ${formatPercent(report.target_recall ?? 0.8)}`} detail="Objectif de recall défini avant le run : part minimale de pannes à détecter si le seuil le permet." />
      <ConfigHint label={`coût FN ${formatNumber(report.false_negative_cost ?? 20)} / FP ${formatNumber(report.false_positive_cost ?? 1)}`} detail="Poids métier utilisés pour comparer les modèles : une panne manquée coûte généralement beaucoup plus qu'une fausse alerte." />
      <ConfigHint label={`${report.random_forest_n_estimators ?? 60} arbres`} detail="Nombre d'arbres entraînés dans le Random Forest. Plus il y en a, plus le résultat est stable, mais plus le run est lent." />
      <ConfigHint label={`profondeur ${report.random_forest_max_depth ?? 12}`} detail="Profondeur maximale de chaque arbre Random Forest. Une profondeur plus grande capte plus de détails, avec plus de risque de surapprentissage." />
      <ConfigHint label={`split ${report.random_forest_min_samples_split ?? 10} · ${forestMaxFeaturesLabel(report.random_forest_max_features)}`} detail="Contraintes de découpe Random Forest : minimum de lignes pour créer une nouvelle branche et nombre de variables candidates à chaque split." />
      <ConfigHint label={report.random_forest_bootstrap ?? true ? 'bootstrap activé' : 'bootstrap désactivé'} detail="Avec bootstrap, chaque arbre apprend sur un échantillon tiré avec remise, ce qui augmente la diversité de la forêt." />
      <ConfigHint label={forestBalanceLabel(report.selected_models?.includes('random_forest_balanced') ?? report.random_forest_balanced)} detail={forestBalanceDetail(report.selected_models?.includes('random_forest_balanced') ?? report.random_forest_balanced)} />
      {report.selected_models?.includes('xgboost') && (
        <ConfigHint label={`XGB ${report.xgboost_n_estimators ?? 100} arbres · lr ${formatNumber(report.xgboost_learning_rate ?? 0.1)}`} detail="Paramètres XGBoost : nombre d'arbres de boosting et vitesse d'apprentissage." />
      )}
    </Box>
  )
}

function ConfigHint({ label, detail }: { label: string; detail: string }) {
  return (
    <Box className="runConfigItem">
      <Typography variant="caption">{label}</Typography>
      <InfoTooltip title={detail} />
    </Box>
  )
}

function MetricBar({ label, value, max, tone }: { label: string; value: number; max: number; tone: 'good' | 'neutral' }) {
  const width = `${Math.max(4, Math.min(100, (value / max) * 100))}%`
  return (
    <Box className="compareBarRow">
      <Typography variant="caption">{label}</Typography>
      <Box className="compareBarTrack">
        <Box className={`compareBarFill compareBarFill-${tone}`} sx={{ width }} />
      </Box>
      <Typography variant="caption" className="compareBarValue">{formatPercent(value)}</Typography>
    </Box>
  )
}

function ErrorBar({ label, value, max, tone }: { label: string; value: number; max: number; tone: 'danger' | 'warning' }) {
  const width = `${Math.max(4, Math.min(100, (value / max) * 100))}%`
  return (
    <Box className="compareBarRow">
      <Typography variant="caption">{label}</Typography>
      <Box className="compareBarTrack">
        <Box className={`compareBarFill compareBarFill-${tone}`} sx={{ width }} />
      </Box>
      <Typography variant="caption" className="compareBarValue">{value.toLocaleString('fr-FR')}</Typography>
    </Box>
  )
}

function RunSettingText({ value, detail }: { value: string; detail: string }) {
  return (
    <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center', minWidth: 0 }}>
      <Typography variant="body2" className="runSettingText">{value}</Typography>
      <InfoTooltip title={detail} />
    </Stack>
  )
}

function ModelSwitch({
  label,
  checked,
  onChange,
  info,
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  info: string
}) {
  return (
    <Box className="modelChoiceItem">
      <Switch checked={checked} onChange={(_, nextChecked) => onChange(nextChecked)} />
      <Typography variant="body2" component="span">{label}</Typography>
      <InfoTooltip title={info} />
    </Box>
  )
}

function ThresholdExplanation({
  strategy,
  targetRecall,
  falseNegativeCost,
  falsePositiveCost,
}: {
  strategy: ThresholdStrategy
  targetRecall: number
  falseNegativeCost: number
  falsePositiveCost: number
}) {
  const details: Record<ThresholdStrategy, { title: string; body: string; tradeoff: string; scoring: string }> = {
    balanced: {
      title: 'Équilibré F1',
      body: 'Cherche un compromis entre détecter les vraies pannes et éviter les fausses alertes.',
      tradeoff: 'Bon choix par défaut pour comparer les modèles sans favoriser un côté.',
      scoring: 'Optimise le F1 sur validation : précision et recall ont le même poids.',
    },
    recall: {
      title: 'Détecter plus de pannes',
      body: 'Baisse plus facilement le seuil pour rattraper davantage de pannes réelles.',
      tradeoff: 'Tu manqueras moins de pannes, mais tu risques plus de fausses alertes.',
      scoring: 'Optimise le F2 sur validation : le recall compte plus que la précision.',
    },
    precision: {
      title: 'Limiter les fausses alertes',
      body: 'Monte plutôt le seuil pour ne déclencher une alerte que quand le modèle est plus sûr.',
      tradeoff: 'Les alertes seront plus fiables, mais certaines pannes peuvent être manquées.',
      scoring: 'Optimise le F0.5 sur validation : la précision compte plus que le recall.',
    },
    target_recall: {
      title: 'Atteindre un recall cible',
      body: `Cherche un seuil qui respecte l'objectif de recall défini : ${formatPercent(targetRecall)}.`,
      tradeoff: "Très adapté au POC : l'objectif est fixé avant le run, puis on vérifie le coût en fausses alertes.",
      scoring: `Cherche d'abord recall >= ${formatPercent(targetRecall)}, puis garde le seuil avec la meilleure précision possible.`,
    },
  }
  const current = details[strategy]
  return (
    <Box className="thresholdExplanation">
      <Typography variant="caption" color="text.secondary">Impact</Typography>
      <Typography variant="body2" sx={{ fontWeight: 650 }}>{current.title}</Typography>
      <Typography variant="body2">{current.body}</Typography>
      <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center', minWidth: 0 }}>
        <Typography variant="caption" color="text.secondary" className="thresholdMethodText">
          Méthode : {current.scoring}
        </Typography>
        <InfoTooltip title={`${current.tradeoff} Coût affiché : FN x${formatNumber(falseNegativeCost)} + FP x${formatNumber(falsePositiveCost)}.`} />
      </Stack>
    </Box>
  )
}

function RunEffectivenessSummary({ report }: { report: MaintenanceMlReport }) {
  const best = bestResult(report)
  const matrix = best.test_confusion_matrix
  const recall = best.recall_test ?? 0
  const precision = best.precision_test ?? 0
  const f1 = best.f1_test ?? 0
  const businessCost = best.business_cost_test ?? (matrix.fn * (report.false_negative_cost ?? 20) + matrix.fp * (report.false_positive_cost ?? 1))
  const targetRecall = report.target_recall ?? 0.8
  const baseline = report.pr_auc_random_baseline ?? report.class_balance.validation_positive_rate ?? 0
  const prLift = baseline > 0 ? best.pr_auc_test / baseline : undefined
  const missesTarget = recall < targetRecall
  const hasManyFalseAlerts = precision < 0.5 && matrix.fp > matrix.tp
  const verdict = missesTarget ? 'À améliorer' : hasManyFalseAlerts ? 'Efficace mais bruyant' : 'Solide'
  const verdictDetail = missesTarget
    ? `Le rappel test est sous l'objectif ${formatPercent(targetRecall)} : des pannes restent manquées.`
    : hasManyFalseAlerts
      ? 'Le modèle capte les pannes, mais la précision indique beaucoup de fausses alertes.'
      : 'Le meilleur modèle atteint le rappel cible avec un compromis précision/coût lisible.'

  return (
    <Stack spacing={2}>
      <Box className="effectivenessHero">
        <Box>
          <Typography variant="caption" color="text.secondary">Verdict entraînement</Typography>
          <Typography variant="h5">{verdict}</Typography>
          <Typography variant="body2" color="text.secondary">{verdictDetail}</Typography>
        </Box>
        <Box className="effectivenessModel">
          <Typography variant="caption" color="text.secondary">Meilleur modèle</Typography>
          <Typography variant="body1" sx={{ fontWeight: 750 }}>{modelDisplayName(best.model)}</Typography>
          <Typography variant="caption" color="text.secondary">sélectionné sur PR-AUC validation</Typography>
        </Box>
      </Box>
      <Box className="summaryGrid reportKpiGrid">
        <MetricCard label="Recall test" value={formatPercent(recall)} helper={`objectif ${formatPercent(targetRecall)}`} info="Parmi les vraies pannes du test, part détectée par le modèle. C'est la métrique la plus opérationnelle si une panne manquée coûte cher." />
        <MetricCard label="Précision test" value={formatPercent(precision)} helper="fiabilité des alertes" info="Parmi les alertes déclenchées, part qui correspond vraiment à une panne. Une précision faible signifie plus de vérifications inutiles." />
        <MetricCard label="F1 test" value={formatNumber(f1)} helper="compromis alerte / panne" info="Score synthétique entre précision et rappel, utile pour comprendre le compromis choisi par le seuil." />
        <MetricCard label="Coût métier" value={formatNumber(businessCost)} helper={`FN ${matrix.fn} · FP ${matrix.fp}`} info="Score relatif calculé avec les pondérations métier du run : faux négatifs et faux positifs." />
      </Box>
      <Box className="reportDecisionGrid">
        <DefinitionItem label="PR-AUC test" detail={`${formatNumber(best.pr_auc_test)}${prLift ? ` · x${formatNumber(prLift)} vs baseline` : ''}`} />
        <DefinitionItem label="Baseline PR-AUC" detail={formatPercent(baseline)} />
        <DefinitionItem label="Seuil retenu" detail={formatNumber(best.threshold)} />
        <DefinitionItem label="Pannes manquées" detail={`${matrix.fn.toLocaleString('fr-FR')} faux négatifs`} />
      </Box>
    </Stack>
  )
}

function MaintenanceComparison({ report }: { report: MaintenanceMlReport }) {
  const bestResult = report.results.find((result) => result.model === report.best_model) ?? report.results[0]

  return (
    <Stack spacing={2}>
      <Box className="modelMetaGrid">
        <ModelMetaItem label="Dataset utilisé" value={report.gold_run_name} helper="Gold dataset source utilisé pour entraîner et évaluer les modèles de maintenance prédictive." />
        <ModelMetaItem label="Modèles lancés" value={(report.selected_models ?? []).map(modelDisplayName).join(', ') || '-'} helper="Liste des modèles réellement demandés au moment du lancement du run." />
        <ModelMetaItem label="Objectif seuil" value={thresholdStrategyLabel(report.threshold_strategy)} helper="Stratégie utilisée pour transformer les scores en alertes : équilibre, rappel prioritaire ou précision prioritaire." />
        <ModelMetaItem label="Seuil retenu" value={bestResult ? formatNumber(bestResult.threshold) : '-'} helper="Valeur calculée sur validation pour le meilleur modèle du run. En production, score >= seuil déclenche une alerte." />
        <ModelMetaItem label="Recall cible" value={formatPercent(report.target_recall ?? 0.8)} helper="Objectif de pannes réelles à détecter, fixé avant le lancement du run." />
        <ModelMetaItem label="Coût métier" value={`FN x${formatNumber(report.false_negative_cost ?? 20)} / FP x${formatNumber(report.false_positive_cost ?? 1)}`} helper="Score utilisé dans le comparatif : une panne manquée pèse plus qu'une fausse alerte." />
        <ModelMetaItem label="Baseline PR-AUC" value={formatPercent(report.pr_auc_random_baseline ?? report.class_balance.validation_positive_rate)} helper="Sur une courbe Precision-Recall, le modele aleatoire a une PR-AUC egale a la prevalence de la classe positive. Le modele doit depasser ce plancher." />
        <ModelMetaItem label="Poids XGBoost" value={formatNumber(report.xgboost_effective_scale_pos_weight ?? report.scale_pos_weight)} helper="Coefficient utilisé par XGBoost pour donner plus de poids aux pannes rares pendant l'entraînement." />
        <ModelMetaItem label="Arbre de décision" value={`d${report.decision_tree_max_depth ?? 6} / leaf ${report.decision_tree_min_samples_leaf ?? 10}`} helper="Profondeur maximale et nombre minimum de lignes par feuille configurés pour l'arbre de décision." />
        <ModelMetaItem label="Random Forest" value={`${report.random_forest_n_estimators ?? 60}x d${report.random_forest_max_depth ?? 12} leaf ${report.random_forest_min_samples_leaf ?? 2} split ${report.random_forest_min_samples_split ?? 10}`} helper={`Paramètres Random Forest : arbres, profondeur, feuilles, split, variables ${forestMaxFeaturesLabel(report.random_forest_max_features)} et bootstrap ${report.random_forest_bootstrap ?? true ? 'actif' : 'inactif'}.`} />
        <ModelMetaItem label="XGBoost" value={`${report.xgboost_n_estimators ?? 100}x d${report.xgboost_max_depth ?? 6} lr ${formatNumber(report.xgboost_learning_rate ?? 0.1)}`} helper="Nombre d'arbres, profondeur et learning rate configurés pour XGBoost." />
        <ModelMetaItem label="Seed" value={`${report.random_state ?? 42}`} helper="Graine aléatoire utilisée pour rendre le run reproductible." />
        <ModelMetaItem label="Suivi MLflow" value="Base SQLite locale" helper={`Les paramètres et métriques du run sont suivis dans MLflow : ${report.mlflow_tracking_uri}`} />
      </Box>
      {report.experiment_hypothesis && (
        <Alert severity="info">Note du run : {report.experiment_hypothesis}</Alert>
      )}
      <ModelArtifactsPanel report={report} result={bestResult} />
      <Typography variant="body2" color="text.secondary">
        Le tableau est trié par PR-AUC validation : plus cette valeur est haute, mieux le modèle retrouve les pannes rares sans se laisser tromper par la majorité des heures normales. La baseline PR-AUC correspond à la prévalence des pannes : c'est le niveau attendu d'un classement aléatoire.
      </Typography>
      <TableContainer className="tableBlock comparisonTable">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Modèle</TableCell>
              <TableCell>
                <MetricHeader
                  title="PR-AUC train"
                  helper="Score sur apprentissage"
                  detail="Score obtenu sur les données vues pendant l'entraînement. S'il est beaucoup plus haut que validation/test, le modèle a peut-être trop mémorisé le train."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="PR-AUC val."
                  helper="Score principal sur validation"
                  detail="Mesure prioritaire ici, car les pannes sont rares. Elle évalue si le modèle retrouve bien les vraies pannes parmi ses alertes sur le jeu de validation. Plus c'est haut, mieux c'est."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="ROC-AUC val."
                  helper="Séparation globale validation"
                  detail="Mesure la capacité du modèle à donner un score plus élevé aux futures pannes qu'aux heures normales sur le jeu de validation. Utile, mais moins prioritaire que PR-AUC quand les pannes sont rares."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="PR-AUC test"
                  helper="Score principal final"
                  detail="Même logique que PR-AUC validation, mais sur le jeu de test final. C'est l'indicateur le plus réaliste pour savoir si le modèle devrait tenir sur des données jamais vues."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="ROC-AUC test"
                  helper="Séparation globale finale"
                  detail="Même logique que ROC-AUC validation, mais sur le jeu de test final. Il vérifie la séparation globale entre pannes et non-pannes sur des données jamais vues."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="Précision test"
                  helper="Qualité des alertes"
                  detail="Parmi les alertes déclenchées par le modèle, indique la part qui correspond vraiment à une panne. Une précision basse veut dire beaucoup de fausses alertes."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="Rappel test"
                  helper="Pannes retrouvées"
                  detail="Parmi les vraies pannes du test, indique la part détectée par le modèle. En maintenance prédictive, c'est souvent critique car un rappel bas signifie des pannes manquées."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="F1 test"
                  helper="Équilibre précision/rappel"
                  detail="Score qui combine précision et rappel. Il aide à choisir un seuil quand on veut limiter les fausses alertes sans manquer trop de pannes."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="Coût métier"
                  helper="FN/FP pondérés"
                  detail="Score calculé avec les coûts configurés : FN x coût panne manquée + FP x coût fausse alerte. Plus c'est bas, mieux c'est."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="Temps train"
                  helper="Entraînement + CV"
                  detail="Durée mesurée pour entraîner le modèle et calculer sa validation croisée temporelle."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="Inférence"
                  helper="ms par ligne"
                  detail="Temps moyen estimé pour produire un score sur une ligne du jeu de test."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="CV PR-AUC"
                  helper="Moyenne ± écart sur splits temporels"
                  detail="Résultat moyen de PR-AUC sur plusieurs découpes temporelles. L'écart indique la stabilité : un écart faible veut dire que le modèle se comporte de façon plus régulière dans le temps."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="Seuil"
                  helper="Score minimum pour déclencher une alerte"
                  detail="Score à partir duquel l'application considère qu'il faut alerter. Un seuil plus bas détecte plus de pannes mais peut créer plus de fausses alertes. Un seuil plus haut réduit les alertes inutiles mais peut manquer des pannes."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="MLflow"
                  helper="Run ID"
                  detail="Identifiant du run MLflow correspondant à cette ligne. Il permet de retrouver paramètres, métriques, artefacts et modèle loggé."
                />
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {report.results.map((result) => (
              <TableRow key={result.model} selected={result.model === report.best_model}>
                <TableCell>{modelDisplayName(result.model)}</TableCell>
                <TableCell>{formatNumber(result.pr_auc_train)}</TableCell>
                <TableCell>{formatNumber(result.pr_auc_validation)}</TableCell>
                <TableCell>{formatNumber(result.roc_auc_validation)}</TableCell>
                <TableCell>{formatNumber(result.pr_auc_test)}</TableCell>
                <TableCell>{formatNumber(result.roc_auc_test)}</TableCell>
                <TableCell>{formatNumber(result.precision_test)}</TableCell>
                <TableCell>{formatNumber(result.recall_test)}</TableCell>
                <TableCell>{formatNumber(result.f1_test)}</TableCell>
                <TableCell>{formatNumber(result.business_cost_test)}</TableCell>
                <TableCell>{formatNumber(result.training_seconds)}s</TableCell>
                <TableCell>{formatNumber(result.inference_ms_per_row)} ms</TableCell>
                <TableCell>{formatNumber(result.cv_pr_auc_mean)} ± {formatNumber(result.cv_pr_auc_std)}</TableCell>
                <TableCell>{formatNumber(result.threshold)}</TableCell>
                <TableCell>{result.mlflow_run_id ? shortMlflowRunId(result.mlflow_run_id) : '-'}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Stack>
  )
}

function ModelMetaItem({ label, value, helper }: { label: string; value: string; helper?: string }) {
  return (
    <Box className="modelMetaItem">
      <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        {helper && <InfoTooltip title={helper} />}
      </Stack>
      <Typography variant="body2">{value}</Typography>
    </Box>
  )
}

function ModelArtifactsPanel({ report, result }: { report: MaintenanceMlReport; result?: MaintenanceMlResult }) {
  const artifacts = Object.entries(result?.artifacts ?? {})
  if (!result || !artifacts.length) {
    return (
      <Alert severity="info">
        Les nouveaux runs produiront des artefacts MLflow locaux : courbes, matrices de confusion, importances et modèle sérialisé.
      </Alert>
    )
  }
  const imageArtifacts = artifacts.filter(([key]) => key.endsWith('_png'))
  const fileArtifacts = artifacts.filter(([key]) => !key.endsWith('_png'))
  return (
    <Box className="artifactPanel">
      <Stack spacing={0.25}>
        <Typography variant="body2" sx={{ fontWeight: 750 }}>Artefacts du meilleur modèle</Typography>
        <Typography variant="caption" color="text.secondary">
          Fichiers générés pendant le run et loggés dans MLflow quand disponible.
        </Typography>
      </Stack>
      <Box className="artifactImageGrid">
        {imageArtifacts.map(([key, path]) => (
          <Box key={key} className="artifactImageItem">
            <Typography variant="caption" color="text.secondary">{artifactLabel(key)}</Typography>
            <img src={artifactUrl(report.run_id, path)} alt={artifactLabel(key)} />
          </Box>
        ))}
      </Box>
      {!!fileArtifacts.length && (
        <Box className="runConfigSummary">
          {fileArtifacts.map(([key, path]) => (
            <a key={key} className="runConfigItem artifactLink" href={artifactUrl(report.run_id, path)} target="_blank" rel="noreferrer">
              <Typography variant="caption">{artifactLabel(key)}</Typography>
            </a>
          ))}
        </Box>
      )}
    </Box>
  )
}

function artifactUrl(runId: string, path: string) {
  return `/api/maintenance-ml-runs/${runId}/artifacts/${path}`
}

function artifactLabel(key: string) {
  const labels: Record<string, string> = {
    validation_confusion_matrix_csv: 'Matrice validation CSV',
    test_confusion_matrix_csv: 'Matrice test CSV',
    test_confusion_matrix_png: 'Matrice de confusion',
    precision_recall_curve_png: 'Courbe précision-rappel',
    roc_curve_png: 'Courbe ROC',
    shap_summary_png: 'SHAP global',
    shap_waterfall_png: 'SHAP local',
    feature_importances_csv: 'Importances CSV',
    model_pickle: 'Modèle pickle',
  }
  return labels[key] ?? key
}

function MetricHeader({ title, helper, detail }: { title: string; helper: string; detail: string }) {
  return (
    <Stack spacing={0.25}>
      <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
        <Typography variant="body2" sx={{ fontWeight: 750 }}>{title}</Typography>
        <InfoTooltip title={detail} />
      </Stack>
      <Typography variant="caption" color="text.secondary">{helper}</Typography>
    </Stack>
  )
}

function ConfusionSummary({ result }: { result: MaintenanceMlResult }) {
  const matrix = result.test_confusion_matrix
  return (
    <Stack spacing={2}>
      <Box className="confusionGrid">
        <MetricCard label="Précision" value={formatPercent(result.precision_test)} helper="Parmi les alertes, part de vraies pannes." info="Si la précision est basse, le modèle déclenche beaucoup d'alertes inutiles. C'est coûteux en vérifications ou interventions." />
        <MetricCard label="Rappel" value={formatPercent(result.recall_test)} helper="Parmi les pannes, part détectée." info="Si le rappel est bas, le modèle laisse passer des pannes sans alerte. C'est souvent le risque le plus grave en maintenance prédictive." />
        <MetricCard label="F1-score" value={formatNumber(result.f1_test)} helper="Compromis précision / rappel." info="Le F1-score résume l'équilibre entre éviter les fausses alertes et retrouver les vraies pannes." />
        <MetricCard label="Seuil 0.5" value={formatNumber(result.test_metrics_at_05?.f1)} helper="F1 avec le seuil standard." info="Comparaison avec le seuil classique 0.5. Le seuil choisi automatiquement peut être meilleur pour détecter les pannes rares." />
      </Box>
      <Box className="confusionGrid">
        <MetricCard label="Vrais négatifs" value={matrix.tn.toLocaleString('fr-FR')} helper="Pas d'alerte, et aucune panne réelle ensuite." info="Cas correctement ignorés : le modèle n'a pas alerté et il n'y avait effectivement pas de panne à venir." />
        <MetricCard label="Faux positifs" value={matrix.fp.toLocaleString('fr-FR')} helper="Alerte déclenchée, mais pas de panne ensuite." info="Alertes inutiles : elles demandent une vérification humaine ou une intervention, mais aucune panne n'arrive ensuite." />
        <MetricCard label="Faux négatifs" value={matrix.fn.toLocaleString('fr-FR')} helper="Pas d'alerte, alors qu'une panne arrive ensuite." info="Pannes manquées : c'est le risque le plus important en maintenance prédictive, car la machine tombe en panne sans alerte préalable." />
        <MetricCard label="Vrais positifs" value={matrix.tp.toLocaleString('fr-FR')} helper="Alerte déclenchée, et panne réelle ensuite." info="Pannes correctement détectées : le modèle a déclenché une alerte avant une panne réelle." />
      </Box>
    </Stack>
  )
}

function FeatureInsightPanel({ result }: { result: MaintenanceMlResult }) {
  const features = result.top_features ?? []
  if (!features.length) {
    return (
      <Alert severity="info">
        Aucune importance de variable disponible pour ce modèle. Les nouveaux runs en fourniront quand le modèle le permet.
      </Alert>
    )
  }

  return (
    <Stack spacing={1.5}>
      <Typography variant="body2" color="text.secondary">
        Top variables du meilleur modèle. Elles aident à expliquer simplement quels signaux pèsent le plus dans le score de panne.
      </Typography>
      <TableContainer className="tableBlock comparisonTable">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Variable</TableCell>
              <TableCell>
                <MetricHeader
                  title="Poids"
                  helper="Importance relative"
                  detail="Pour un arbre ou une forêt, c'est l'importance de la variable dans les séparations. Pour une régression logistique, c'est la force du coefficient."
                />
              </TableCell>
              <TableCell>
                <MetricHeader
                  title="Sens"
                  helper="Régression logistique"
                  detail="Quand disponible, indique si la variable augmente ou diminue le score de panne. Les arbres donnent plutôt une importance, pas un sens unique."
                />
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {features.map((feature) => (
              <TableRow key={feature.feature}>
                <TableCell>{feature.feature}</TableCell>
                <TableCell>{formatNumber(feature.importance)}</TableCell>
                <TableCell>{feature.direction ?? '-'}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Stack>
  )
}

function ShapExplanationPanel({ report, result }: { report: MaintenanceMlReport; result: MaintenanceMlResult }) {
  const shap = result.shap_explanations
  if (!shap?.available) {
    return (
      <Alert severity="info">
        SHAP sera généré sur les nouveaux runs pour Random Forest, RF rééquilibré et XGBoost si la librairie SHAP est installée. {shap?.reason ? `Détail : ${shap.reason}` : ''}
      </Alert>
    )
  }

  const sample = shap.sample
  return (
    <Stack spacing={1.5}>
      <Alert severity="warning">
        SHAP explique une contribution au score, pas une causalité. En cas de variables corrélées, il faut lire les facteurs ensemble et vérifier l'absence de leakage.
      </Alert>
      <Box className="runCompareHelp">
        <Typography variant="body2" sx={{ fontWeight: 750 }}>Explication locale</Typography>
        {sample ? (
          <Box className="runCompareHelpGrid">
            <DefinitionItem label="Ligne expliquée" detail={`${sample.row_number}`} />
            <DefinitionItem label="Score" detail={`${formatPercent(sample.score)} vs seuil ${formatNumber(sample.threshold)}`} />
            <DefinitionItem label="Décision" detail={sample.is_alert ? 'Alerte panne déclenchée' : 'Pas d’alerte'} />
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">Aucun exemple local disponible pour ce run.</Typography>
        )}
        {!!sample?.factors?.length && (
          <Box className="shapNarrative">
            <Typography variant="caption" color="text.secondary">Narrative métier</Typography>
            {sample.factors.slice(0, 3).map((factor) => (
              <Typography key={`${factor.feature}-${factor.contribution}`} variant="body2">
                {factor.feature} = {formatNumber(factor.value)} {factor.direction} le risque ({formatNumber(factor.contribution)}).
              </Typography>
            ))}
          </Box>
        )}
      </Box>
      {!!shap.top_features?.length && (
        <TableContainer className="tableBlock comparisonTable">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Variable SHAP</TableCell>
                <TableCell>Impact moyen</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {shap.top_features.map((feature) => (
                <TableRow key={feature.feature}>
                  <TableCell>{feature.feature}</TableCell>
                  <TableCell>{formatNumber(feature.mean_abs_shap)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
      <Typography variant="caption" color="text.secondary">
        Les images SHAP globales et locales sont aussi visibles dans les artefacts du meilleur modèle du run {shortRunId(report.run_id)}.
      </Typography>
    </Stack>
  )
}

function modelDisplayName(model: string) {
  const names: Record<string, string> = {
    logistic_regression: 'Régression logistique',
    decision_tree: 'Arbre de décision',
    random_forest: 'Random Forest',
    random_forest_balanced: 'RF rééquilibré',
    xgboost: 'XGBoost',
  }
  return names[model] ?? model
}

function eventLabel(step: string) {
  const labels: Record<string, string> = {
    start: 'Démarrage',
    load_dataset: 'Dataset',
    prepare_features: 'Features',
    split: 'Découpage',
    class_balance: 'Classes',
    train_model: 'Entraînement',
    threshold: 'Seuil',
    evaluate_model: 'Évaluation',
    select_best_model: 'Sélection',
    finish: 'Fin',
  }
  return labels[step] ?? step
}

function eventDetailLabel(key: string) {
  const labels: Record<string, string> = {
    rows: 'lignes',
    columns: 'colonnes',
    features: 'features',
    dataset_hash: 'hash',
    train_rows: 'train',
    validation_rows: 'validation',
    test_rows: 'test',
    train_positive_rate: 'taux panne',
    train_positive_count: 'pannes train',
    train_negative_count: 'normaux train',
    model: 'modèle',
    threshold: 'seuil',
    strategy: 'stratégie',
    validation_pr_auc: 'PR-AUC val.',
    test_pr_auc: 'PR-AUC test',
    test_recall: 'recall test',
    business_cost: 'coût',
  }
  return labels[key] ?? key
}

function formatEventValue(value: unknown) {
  if (typeof value === 'number') {
    if (Math.abs(value) <= 1) return formatNumber(value)
    return Number.isInteger(value) ? value.toLocaleString('fr-FR') : formatNumber(value)
  }
  if (Array.isArray(value)) return value.join(', ')
  return String(value)
}

function formatDateTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('fr-FR', {
    day: '2-digit',
    month: '2-digit',
    year: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function bestMlflowRun(runs: MlflowTrackingSummary['runs'], metric: keyof MlflowTrackingSummary['runs'][number], highest = true) {
  const candidates = runs.filter((run) => typeof run[metric] === 'number')
  if (!candidates.length) return undefined
  return candidates.reduce((best, run) => {
    const bestValue = best[metric] as number
    const runValue = run[metric] as number
    return highest ? (runValue > bestValue ? run : best) : (runValue < bestValue ? run : best)
  })
}

function latestMlflowRun(runs: MlflowTrackingSummary['runs']) {
  const datedRuns = runs.filter((run) => run.start_time)
  if (!datedRuns.length) return undefined
  return datedRuns.reduce((latest, run) => {
    const latestTime = Date.parse(latest.start_time ?? '')
    const runTime = Date.parse(run.start_time ?? '')
    return runTime > latestTime ? run : latest
  })
}

function sortMlflowRunsByDateDesc(runs: MlflowTrackingSummary['runs']) {
  return [...runs].sort((left, right) => {
    const leftTime = Date.parse(left.start_time ?? '')
    const rightTime = Date.parse(right.start_time ?? '')
    const leftHasDate = !Number.isNaN(leftTime)
    const rightHasDate = !Number.isNaN(rightTime)
    if (leftHasDate && rightHasDate && leftTime !== rightTime) return rightTime - leftTime
    if (leftHasDate !== rightHasDate) return leftHasDate ? -1 : 1
    return right.run_id.localeCompare(left.run_id)
  })
}

function shortMlflowRunId(runId: string) {
  return runId ? runId.slice(0, 8) : '-'
}

function modelDetail(model: string) {
  const details: Record<string, string> = {
    logistic_regression: 'Modèle simple et interprétable qui sert souvent de référence de base.',
    decision_tree: 'Arbre lisible sous forme de règles. Utile pour comprendre les séparations, mais sensible au surapprentissage.',
    random_forest: "Ensemble de plusieurs arbres. Plus robuste qu'un arbre seul sur des données tabulaires.",
    random_forest_balanced: "Random Forest avec compensation des classes : les pannes rares pèsent plus pendant l'entraînement.",
    xgboost: 'Modèle de boosting performant qui corrige progressivement ses erreurs. Souvent fort sur données tabulaires.',
  }
  return details[model] ?? 'Modèle entraîné pendant ce run.'
}

function targetWindowLabel(labelColumn: string) {
  const match = labelColumn.match(/next_(\d+)h/)
  return match ? `cible ${match[1]}h` : labelColumn
}

function targetWindowDetail(labelColumn: string) {
  const match = labelColumn.match(/next_(\d+)h/)
  if (!match) return 'Colonne cible utilisée pour apprendre ce que le modèle doit prédire.'
  return `Le modèle cherche à prédire si une panne arrive dans les ${match[1]} prochaines heures.`
}

function thresholdStrategyLabel(strategy?: ThresholdStrategy) {
  if (!strategy) return 'Non enregistré'
  const labels: Record<ThresholdStrategy, string> = {
    balanced: 'Équilibré F1',
    recall: 'Détecter plus',
    precision: 'Limiter alertes',
    target_recall: 'Recall cible',
  }
  return labels[strategy ?? 'balanced']
}

function thresholdStrategyDetail(strategy?: ThresholdStrategy) {
  if (!strategy) return "Ce run ne contient pas la stratégie de seuil dans ses métadonnées. Relance-le après redémarrage du backend pour l'enregistrer."
  const details: Record<ThresholdStrategy, string> = {
    balanced: 'Cherche un compromis entre précision et rappel avec le F1-score.',
    recall: 'Favorise la détection des pannes, avec plus de risque de fausses alertes.',
    precision: 'Favorise des alertes plus fiables, avec plus de risque de manquer des pannes.',
    target_recall: "Cherche un seuil qui atteint l'objectif de recall configuré, si les scores du modèle le permettent.",
  }
  return details[strategy ?? 'balanced']
}

function formatRunModels(run: MaintenanceMlRunInfo) {
  const models = run.selected_models ?? ['logistic_regression', 'decision_tree', 'random_forest', 'random_forest_balanced', 'xgboost']
  return models.map(shortModelName).join(', ')
}

function formatRunModelsDetail(run: MaintenanceMlRunInfo) {
  const models = run.selected_models ?? ['logistic_regression', 'decision_tree', 'random_forest', 'random_forest_balanced', 'xgboost']
  return `Modèles entraînés : ${models.map(modelDisplayName).join(', ')}.`
}

function shortModelName(model: string) {
  const names: Record<string, string> = {
    logistic_regression: 'LogReg',
    decision_tree: 'Tree',
    random_forest: 'RF',
    random_forest_balanced: 'RF rééqu.',
    xgboost: 'XGB',
  }
  return names[model] ?? model
}

function formatTreeSettings(run: MaintenanceMlRunInfo) {
  return `d${run.decision_tree_max_depth ?? 6} / leaf ${run.decision_tree_min_samples_leaf ?? 10}`
}

function formatForestSettings(run: MaintenanceMlRunInfo) {
  const variants = [
    run.selected_models?.includes('random_forest') ? 'standard' : '',
    run.selected_models?.includes('random_forest_balanced') || run.random_forest_balanced ? 'rééquilibré' : '',
  ].filter(Boolean).join(' + ')
  return `${run.random_forest_n_estimators ?? 60}x d${run.random_forest_max_depth ?? 12} leaf ${run.random_forest_min_samples_leaf ?? 2} split ${run.random_forest_min_samples_split ?? 10} · ${forestMaxFeaturesLabel(run.random_forest_max_features)} · ${run.random_forest_bootstrap ?? true ? 'bootstrap' : 'no bootstrap'}${variants ? ` · ${variants}` : ''}`
}

function forestMaxFeaturesLabel(value?: string | null) {
  if (value === 'log2') return 'max_features log2'
  if (value === 'all' || value === 'none' || value === null) return 'toutes variables'
  return 'max_features sqrt'
}

function forestBalanceLabel(balanced?: boolean) {
  return balanced ? 'RF rééquilibré' : 'RF non rééquilibré'
}

function forestBalanceDetail(balanced?: boolean) {
  if (balanced) return 'Le Random Forest donne plus de poids aux exemples de panne, car ils sont rares dans le dataset.'
  return "Le Random Forest n'applique pas de compensation spéciale pour les pannes rares."
}

function B7StudySummary({ report }: { report: MaintenanceMlReport }) {
  const carbon = report.carbon
  const tuning = report.tuning as { enabled?: boolean; available?: boolean; mode?: string; best_value?: number; trials?: number; pruned_trials?: number; reason?: string } | undefined
  if (!carbon && !tuning?.enabled) return null
  return <Paper className="panel">
    <SectionHeader title="B7 · optimisation et éco-conception" />
    <Box className="summaryGrid">
      <MetricCard label="Étude Optuna" value={tuning?.enabled ? (tuning.available === false ? 'indisponible' : tuning.mode ?? 'lancée') : 'non lancée'} helper={tuning?.reason ?? `${tuning?.trials ?? 0} essai(s), ${tuning?.pruned_trials ?? 0} élagué(s)`} />
      <MetricCard label="PR-AUC CV Optuna" value={typeof tuning?.best_value === 'number' ? tuning.best_value.toFixed(3) : '—'} helper="moyenne des folds temporels" />
      <MetricCard label="Durée mesurée" value={carbon?.duration_seconds != null ? `${carbon.duration_seconds.toFixed(1)} s` : '—'} helper="entraînement et évaluation" />
      <MetricCard label="CO₂ estimé" value={carbon?.emissions_gco2eq != null ? `${carbon.emissions_gco2eq.toFixed(2)} g` : '—'} helper={carbon?.available ? `${carbon.energy_kwh?.toFixed(5) ?? '—'} kWh` : carbon?.reason ?? 'CodeCarbon'} />
    </Box>
    {report.b7_artifacts?.performance_vs_carbon_png && <Button size="small" href={`/api/maintenance-ml-runs/${report.run_id}/artifacts/${report.b7_artifacts.performance_vs_carbon_png}`} target="_blank">Voir le graphe performance / CO₂</Button>}
    {report.b7_artifacts?.arbitration_csv && <Button size="small" href={`/api/maintenance-ml-runs/${report.run_id}/artifacts/${report.b7_artifacts.arbitration_csv}`} target="_blank">Télécharger le tableau d’arbitrage</Button>}
  </Paper>
}

function bestResult(report: MaintenanceMlReport) {
  return report.results.find((result) => result.model === report.best_model) ?? report.results[0]
}

function shortRunId(runId: string) {
  return runId.replace('maintenance_ml_', '').slice(0, 17)
}

function clampInteger(value: string, min: number, max: number) {
  const parsed = Number.parseInt(value, 10)
  if (Number.isNaN(parsed)) return min
  return Math.min(max, Math.max(min, parsed))
}

function clampFloat(value: string, min: number, max: number) {
  const parsed = Number.parseFloat(value.replace(',', '.'))
  if (Number.isNaN(parsed)) return min
  return Math.min(max, Math.max(min, parsed))
}
