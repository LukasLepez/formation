import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Container,
  FormControl,
  FormControlLabel,
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
  Typography,
} from '@mui/material'
import type { SelectChangeEvent } from '@mui/material'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import RefreshIcon from '@mui/icons-material/Refresh'
import ScienceIcon from '@mui/icons-material/Science'
import TerminalIcon from '@mui/icons-material/Terminal'
import { api, apiText, messageFrom } from '../lib/api'
import { formatNumber, formatPercent } from '../lib/format'
import { statusLabel, statusValueColor } from '../lib/status'
import type { GoldCsvInfo, MaintenanceMlReport, MaintenanceMlResult, MaintenanceMlRunInfo } from '../types'
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
  const [randomForestBalanced, setRandomForestBalanced] = useState(true)
  const [maintenanceLogs, setMaintenanceLogs] = useState('')
  const [maintenanceReport, setMaintenanceReport] = useState<MaintenanceMlReport | null>(null)
  const [activeTab, setActiveTab] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const latestMaintenanceRun = maintenanceRuns[0]

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
    } catch (refreshError) {
      setError(messageFrom(refreshError))
    }
  }

  async function launchMaintenanceRun() {
    setBusy(true)
    setError('')
    setMaintenanceReport(null)
    try {
      const run = await api<MaintenanceMlRunInfo>('/maintenance-ml-runs', {
        method: 'POST',
        body: JSON.stringify({
          label_column: mlLabel,
          gold_run_name: mlGoldRun || null,
          random_forest_balanced: randomForestBalanced,
        }),
      })
      setSelectedMaintenanceRunId(run.run_id)
      setActiveTab(1)
      await refresh()
    } catch (mlError) {
      setError(messageFrom(mlError))
    } finally {
      setBusy(false)
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

  async function openMaintenanceReport(runId = selectedMaintenanceRunId) {
    if (!runId) return
    await loadMaintenanceReport(runId)
    setActiveTab(1)
  }

  useEffect(() => {
    void refresh()
  }, [])

  useEffect(() => {
    if (!selectedMaintenanceRunId) return
    const selected = maintenanceRuns.find((run) => run.run_id === selectedMaintenanceRunId)
    if (selected?.status === 'success') void loadMaintenanceReport(selectedMaintenanceRunId)
  }, [selectedMaintenanceRunId, maintenanceRuns])

  useEffect(() => {
    if (!selectedMaintenanceRunId) return
    let active = true
    async function poll() {
      try {
        const text = await apiText(`/maintenance-ml-runs/${selectedMaintenanceRunId}/logs/raw`)
        if (active) setMaintenanceLogs(text)
        const current = await api<MaintenanceMlRunInfo>(`/maintenance-ml-runs/${selectedMaintenanceRunId}`)
        await refresh()
        if (current?.status === 'success') await loadMaintenanceReport(selectedMaintenanceRunId)
      } catch {
        if (active) setMaintenanceLogs('')
      }
    }
    void poll()
    const interval = window.setInterval(() => void poll(), 3000)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [selectedMaintenanceRunId])

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

        {busy && <LinearProgress />}
        {error && <Alert severity="error">{error}</Alert>}

        <Box className="summaryGrid">
          <MetricCard
            label="Dernier run ML"
            value={latestMaintenanceRun ? statusLabel[latestMaintenanceRun.status] : '-'}
            helper={latestMaintenanceRun?.run_id ?? 'Aucun run'}
            valueColor={latestMaintenanceRun ? statusValueColor[latestMaintenanceRun.status] : undefined}
          />
          <MetricCard label="Meilleur modèle" value={latestMaintenanceRun?.best_model ?? maintenanceReport?.best_model ?? '-'} helper="sélection par PR-AUC validation" />
          <MetricCard label="Lignes Gold" value={maintenanceReport?.rows.toLocaleString('fr-FR') ?? latestMaintenanceRun?.rows?.toLocaleString('fr-FR') ?? '-'} helper={`${maintenanceReport?.features ?? latestMaintenanceRun?.features ?? '-'} features`} />
          <MetricCard label="Taux panne train" value={formatPercent(maintenanceReport?.class_balance.train_positive_rate)} helper="accuracy volontairement ignorée" />
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
          </Tabs>
        </Paper>

        {activeTab === 0 && (
          <Stack spacing={2}>
            <Paper className="panel">
              <SectionHeader title="Réglages d'entraînement" icon={<ScienceIcon fontSize="small" />} />
              <Stack spacing={2}>
                <Stack direction={{ xs: 'column', xl: 'row' }} spacing={2}>
                  <FormControl size="small" sx={{ minWidth: 0, flex: 1 }}>
                    <InputLabel>Gold dataset</InputLabel>
                    <Select value={mlGoldRun} label="Gold dataset" onChange={(event: SelectChangeEvent) => setMlGoldRun(event.target.value)}>
                      {goldCsvs.map((csv) => <MenuItem key={csv.run_name} value={csv.run_name}>{csv.run_name}</MenuItem>)}
                    </Select>
                  </FormControl>
                  <FormControl size="small" sx={{ minWidth: 0, flex: 1 }}>
                    <InputLabel>Cible</InputLabel>
                    <Select value={mlLabel} label="Cible" onChange={(event: SelectChangeEvent) => setMlLabel(event.target.value)}>
                      <MenuItem value="label_failure_next_6h">label_failure_next_6h</MenuItem>
                      <MenuItem value="label_failure_next_12h">label_failure_next_12h</MenuItem>
                      <MenuItem value="label_failure_next_24h">label_failure_next_24h</MenuItem>
                      <MenuItem value="label_failure_next_48h">label_failure_next_48h</MenuItem>
                    </Select>
                  </FormControl>
                </Stack>
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ alignItems: { md: 'center' }, justifyContent: 'space-between' }}>
                  <FormControlLabel
                    control={<Switch checked={randomForestBalanced} onChange={(_, checked) => setRandomForestBalanced(checked)} />}
                    label={
                      <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center' }}>
                        <span>Rééquilibrage Random Forest</span>
                        <InfoTooltip title="Activé : le Random Forest donne plus d'importance aux pannes rares. Désactivé : il apprend sans compensation, donc il peut favoriser les heures sans panne." />
                      </Stack>
                    }
                  />
                  <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap' }}>
                    <Button variant="contained" startIcon={<PlayArrowIcon />} onClick={() => void launchMaintenanceRun()} disabled={busy || !goldCsvs.length}>
                      Entraîner
                    </Button>
                    <Button variant="outlined" startIcon={<RefreshIcon />} onClick={() => void openMaintenanceReport()} disabled={!selectedMaintenanceRunId}>
                      Rapport
                    </Button>
                  </Stack>
                </Stack>
              </Stack>
            </Paper>
            <Paper className="panel">
              <SectionHeader title="Logs ML" icon={<TerminalIcon fontSize="small" />} />
              <Typography variant="caption" color="text.secondary">{maintenanceLogs.split('\n').filter(Boolean).length.toLocaleString('fr-FR')} lignes</Typography>
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
                action={
                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                    <MaintenanceRunPicker runs={maintenanceRuns} value={selectedMaintenanceRunId} onChange={setSelectedMaintenanceRunId} />
                    <Button variant="outlined" startIcon={<RefreshIcon />} onClick={() => void openMaintenanceReport()} disabled={!selectedMaintenanceRunId}>
                      Afficher le rapport
                    </Button>
                  </Stack>
                }
              />
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Sélectionne un run pour consulter ses logs sur le côté. Le rapport du run réussi apparaît sous la liste.
              </Typography>
              <MaintenanceRunTable runs={maintenanceRuns} selected={selectedMaintenanceRunId} onSelect={setSelectedMaintenanceRunId} />
            </Paper>
            {maintenanceReport ? (
              <>
                <Paper className="panel">
                  <SectionHeader title="Comparatif modèles" icon={<ScienceIcon fontSize="small" />} />
                  <MaintenanceComparison report={maintenanceReport} />
                </Paper>
                <Paper className="panel">
                  <SectionHeader title="Matrice de confusion" />
                  <ConfusionSummary result={maintenanceReport.results[0]} />
                </Paper>
                <Alert severity="success">{maintenanceReport.conclusion}</Alert>
              </>
            ) : (
              <Alert severity="info">Aucun rapport chargé pour le moment. Lance un entraînement ou choisis un run réussi.</Alert>
            )}
          </Stack>
        )}
      </Stack>
    </Container>
  )
}

function MaintenanceRunPicker({ runs, value, onChange }: { runs: MaintenanceMlRunInfo[]; value: string; onChange: (value: string) => void }) {
  return (
    <FormControl size="small" sx={{ minWidth: 300 }}>
      <InputLabel>Run ML</InputLabel>
      <Select value={value} label="Run ML" onChange={(event: SelectChangeEvent) => onChange(event.target.value)}>
        {runs.map((run) => <MenuItem key={run.run_id} value={run.run_id}>{run.run_id}</MenuItem>)}
      </Select>
    </FormControl>
  )
}

function MaintenanceRunTable({ runs, selected, onSelect }: { runs: MaintenanceMlRunInfo[]; selected: string; onSelect: (id: string) => void }) {
  return (
    <TableContainer sx={{ maxHeight: 430 }}>
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Run</TableCell>
            <TableCell>Statut</TableCell>
            <TableCell>Gold</TableCell>
            <TableCell>Cible</TableCell>
            <TableCell>Meilleur</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.run_id} hover selected={run.run_id === selected} onClick={() => onSelect(run.run_id)} sx={{ cursor: 'pointer' }}>
              <TableCell>{run.run_id}</TableCell>
              <TableCell><StatusChip status={run.status} /></TableCell>
              <TableCell>{run.gold_run_name ?? '-'}</TableCell>
              <TableCell>{run.label_column}</TableCell>
              <TableCell>{run.best_model ?? '-'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

function MaintenanceComparison({ report }: { report: MaintenanceMlReport }) {
  return (
    <Stack spacing={2}>
      <Box className="modelMetaGrid">
        <ModelMetaItem label="Dataset utilisé" value={report.gold_run_name} />
        <ModelMetaItem label="Poids XGBoost" value={formatNumber(report.scale_pos_weight)} helper="Compensation des pannes rares" />
        <ModelMetaItem label="Random Forest" value={report.random_forest_balanced ? 'Rééquilibré' : 'Sans rééquilibrage'} />
        <ModelMetaItem label="Suivi MLflow" value="Base SQLite locale" helper={report.mlflow_tracking_uri} />
      </Box>
      <Typography variant="body2" color="text.secondary">
        Le tableau est trié par PR-AUC validation : plus cette valeur est haute, mieux le modèle retrouve les pannes rares sans se laisser tromper par la majorité des heures normales.
      </Typography>
      <TableContainer className="tableBlock comparisonTable">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Modèle</TableCell>
              <TableCell><MetricHeader title="PR-AUC val." helper="Score principal sur validation" /></TableCell>
              <TableCell><MetricHeader title="ROC-AUC val." helper="Séparation globale validation" /></TableCell>
              <TableCell><MetricHeader title="PR-AUC test" helper="Score principal final" /></TableCell>
              <TableCell><MetricHeader title="ROC-AUC test" helper="Séparation globale finale" /></TableCell>
              <TableCell><MetricHeader title="CV PR-AUC" helper="Moyenne ± écart sur splits temporels" /></TableCell>
              <TableCell><MetricHeader title="Seuil" helper="Score minimum pour déclencher une alerte" /></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {report.results.map((result) => (
              <TableRow key={result.model} selected={result.model === report.best_model}>
                <TableCell>{result.model}</TableCell>
                <TableCell>{formatNumber(result.pr_auc_validation)}</TableCell>
                <TableCell>{formatNumber(result.roc_auc_validation)}</TableCell>
                <TableCell>{formatNumber(result.pr_auc_test)}</TableCell>
                <TableCell>{formatNumber(result.roc_auc_test)}</TableCell>
                <TableCell>{formatNumber(result.cv_pr_auc_mean)} ± {formatNumber(result.cv_pr_auc_std)}</TableCell>
                <TableCell>{formatNumber(result.threshold)}</TableCell>
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
    <Box className="modelMetaItem" title={helper}>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography variant="body2">{value}</Typography>
      {helper && <Typography variant="caption" color="text.secondary" className="metaHelper">{helper}</Typography>}
    </Box>
  )
}

function MetricHeader({ title, helper }: { title: string; helper: string }) {
  return (
    <Stack spacing={0.25}>
      <Typography variant="body2" sx={{ fontWeight: 750 }}>{title}</Typography>
      <Typography variant="caption" color="text.secondary">{helper}</Typography>
    </Stack>
  )
}

function ConfusionSummary({ result }: { result: MaintenanceMlResult }) {
  const matrix = result.test_confusion_matrix
  return (
    <Box className="confusionGrid">
      <MetricCard label="Vrais négatifs" value={matrix.tn.toLocaleString('fr-FR')} helper="Pas d'alerte, et aucune panne réelle ensuite." />
      <MetricCard label="Faux positifs" value={matrix.fp.toLocaleString('fr-FR')} helper="Alerte déclenchée, mais pas de panne ensuite." />
      <MetricCard label="Faux négatifs" value={matrix.fn.toLocaleString('fr-FR')} helper="Pas d'alerte, alors qu'une panne arrive ensuite." />
      <MetricCard label="Vrais positifs" value={matrix.tp.toLocaleString('fr-FR')} helper="Alerte déclenchée, et panne réelle ensuite." />
    </Box>
  )
}
