import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
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
  Tooltip,
  Typography,
} from '@mui/material'
import type { SelectChangeEvent } from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
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
  const [runToDelete, setRunToDelete] = useState<MaintenanceMlRunInfo | null>(null)
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
      }
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
            info="Indique l'état du dernier entraînement ML lancé : succès, échec, en cours ou en attente."
          />
          <MetricCard
            label="Meilleur modèle"
            value={latestMaintenanceRun?.best_model ?? maintenanceReport?.best_model ?? '-'}
            helper="sélection par PR-AUC validation"
            info="Modèle retenu parmi la régression logistique, le Random Forest et XGBoost. Il est choisi sur la PR-AUC de validation, adaptée aux pannes rares."
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
                Sélectionne un run pour consulter ses logs dans l'onglet Entraînement. Le rapport du run réussi apparaît sous la liste.
              </Typography>
              <MaintenanceRunTable runs={maintenanceRuns} selected={selectedMaintenanceRunId} onSelect={setSelectedMaintenanceRunId} onDelete={setRunToDelete} />
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
                <TableCell>{run.best_model ?? '-'}</TableCell>
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

function MaintenanceComparison({ report }: { report: MaintenanceMlReport }) {
  return (
    <Stack spacing={2}>
      <Box className="modelMetaGrid">
        <ModelMetaItem label="Dataset utilisé" value={report.gold_run_name} helper="Gold dataset source utilisé pour entraîner et évaluer les modèles de maintenance prédictive." />
        <ModelMetaItem label="Poids XGBoost" value={formatNumber(report.scale_pos_weight)} helper="Coefficient utilisé par XGBoost pour donner plus de poids aux pannes rares pendant l'entraînement." />
        <ModelMetaItem label="Random Forest" value={report.random_forest_balanced ? 'Rééquilibré' : 'Sans rééquilibrage'} helper="Indique si le Random Forest compense le déséquilibre entre pannes rares et heures normales." />
        <ModelMetaItem label="Suivi MLflow" value="Base SQLite locale" helper={`Les paramètres et métriques du run sont suivis dans MLflow : ${report.mlflow_tracking_uri}`} />
      </Box>
      <Typography variant="body2" color="text.secondary">
        Le tableau est trié par PR-AUC validation : plus cette valeur est haute, mieux le modèle retrouve les pannes rares sans se laisser tromper par la majorité des heures normales.
      </Typography>
      <TableContainer className="tableBlock comparisonTable">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Modèle</TableCell>
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
    <Box className="modelMetaItem">
      <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        {helper && <InfoTooltip title={helper} />}
      </Stack>
      <Typography variant="body2">{value}</Typography>
      {helper && <Typography variant="caption" color="text.secondary" className="metaHelper">{helper}</Typography>}
    </Box>
  )
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
    <Box className="confusionGrid">
      <MetricCard label="Vrais négatifs" value={matrix.tn.toLocaleString('fr-FR')} helper="Pas d'alerte, et aucune panne réelle ensuite." info="Cas correctement ignorés : le modèle n'a pas alerté et il n'y avait effectivement pas de panne à venir." />
      <MetricCard label="Faux positifs" value={matrix.fp.toLocaleString('fr-FR')} helper="Alerte déclenchée, mais pas de panne ensuite." info="Alertes inutiles : elles demandent une vérification humaine ou une intervention, mais aucune panne n'arrive ensuite." />
      <MetricCard label="Faux négatifs" value={matrix.fn.toLocaleString('fr-FR')} helper="Pas d'alerte, alors qu'une panne arrive ensuite." info="Pannes manquées : c'est le risque le plus important en maintenance prédictive, car la machine tombe en panne sans alerte préalable." />
      <MetricCard label="Vrais positifs" value={matrix.tp.toLocaleString('fr-FR')} helper="Alerte déclenchée, et panne réelle ensuite." info="Pannes correctement détectées : le modèle a déclenché une alerte avant une panne réelle." />
    </Box>
  )
}
