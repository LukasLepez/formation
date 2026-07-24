import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert, Box, Button, Chip, Container, FormControl, InputLabel, LinearProgress,
  MenuItem, Paper, Select, Stack, Tab, Tabs, TextField, Typography,
} from '@mui/material'
import AutoGraphIcon from '@mui/icons-material/AutoGraph'
import MemoryIcon from '@mui/icons-material/Memory'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import ScienceIcon from '@mui/icons-material/Science'
import { api, apiText, messageFrom } from '../lib/api'
import { MetricCard } from '../components/MetricCard'
import { SectionHeader } from '../components/SectionHeader'
import { InfoTooltip } from '../components/InfoTooltip'
import type { VisionModelReport, VisionModelRunInfo } from '../types'

type FormState = {
  model_type: 'autoencoder' | 'patchcore'
  epochs: number
  batch_size: number
  learning_rate: number
  loss_name: 'mse' | 'ssim'
  latent_filters: number
  threshold_percentile: number
  early_stopping_patience: number
  patchcore_coreset_ratio: number
  patchcore_max_memory_patches: number
  patchcore_candidate_patches: number
}

const initialForm: FormState = {
  model_type: 'autoencoder', epochs: 20, batch_size: 8, learning_rate: 0.001, loss_name: 'mse',
  latent_filters: 16, threshold_percentile: 99, early_stopping_patience: 5,
  patchcore_coreset_ratio: 0.05, patchcore_max_memory_patches: 1024, patchcore_candidate_patches: 10000,
}

const patchCoreDefaults = {
  patchcore_coreset_ratio: 0.05,
  patchcore_max_memory_patches: 1024,
  patchcore_candidate_patches: 10000,
} satisfies Pick<FormState, 'patchcore_coreset_ratio' | 'patchcore_max_memory_patches' | 'patchcore_candidate_patches'>

export function VisionModelPage() {
  const [form, setForm] = useState<FormState>(initialForm)
  const [runs, setRuns] = useState<VisionModelRunInfo[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [report, setReport] = useState<VisionModelReport | null>(null)
  const [modelCard, setModelCard] = useState('')
  const [reportTab, setReportTab] = useState(0)
  const [logs, setLogs] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const selectedRun = useMemo(() => runs.find((run) => run.run_id === selectedRunId) ?? null, [runs, selectedRunId])
  const runActive = selectedRun?.status === 'queued' || selectedRun?.status === 'running'

  const loadRuns = useCallback(async (preferredRunId?: string) => {
    const nextRuns = await api<VisionModelRunInfo[]>('/vision-model-runs')
    setRuns(nextRuns)
    setSelectedRunId((current) => preferredRunId || current || nextRuns[0]?.run_id || '')
  }, [])

  const loadRunDetails = useCallback(async (run: VisionModelRunInfo | null) => {
    if (!run) { setReport(null); setModelCard(''); setLogs(''); return }
    setLogs(await apiText(`/vision-model-runs/${run.run_id}/logs`))
    if (run.status === 'success') {
      const [nextReport, nextModelCard] = await Promise.all([
        api<VisionModelReport>(`/vision-model-runs/${run.run_id}/report`),
        apiText(`/vision-model-runs/${run.run_id}/model-card`),
      ])
      setReport(nextReport)
      setModelCard(nextModelCard)
    } else {
      setReport(null)
      setModelCard('')
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadRuns().catch((loadError) => setError(messageFrom(loadError)))
  }, [loadRuns])
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadRunDetails(selectedRun).catch((loadError) => setError(messageFrom(loadError)))
  }, [loadRunDetails, selectedRun])
  useEffect(() => {
    if (!runActive) return
    const timer = window.setInterval(() => void loadRuns(selectedRunId).catch((loadError) => setError(messageFrom(loadError))), 2000)
    return () => window.clearInterval(timer)
  }, [loadRuns, runActive, selectedRunId])

  async function startRun() {
    setLoading(true); setError('')
    try {
      const created = await api<VisionModelRunInfo>('/vision-model-runs', {
        method: 'POST', body: JSON.stringify({ ...form, random_seed: 42 }),
      })
      setReportTab(0)
      await loadRuns(created.run_id)
    } catch (runError) { setError(messageFrom(runError)) } finally { setLoading(false) }
  }

  return (
    <Container maxWidth={false} className="pageShell" id="vision-model">
      <Stack spacing={2}>
        <Box className="topbar">
          <Box>
            <Typography variant="caption" color="text.secondary">Apprentissage profond — B6 partie 2</Typography>
            <Typography variant="h4">Auto-encodeur & anomalies</Typography>
            <Typography color="text.secondary">Entraînement non supervisé, seuil sain, AUROC et localisation des défauts.</Typography>
          </Box>
          <Button variant="contained" startIcon={<PlayArrowIcon />} onClick={() => void startRun()} disabled={loading || Boolean(runActive)}>Lancer l’entraînement</Button>
        </Box>
        {(loading || runActive) && <LinearProgress />}
        {error && <Alert severity="error">{error}</Alert>}
        {selectedRun?.status === 'failed' && <Alert severity="error">{selectedRun.error ?? 'Le run a échoué.'}</Alert>}
        {runActive && <Alert severity="info">Le run s’exécute en arrière-plan. Actualisation toutes les 2 secondes.</Alert>}

        <Paper className="panel">
          <SectionHeader title="Configuration pédagogique" icon={<ScienceIcon fontSize="small" />} />
          <Box className="visionModelSettings">
            <FormControl size="small"><InputLabel>Modèle</InputLabel><Select label="Modèle" value={form.model_type} onChange={(event) => { const modelType = event.target.value as FormState['model_type']; setForm({ ...form, model_type: modelType, ...(modelType === 'patchcore' ? patchCoreDefaults : {}) }) }}><MenuItem value="autoencoder">Auto-encodeur</MenuItem><MenuItem value="patchcore">PatchCore (ResNet-18)</MenuItem></Select></FormControl>
            <NumberSetting label="Taille du lot" value={form.batch_size} min={1} max={64} onChange={(value) => setForm({ ...form, batch_size: value })} />
            {form.model_type === 'autoencoder' ? <><NumberSetting label="Époques max." value={form.epochs} min={1} max={200} onChange={(value) => setForm({ ...form, epochs: value })} /><TextField label="Taux d’apprentissage" type="number" size="small" value={form.learning_rate} slotProps={{ htmlInput: { step: 0.0001, min: 0.000001 } }} onChange={(event) => setForm({ ...form, learning_rate: Number(event.target.value) })} /><FormControl size="small"><InputLabel>Perte</InputLabel><Select label="Perte" value={form.loss_name} onChange={(event) => setForm({ ...form, loss_name: event.target.value as 'mse' | 'ssim' })}><MenuItem value="mse">MSE</MenuItem><MenuItem value="ssim">SSIM</MenuItem></Select></FormControl><NumberSetting label="Filtres latents" value={form.latent_filters} min={1} max={128} onChange={(value) => setForm({ ...form, latent_filters: value })} /><NumberSetting label="Patience" value={form.early_stopping_patience} min={1} max={30} onChange={(value) => setForm({ ...form, early_stopping_patience: value })} /></> : <><TextField label="Ratio coreset" type="number" size="small" value={form.patchcore_coreset_ratio} slotProps={{ htmlInput: { step: 0.01, min: 0.001, max: 1 } }} onChange={(event) => setForm({ ...form, patchcore_coreset_ratio: Number(event.target.value) })} /><NumberSetting label="Patchs mémoire max." value={form.patchcore_max_memory_patches} min={32} max={20000} onChange={(value) => setForm({ ...form, patchcore_max_memory_patches: value })} /><NumberSetting label="Patchs candidats" value={form.patchcore_candidate_patches} min={512} max={100000} onChange={(value) => setForm({ ...form, patchcore_candidate_patches: value })} /></>}
            <NumberSetting label="Centile seuil" value={form.threshold_percentile} min={50} max={100} onChange={(value) => setForm({ ...form, threshold_percentile: value })} />
          </Box>
          {form.model_type === 'patchcore' && <Box className="visionPatchcoreHelp">
            <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap', alignItems: 'center' }}>
              <Typography variant="body2">Valeurs recommandées pour démarrer</Typography>
              <Button size="small" variant="outlined" onClick={() => setForm({ ...form, ...patchCoreDefaults })}>Réinitialiser : 5 % · 1 024 · 10 000</Button>
            </Stack>
            <Box className="visionPatchcoreDefinitions">
              <PatchCoreDefinition label="Ratio coreset" value="5 %" detail="Part théorique des patchs sains à conserver. Un ratio plus élevé couvre davantage de cas normaux, mais ralentit fortement la sélection et les comparaisons." />
              <PatchCoreDefinition label="Patchs mémoire max." value="1 024" detail="Limite finale de la banque de patchs sains. Plus cette valeur est haute, plus le modèle est précis potentiellement, mais plus l'inférence est lente et gourmande en mémoire." />
              <PatchCoreDefinition label="Patchs candidats" value="10 000" detail="Nombre de patchs examinés avant de sélectionner le coreset. Plus il est élevé, plus la banque représente la variété des images saines, au prix d'un entraînement beaucoup plus long." />
              <PatchCoreDefinition label="Taille du lot" value="8" detail="Nombre d'images dont les caractéristiques ResNet-18 sont extraites simultanément. Augmenter cette valeur accélère le run si la mémoire disponible le permet." />
              <PatchCoreDefinition label="Centile seuil" value="99" detail="Seuil calculé seulement avec les images saines de validation. À 99, seules les images dont le score dépasse les 99 % de référence déclenchent une alerte." />
            </Box>
          </Box>}
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5 }}>{form.model_type === 'patchcore' ? 'PatchCore télécharge au premier lancement les poids ResNet-18 ImageNet puis constitue une banque de patchs sains. Augmenter la mémoire et les candidats améliore potentiellement le résultat, mais augmente fortement la durée.' : 'Le seuil est calibré exclusivement sur les images saines de validation. Les défauts de validation ne servent ni à l’apprentissage ni au réglage.'}</Typography>
        </Paper>

        <Paper className="panel">
          <SectionHeader title="Exécutions vision" icon={<MemoryIcon fontSize="small" />} action={runs.length ? <FormControl size="small" sx={{ minWidth: 310 }}><InputLabel>Exécution affichée</InputLabel><Select label="Exécution affichée" value={selectedRunId} onChange={(event) => { setSelectedRunId(event.target.value); setReportTab(0) }}>{runs.map((run) => <MenuItem key={run.run_id} value={run.run_id} >{run.run_id} · {statusLabel(run.status)}</MenuItem>)}</Select></FormControl> : undefined} />
          {selectedRun ? <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap' }}><Chip label={statusLabel(selectedRun.status)} color={statusColor(selectedRun.status)} size="small" /><Chip label={selectedRun.model_type === 'patchcore' ? 'PatchCore' : 'Auto-encodeur'} variant="outlined" size="small" /><Chip label={selectedRun.dataset_version} variant="outlined" size="small" /><Chip label={`${selectedRun.epochs} époques · lot de ${selectedRun.batch_size}`} variant="outlined" size="small" /><Chip label={`perte ${selectedRun.loss_name.toUpperCase()}`} variant="outlined" size="small" /></Stack> : <Alert severity="info">Aucune exécution. Préparez d’abord le jeu de données, puis lancez l’entraînement.</Alert>}
        </Paper>

        {report && (
          <>
            <Paper className="tabsPanel">
              <Tabs value={reportTab} onChange={(_, value) => setReportTab(value)} variant="scrollable" scrollButtons="auto">
                <Tab label="Résultats du modèle" />
                <Tab label="B8 · Model card" />
              </Tabs>
            </Paper>
            {reportTab === 0 && <VisionReport report={report} />}
            {reportTab === 1 && (
              <Paper className="panel">
                <SectionHeader
                  title="Model card Hugging Face — modèle vision"
                  action={
                    <Button
                      variant="outlined"
                      href={`/api/vision-model-runs/${report.run_id}/model-card`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Ouvrir le Markdown
                    </Button>
                  }
                />
                <Alert severity="info" sx={{ mb: 2 }}>
                  Fiche B8 générée depuis ce run : usages, hors-périmètre, limites, seuil,
                  métriques image/pixel, CodeCarbon, version, auteurs et contact.
                </Alert>
                <Box
                  component="pre"
                  sx={{
                    m: 0, p: 2, maxHeight: 720, overflow: 'auto', whiteSpace: 'pre-wrap',
                    overflowWrap: 'anywhere', borderRadius: 2, bgcolor: 'action.hover',
                    fontFamily: 'monospace', fontSize: '0.78rem', lineHeight: 1.55,
                  }}
                >
                  {modelCard || 'Model card indisponible.'}
                </Box>
              </Paper>
            )}
          </>
        )}
        {selectedRun && <Paper className="panel"><SectionHeader title="Journal d’exécution" /><Box className="visionLogBox">{logs || 'Le journal est encore vide.'}</Box></Paper>}
      </Stack>
    </Container>
  )
}

function VisionReport({ report }: { report: VisionModelReport }) {
  const image = report.metrics.image
  const pixel = report.metrics.pixel
  const isPatchCore = report.model_type === 'patchcore'
  const artifactUrl = (path: string) => `/api/vision-model-runs/${report.run_id}/artifacts/${path}`
  return <>
    <Box className="summaryGrid">
      <MetricCard label="AUROC image" value={formatMetric(image.auroc)} helper="séparation saine / défaut" info="Mesure la capacité du modèle à classer une image défectueuse avec un score plus élevé qu'une image saine, tous seuils confondus. 1 est excellent, 0,5 correspond au hasard." />
      <MetricCard label="AUROC pixel" value={formatMetric(pixel.auroc)} helper="localisation comparée aux masques" info="Mesure si les zones mises en évidence par la carte d'anomalie correspondent aux pixels réellement annotés comme défectueux. 1 est excellent, 0,5 correspond au hasard." />
      <MetricCard label="Rappel" value={formatPercent(image.recall)} helper={`${image.confusion_matrix.fn} défaut(s) manqué(s)`} info="Part des images réellement défectueuses détectées par le modèle au seuil choisi. Un rappel élevé limite les défauts manqués, mais peut créer davantage de fausses alertes." />
      <MetricCard label={isPatchCore ? 'Seuil distance' : 'Seuil MSE'} value={report.threshold.value.toExponential(3)} helper={`centile ${report.threshold.percentile} sur ${report.threshold.calibration_images} saines`} info={isPatchCore ? "Distance minimale aux patchs sains : une image dont le score est supérieur au seuil déclenche une alerte. Ce seuil est calculé uniquement sur les images saines de validation." : "Erreur moyenne de reconstruction : une image dont la MSE est supérieure au seuil déclenche une alerte. Ce seuil est calculé uniquement sur les images saines de validation."} />
      <MetricCard label="Empreinte entraînement" value={report.carbon?.emissions_gco2eq != null ? `${report.carbon.emissions_gco2eq.toFixed(2)} gCO₂e` : '—'} helper={report.carbon?.available ? `${report.carbon.energy_kwh?.toFixed(5) ?? '—'} kWh · ${report.carbon.duration_seconds?.toFixed(1) ?? '—'} s` : report.carbon?.reason ?? 'CodeCarbon'} info="Mesure CodeCarbon de l’entraînement avec le mix électrique français configuré." />
    </Box>
    <Box className="visionWorkGrid">
      <Paper className="panel"><SectionHeader title={isPatchCore ? 'PatchCore : ResNet-18 et banque de patchs' : 'Architecture : 3 convolutions / 3 déconvolutions'} icon={<MemoryIcon fontSize="small" />} /><Box className="visionChecklist"><ResultLine label="Entrée" value={report.architecture.input_shape.join(' × ')} /><ResultLine label={isPatchCore ? 'Carte de patchs' : 'Espace latent'} value={report.architecture.latent_shape.join(' × ')} />{!isPatchCore && <><ResultLine label="Compression" value={`× ${report.architecture.compression_ratio.toFixed(2)}`} /><ResultLine label="Paramètres" value={report.architecture.parameter_count.toLocaleString('fr-FR')} /><ResultLine label="Meilleure époque" value={`${report.training.best_epoch} / ${report.training.epochs_completed}`} /></>}</Box><Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>{report.architecture.comment}</Typography><Button size="small" href={artifactUrl(report.architecture.summary_artifact)} target="_blank" sx={{ mt: 1 }}>{isPatchCore ? 'Voir le résumé PatchCore' : 'Voir model.summary()'}</Button></Paper>
      <Paper className="panel"><SectionHeader title="Métriques au seuil" icon={<AutoGraphIcon fontSize="small" />} /><Box className="visionChecklist"><ResultLine label="Précision" value={formatPercent(image.precision)} /><ResultLine label="Rappel" value={formatPercent(image.recall)} /><ResultLine label="F1" value={formatMetric(image.f1)} /><ResultLine label="Précision moyenne — image" value={formatMetric(image.average_precision)} /><ResultLine label="Précision moyenne — pixel" value={formatMetric(pixel.average_precision)} /><ResultLine label="MLflow" value={report.mlflow_run_id ? report.mlflow_run_id.slice(0, 12) : 'suivi indisponible'} /></Box></Paper>
    </Box>
    <Paper className="panel"><SectionHeader title="Courbes et diagnostics" /><Box className="visionArtifactGrid">{(['learning_curve', 'score_histogram', 'confusion_matrix', 'reconstructions'] as const).map((key) => <a key={key} href={artifactUrl(report.artifacts[key])} target="_blank" rel="noreferrer"><img src={artifactUrl(report.artifacts[key])} alt={isPatchCore && key === 'reconstructions' ? 'Images et cartes PatchCore' : key} /></a>)}</Box></Paper>
    <Paper className="panel"><SectionHeader title="Heatmaps sur défauts réels" /><Box className="visionHeatmapGrid">{report.samples.map((sample) => <Box key={sample.heatmap_artifact} className="visionHeatmapCard"><img src={artifactUrl(sample.heatmap_artifact)} alt={`Heatmap ${sample.label}`} /><Stack direction="row" spacing={1} sx={{ justifyContent: 'space-between' }}><Typography variant="caption">{sample.label}</Typography><Chip size="small" color={sample.predicted_anomaly ? 'warning' : 'default'} label={sample.predicted_anomaly ? 'détecté' : 'raté'} /></Stack></Box>)}</Box></Paper>
    <Alert severity="info" icon={<ScienceIcon />}>{report.critical_analysis}</Alert>
  </>
}

function NumberSetting({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (value: number) => void }) {
  return <TextField label={label} type="number" size="small" value={value} slotProps={{ htmlInput: { min, max } }} onChange={(event) => onChange(Number(event.target.value))} />
}
function PatchCoreDefinition({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <Box className="visionPatchcoreDefinition"><Typography variant="caption" color="text.secondary">{label}</Typography><Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}><Typography variant="body2">Défaut : {value}</Typography><InfoTooltip title={detail} /></Stack></Box>
}
function ResultLine({ label, value }: { label: string; value: string }) { return <Box className="visionResultLine"><Typography variant="body2" color="text.secondary">{label}</Typography><Typography variant="body2">{value}</Typography></Box> }
function formatMetric(value?: number | null) { return value == null ? '—' : value.toFixed(3) }
function formatPercent(value: number) { return new Intl.NumberFormat('fr-FR', { style: 'percent', maximumFractionDigits: 1 }).format(value) }
function statusLabel(status: VisionModelRunInfo['status']) { if (status === 'success') return 'terminée'; if (status === 'failed') return 'échouée'; if (status === 'running') return 'en cours'; return 'en attente' }
function statusColor(status: VisionModelRunInfo['status']): 'default' | 'info' | 'success' | 'error' { if (status === 'success') return 'success'; if (status === 'failed') return 'error'; if (status === 'running') return 'info'; return 'default' }
