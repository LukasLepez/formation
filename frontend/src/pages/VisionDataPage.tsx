import { useEffect, useState } from 'react'
import { Alert, Box, Button, Chip, Container, LinearProgress, Paper, Stack, Typography } from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import DatasetIcon from '@mui/icons-material/Dataset'
import ImageSearchIcon from '@mui/icons-material/ImageSearch'
import RefreshIcon from '@mui/icons-material/Refresh'
import RuleIcon from '@mui/icons-material/Rule'
import { api, messageFrom } from '../lib/api'
import { formatBytes } from '../lib/format'
import type { VisionDatasetInfo, VisionDatasetPreparation, VisionDatasetSplit } from '../types'
import { MetricCard } from '../components/MetricCard'
import { SectionHeader } from '../components/SectionHeader'

export function VisionDataPage() {
  const [dataset, setDataset] = useState<VisionDatasetInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [preparing, setPreparing] = useState(false)
  const [error, setError] = useState('')

  async function refresh() {
    setLoading(true)
    setError('')
    try {
      setDataset(await api<VisionDatasetInfo>('/vision-datasets/bottle'))
    } catch (refreshError) {
      setDataset(null)
      setError(messageFrom(refreshError))
    } finally {
      setLoading(false)
    }
  }

  async function prepareDataset() {
    setPreparing(true)
    setError('')
    try {
      await api<VisionDatasetPreparation>('/vision-datasets/bottle/prepare', {
        method: 'POST',
        body: JSON.stringify({
          target_size: 256,
          validation_ratio: 0.2,
          defect_validation_ratio: 0.3,
          random_seed: 42,
          padding_value: 0,
          interpolation: 'bilinear',
        }),
      })
      await refresh()
    } catch (prepareError) {
      setError(messageFrom(prepareError))
    } finally {
      setPreparing(false)
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh()
  }, [])

  const preparation = dataset?.preparation
  const totalDefects = dataset?.test_defects.reduce((total, split) => total + split.count, 0) ?? 0
  const totalMasks = dataset?.ground_truth_masks.reduce((total, split) => total + split.count, 0) ?? 0
  const previewSplits = dataset ? [dataset.train_good, dataset.test_good, ...dataset.test_defects] : []
  const preparedTrain: VisionDatasetSplit | null = preparation ? {
    name: 'train',
    label: 'Train sain préparé',
    count: preparation.split_counts.train ?? 0,
    sample_paths: [],
  } : null
  const preparedValidation: VisionDatasetSplit | null = preparation ? {
    name: 'validation',
    label: 'Validation mixte',
    count: preparation.split_counts.validation ?? 0,
    sample_paths: [],
  } : null

  return (
    <Container maxWidth={false} className="pageShell" id="vision-data">
      <Stack spacing={2}>
        <Box className="topbar">
          <Box>
            <Typography variant="caption" color="text.secondary">Deep Learning - B6</Typography>
            <Typography variant="h4">Données images</Typography>
            <Typography color="text.secondary">
              Préparation reproductible de MVTec AD bottle pour une baseline auto-encodeur.
            </Typography>
          </Box>
          <Stack direction="row" spacing={1}>
            <Button variant="contained" onClick={() => void prepareDataset()} disabled={loading || preparing}>
              {preparing ? 'Préparation…' : 'Préparer le dataset'}
            </Button>
            <Button variant="outlined" startIcon={<RefreshIcon />} onClick={() => void refresh()} disabled={loading || preparing}>
              Actualiser
            </Button>
          </Stack>
        </Box>

        {(loading || preparing) && <LinearProgress />}
        {error && <Alert severity="error">{error}</Alert>}
        {preparation && (
          <Alert severity={preparation.leakage_free ? 'success' : 'warning'}>
            Version {preparation.version_id} prête — split déterministe, statistiques calculées sur le train et contrôle SHA-256 sans fuite.
          </Alert>
        )}

        <Box className="summaryGrid">
          <MetricCard label="Images utiles" value={dataset?.total_images.toLocaleString('fr-FR') ?? '-'} helper="train + test officiels" />
          <MetricCard label="Train sain" value={(preparation?.split_counts.train ?? dataset?.train_good.count)?.toLocaleString('fr-FR') ?? '-'} helper="apprentissage uniquement" />
          <MetricCard label="Validation mixte" value={(preparation?.split_counts.validation ?? 0).toLocaleString('fr-FR')} helper="normaux + défauts pour le réglage" />
          <MetricCard label="Défauts test" value={totalDefects.toLocaleString('fr-FR')} helper={`${dataset?.test_defects.length ?? 0} familles, jamais entraînées`} />
        </Box>

        <Paper className="panel">
          <SectionHeader title="Dataset et version" icon={<DatasetIcon fontSize="small" />} />
          {dataset ? (
            <Box className="visionDatasetGrid">
              <DatasetMeta label="Catégorie" value={dataset.name} />
              <DatasetMeta label="Dossier source" value={dataset.root_path} helper="images conservées hors Git" />
              <DatasetMeta label="Archive" value={dataset.archive_path ?? '-'} helper={dataset.archive_size_bytes ? formatBytes(dataset.archive_size_bytes) : undefined} />
              <DatasetMeta label="Version préparée" value={preparation?.version_id ?? 'Non préparée'} helper={preparation?.manifest_path} />
            </Box>
          ) : (
            <Typography color="text.secondary">Aucun dataset chargé.</Typography>
          )}
        </Paper>

        <Box className="visionWorkGrid">
          <Paper className="panel">
            <SectionHeader title="Découpage sans fuite" icon={<ImageSearchIcon fontSize="small" />} />
            {dataset && (
              <Stack spacing={1.2}>
                {preparedTrain ? <SplitRow split={preparedTrain} tone="success" /> : <SplitRow split={dataset.train_good} tone="success" />}
                {preparedValidation && <SplitRow split={preparedValidation} tone="info" />}
                <SplitRow split={dataset.test_good} tone="info" />
                {dataset.test_defects.map((split) => <SplitRow key={split.name} split={split} tone="warning" />)}
                {dataset.ground_truth_masks.map((split) => <SplitRow key={split.name} split={split} tone="default" suffix="masques" />)}
              </Stack>
            )}
          </Paper>

          <Paper className="panel">
            <SectionHeader title="Pipeline conforme au document" icon={<RuleIcon fontSize="small" />} />
            <Box className="visionChecklist">
              <ChecklistItem title="1. Split reproductible" text="20 % des normaux et 30 % de chaque famille de défauts alimentent une validation mixte. Le test final reste totalement isolé." />
              <ChecklistItem title="2. Redimensionnement sans déformation" text="Conversion RGB, conservation du ratio, padding letterbox puis mise à l'échelle des pixels dans [0, 1]." />
              <ChecklistItem title="3. Statistiques du train uniquement" text="Les moyennes et écarts-types par canal sont calculés après prétraitement, sans observer validation ni test. L'early stopping ne regarde que les normaux de validation." />
              <ChecklistItem title="4. Augmentation saine uniquement" text="Transformations faibles à la volée sur le train normal. Aucun défaut, masque, validation ou test n'est augmenté." />
              <ChecklistItem title="5. Traçabilité" text="Le manifeste relie chemins, labels, masques et SHA-256. Une même version reproduit toujours le même dataset." />
            </Box>
          </Paper>
        </Box>

        {preparation && (
          <Paper className="panel">
            <SectionHeader title="Prétraitement calculé" icon={<RuleIcon fontSize="small" />} />
            <Box className="visionDatasetGrid">
              <DatasetMeta label="Entrée réseau" value={`${preparation.target_size} × ${preparation.target_size} × 3`} helper="RGB avec padding" />
              <DatasetMeta label="Moyennes RGB" value={formatChannels(preparation.channel_mean)} helper="calculées sur train uniquement" />
              <DatasetMeta label="Écarts-types RGB" value={formatChannels(preparation.channel_std)} helper={preparation.pixel_scaling} />
              <DatasetMeta label="Contrôle de fuite" value={preparation.leakage_free ? 'Aucun doublon inter-split' : 'Doublons détectés'} helper={`seed ${preparation.random_seed}`} />
            </Box>
          </Paper>
        )}

        <Paper className="panel">
          <SectionHeader title="Échantillons après redimensionnement" icon={<ImageSearchIcon fontSize="small" />} />
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            L’aperçu utilise le même letterbox que le futur chargeur Deep Learning afin de rendre toute déformation visible.
          </Typography>
          <Box className="visionImageSections">
            {previewSplits.map((split) => <ImageStrip key={split.name} split={split} prepared={Boolean(preparation)} />)}
          </Box>
        </Paper>

        <Paper className="panel">
          <SectionHeader title="Augmentation retenue" icon={<AutoFixHighIcon fontSize="small" />} />
          <Box className="augmentationGrid">
            <AugmentationItem title="Miroir horizontal léger" text="Probabilité de 50 %, uniquement si la symétrie horizontale reste plausible pour la prise de vue." />
            <AugmentationItem title="Rotation ±5°" text="Simule un petit défaut de positionnement sans transformer ni masquer une anomalie réelle." />
            <AugmentationItem title="Translation ±3 %" text="Améliore la robustesse au cadrage tout en maintenant la taille physique apparente." />
            <AugmentationItem title="Luminosité et contraste ±10 %" text="Reproduit de faibles variations d'éclairage industriel. Le retournement vertical est explicitement interdit." />
          </Box>
        </Paper>

        <Box className="visionWorkGrid">
          <Paper className="panel">
            <SectionHeader title="Étiquettes et vérité terrain" icon={<RuleIcon fontSize="small" />} />
            <Box className="visionChecklist">
              <ChecklistItem title="Niveau image" text="good = normal ; broken_large, broken_small et contamination = anomalie." />
              <ChecklistItem title="Niveau pixel" text={`${totalMasks.toLocaleString('fr-FR')} masques sont associés aux défauts pour évaluer leur localisation.`} />
              <ChecklistItem title="Cas ambigus" text="Ils doivent être tracés puis arbitrés selon une convention définie avec un expert métier." />
            </Box>
          </Paper>
          <Paper className="panel">
            <SectionHeader title="Évaluation prévue" icon={<RuleIcon fontSize="small" />} />
            <Box className="visionChecklist">
              <ChecklistItem title="Niveau image" text="Rappel, F1, average precision, AUROC et matrice de confusion." />
              <ChecklistItem title="Niveau pixel" text="Pixel-AUROC et pixel-average-precision à partir des masques." />
              <ChecklistItem title="À éviter" text="L'accuracy ne sera pas la métrique principale : elle masque la rareté des défauts." />
            </Box>
          </Paper>
        </Box>
      </Stack>
    </Container>
  )
}

function DatasetMeta({ label, value, helper }: { label: string; value: string; helper?: string }) {
  return (
    <Box className="modelMetaItem">
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography className="pathText" title={value}>{value}</Typography>
      {helper && <Typography variant="caption" color="text.secondary">{helper}</Typography>}
    </Box>
  )
}

function SplitRow({ split, tone, suffix = 'images' }: { split: VisionDatasetSplit; tone: 'success' | 'info' | 'warning' | 'default'; suffix?: string }) {
  return (
    <Box className="visionSplitRow">
      <Box>
        <Typography sx={{ fontWeight: 700 }}>{split.label}</Typography>
        <Typography variant="caption" color="text.secondary">{split.name}</Typography>
      </Box>
      <Chip color={tone} size="small" label={`${split.count.toLocaleString('fr-FR')} ${suffix}`} />
    </Box>
  )
}

function ChecklistItem({ title, text }: { title: string; text: string }) {
  return (
    <Box className="visionChecklistItem">
      <Typography sx={{ fontWeight: 750 }}>{title}</Typography>
      <Typography variant="body2" color="text.secondary">{text}</Typography>
    </Box>
  )
}

function ImageStrip({ split, prepared }: { split: VisionDatasetSplit; prepared: boolean }) {
  if (!split.sample_paths.length) return null
  return (
    <Box className="visionImageStrip">
      <Box className="visionStripHeader">
        <Typography sx={{ fontWeight: 750 }}>{split.label}</Typography>
      </Box>
      <Box className="visionSampleGrid">
        {split.sample_paths.map((path) => {
          const endpoint = prepared ? '/api/vision-datasets/bottle/prepared-image' : '/api/vision-datasets/bottle/image'
          return (
            <Box className="visionSampleItem" key={path}>
              <img src={`${endpoint}?path=${encodeURIComponent(path)}`} alt={path} loading="lazy" />
              <Typography variant="caption" className="pathText" title={path}>{path}</Typography>
            </Box>
          )
        })}
      </Box>
    </Box>
  )
}

function AugmentationItem({ title, text }: { title: string; text: string }) {
  return (
    <Box className="augmentationItem">
      <Typography sx={{ fontWeight: 750 }}>{title}</Typography>
      <Typography variant="body2" color="text.secondary">{text}</Typography>
    </Box>
  )
}

function formatChannels(values: number[]) {
  return values.map((value) => value.toFixed(4)).join(' / ')
}
