import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  Container,
  CssBaseline,
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
  ThemeProvider,
  Tooltip,
  Typography,
  createTheme,
} from '@mui/material'
import type { SelectChangeEvent } from '@mui/material'
import AutoGraphIcon from '@mui/icons-material/AutoGraph'
import DatasetIcon from '@mui/icons-material/Dataset'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import RefreshIcon from '@mui/icons-material/Refresh'
import TerminalIcon from '@mui/icons-material/Terminal'
import './App.css'

type RunStatus = 'queued' | 'running' | 'success' | 'failed'
type LayerName = 'all' | 'bronze' | 'silver' | 'gold'
type GraphSourceLayer = 'bronze' | 'silver'

type RunInfo = {
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

type GraphRunInfo = {
  run_id: string
  status: RunStatus
  source_layer: GraphSourceLayer
  created_at: string
  graph_count?: number | null
  incident_rows?: number | null
  telemetry_rows?: number | null
  error?: string | null
}

type TablePreview = {
  layer: string
  table: string
  columns: string[]
  rows: Record<string, unknown>[]
  total_rows: number
  limit: number
  offset: number
}

const CSV_PAGE_SIZE = 500
const TABLE_PAGE_SIZE = 500

type GoldCsvInfo = {
  run_name: string
  csv_path: string
  created_at?: string | null
  rows?: number | null
  columns?: number | null
  size_bytes: number
}

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#93c5fd' },
    secondary: { main: '#a7f3d0' },
    background: { default: '#0b0d12', paper: '#141821' },
    text: { primary: '#e5e7eb', secondary: '#9ca3af' },
    success: { main: '#22c55e' },
    error: { main: '#f87171' },
    warning: { main: '#fbbf24' },
  },
  shape: { borderRadius: 6 },
  typography: {
    fontFamily: '"Inter", "Segoe UI", system-ui, sans-serif',
    h4: { fontWeight: 750, letterSpacing: 0 },
    h6: { fontWeight: 750, letterSpacing: 0 },
    button: { textTransform: 'none', fontWeight: 700 },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          border: '1px solid #252b36',
          boxShadow: 'none',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: { borderColor: '#252b36', whiteSpace: 'nowrap' },
        head: { fontWeight: 750, color: '#d1d5db', backgroundColor: '#10141c' },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: { boxShadow: 'none' },
      },
    },
  },
})

const statusColor: Record<RunStatus, 'default' | 'success' | 'error' | 'warning' | 'info'> = {
  queued: 'info',
  running: 'warning',
  success: 'success',
  failed: 'error',
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<T>
}

async function apiText(path: string): Promise<string> {
  const response = await fetch(`/api${path}`)
  if (!response.ok) throw new Error(await response.text())
  return response.text()
}

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [health, setHealth] = useState<'ok' | 'ko' | 'loading'>('loading')
  const [runs, setRuns] = useState<RunInfo[]>([])
  const [graphRuns, setGraphRuns] = useState<GraphRunInfo[]>([])
  const [datasets, setDatasets] = useState<Record<string, string[]>>({})
  const [goldCsvs, setGoldCsvs] = useState<GoldCsvInfo[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [selectedGraphRunId, setSelectedGraphRunId] = useState('')
  const [selectedGoldCsv, setSelectedGoldCsv] = useState('')
  const [logs, setLogs] = useState('')
  const [graphLogs, setGraphLogs] = useState('')
  const [layer, setLayer] = useState<LayerName>('all')
  const [persistDb, setPersistDb] = useState(true)
  const [autoDocker, setAutoDocker] = useState(true)
  const [graphLayer, setGraphLayer] = useState<GraphSourceLayer>('silver')
  const [dataLayer, setDataLayer] = useState('gold')
  const [dataTable, setDataTable] = useState('gold_dataset')
  const [preview, setPreview] = useState<TablePreview | null>(null)
  const [goldPreview, setGoldPreview] = useState<TablePreview | null>(null)
  const [previewLoadingMore, setPreviewLoadingMore] = useState(false)
  const [goldLoadingMore, setGoldLoadingMore] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const latestRun = runs[0]

  async function refresh() {
    setError('')
    try {
      const [healthData, runData, graphData, datasetData, csvData] = await Promise.all([
        api<{ status: string }>('/health'),
        api<RunInfo[]>('/runs'),
        api<GraphRunInfo[]>('/graph-runs'),
        api<Record<string, string[]>>('/datasets'),
        api<GoldCsvInfo[]>('/gold-csvs'),
      ])
      setHealth(healthData.status === 'ok' ? 'ok' : 'ko')
      setRuns(runData)
      setGraphRuns(graphData)
      setDatasets(datasetData)
      setGoldCsvs(csvData)
      if (!selectedRunId && runData[0]) setSelectedRunId(runData[0].run_id)
      if (!selectedGraphRunId && graphData[0]) setSelectedGraphRunId(graphData[0].run_id)
      if (!selectedGoldCsv && csvData[0]) setSelectedGoldCsv(csvData[0].run_name)
    } catch (refreshError) {
      setHealth('ko')
      setError(messageFrom(refreshError))
    }
  }

  async function launchRun() {
    setBusy(true)
    setError('')
    try {
      const run = await api<RunInfo>('/runs', {
        method: 'POST',
        body: JSON.stringify({ layer, persist_db: persistDb, auto_start_docker: autoDocker, log_level: 'INFO' }),
      })
      setSelectedRunId(run.run_id)
      setActiveTab(0)
      await refresh()
    } catch (runError) {
      setError(messageFrom(runError))
    } finally {
      setBusy(false)
    }
  }

  async function launchGraphRun() {
    setBusy(true)
    setError('')
    try {
      const run = await api<GraphRunInfo>('/graph-runs', {
        method: 'POST',
        body: JSON.stringify({ source_layer: graphLayer, log_level: 'INFO' }),
      })
      setSelectedGraphRunId(run.run_id)
      setActiveTab(3)
      await refresh()
    } catch (graphError) {
      setError(messageFrom(graphError))
    } finally {
      setBusy(false)
    }
  }

  async function loadPreview(nextLayer = dataLayer, nextTable = dataTable, offset = 0, append = false) {
    setError('')
    setPreviewLoadingMore(true)
    try {
      const nextPreview = await api<TablePreview>(`/datasets/${nextLayer}/${nextTable}?limit=${TABLE_PAGE_SIZE}&offset=${offset}`)
      setPreview((current) => (
        append && current && current.layer === nextPreview.layer && current.table === nextPreview.table
          ? { ...nextPreview, rows: [...current.rows, ...nextPreview.rows], offset: 0, limit: current.rows.length + nextPreview.rows.length }
          : nextPreview
      ))
    } catch (previewError) {
      setPreview(null)
      setError(messageFrom(previewError))
    } finally {
      setPreviewLoadingMore(false)
    }
  }

  async function loadGoldCsv(runName = selectedGoldCsv, offset = 0, append = false) {
    if (!runName) return
    setError('')
    setGoldLoadingMore(true)
    try {
      const nextPreview = await api<TablePreview>(`/gold-csvs/${encodeURIComponent(runName)}?limit=${CSV_PAGE_SIZE}&offset=${offset}`)
      setGoldPreview((current) => (
        append && current
          ? { ...nextPreview, rows: [...current.rows, ...nextPreview.rows], offset: 0, limit: current.rows.length + nextPreview.rows.length }
          : nextPreview
      ))
    } catch (csvError) {
      setGoldPreview(null)
      setError(messageFrom(csvError))
    } finally {
      setGoldLoadingMore(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  useEffect(() => {
    const tables = datasets[dataLayer] ?? []
    if (tables.length && !tables.includes(dataTable)) {
      setDataTable(tables[0])
      void loadPreview(dataLayer, tables[0])
    }
  }, [datasets, dataLayer])

  useEffect(() => {
    if (Object.keys(datasets).length) void loadPreview()
  }, [dataLayer, dataTable])

  useEffect(() => {
    if (selectedGoldCsv) void loadGoldCsv(selectedGoldCsv, 0, false)
  }, [selectedGoldCsv])

  useEffect(() => {
    if (!selectedRunId) return
    let active = true
    async function poll() {
      try {
        const text = await apiText(`/runs/${selectedRunId}/logs/raw`)
        if (active) setLogs(text)
      } catch {
        if (active) setLogs('')
      }
    }
    void poll()
    const interval = window.setInterval(() => {
      void poll()
      void refresh()
    }, 2500)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [selectedRunId])

  useEffect(() => {
    if (!selectedGraphRunId) return
    let active = true
    async function poll() {
      try {
        const text = await apiText(`/graph-runs/${selectedGraphRunId}/logs/raw`)
        if (active) setGraphLogs(text)
      } catch {
        if (active) setGraphLogs('')
      }
    }
    void poll()
    const interval = window.setInterval(() => void poll(), 2500)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [selectedGraphRunId])

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box className="appFrame">
        <SideNavigation health={health} onRefresh={() => void refresh()} />
        <Container maxWidth={false} className="ingestionShell" id="ingestion">
          <Stack spacing={2}>
            <Box className="topbar">
              <Box>
                <Typography variant="caption" color="text.secondary">InduSense</Typography>
                <Typography variant="h4">Ingestion</Typography>
                <Typography color="text.secondary">Pipeline, Docker, logs, données et CSV Gold au même endroit.</Typography>
              </Box>
            </Box>

            {busy && <LinearProgress />}
            {error && <Alert severity="error">{error}</Alert>}

            <Box className="summaryGrid">
              <MetricCard label="Dernier run" value={latestRun?.status ?? '-'} helper={latestRun?.run_id ?? 'Aucun run'} />
              <MetricCard label="Lignes Gold" value={latestRun?.rows?.toLocaleString('fr-FR') ?? '-'} helper={`${latestRun?.columns ?? '-'} colonnes`} />
              <MetricCard label="CSV Gold" value={goldCsvs.length.toLocaleString('fr-FR')} helper="versions générées" />
              <MetricCard label="Graphes" value={graphRuns[0]?.graph_count?.toLocaleString('fr-FR') ?? '-'} helper={graphRuns[0]?.source_layer ?? 'Aucun rapport'} />
            </Box>

            <Paper className="tabsPanel">
              <Tabs value={activeTab} onChange={(_, value) => setActiveTab(value)} variant="scrollable" scrollButtons="auto">
                <Tab label="Pipeline" />
                <Tab label="Base de données" />
                <Tab label="CSV Gold" />
                <Tab label="Graphes" />
              </Tabs>
            </Paper>

            {activeTab === 0 && (
              <Paper className="panel">
                <SectionHeader title="Commandes" />
                <Stack direction={{ xs: 'column', lg: 'row' }} spacing={2} sx={{ alignItems: { lg: 'center' } }}>
                  <FormControl size="small" sx={{ minWidth: 150 }}>
                    <InputLabel>Couche</InputLabel>
                    <Select value={layer} label="Couche" onChange={(event: SelectChangeEvent) => setLayer(event.target.value as LayerName)}>
                      <MenuItem value="all">Toutes</MenuItem>
                      <MenuItem value="bronze">Bronze</MenuItem>
                      <MenuItem value="silver">Silver</MenuItem>
                      <MenuItem value="gold">Gold</MenuItem>
                    </Select>
                  </FormControl>
                  <FormControlLabel control={<Switch checked={persistDb} onChange={(_, checked) => setPersistDb(checked)} />} label="PostgreSQL" />
                  <FormControlLabel control={<Switch checked={autoDocker} onChange={(_, checked) => setAutoDocker(checked)} />} label="Docker" />
                  <Button variant="contained" startIcon={<PlayArrowIcon />} onClick={() => void launchRun()} disabled={busy}>
                    Lancer
                  </Button>
                  <FormControl size="small" sx={{ minWidth: 150 }}>
                    <InputLabel>Graphes</InputLabel>
                    <Select value={graphLayer} label="Graphes" onChange={(event: SelectChangeEvent) => setGraphLayer(event.target.value as GraphSourceLayer)}>
                      <MenuItem value="bronze">Bronze</MenuItem>
                      <MenuItem value="silver">Silver</MenuItem>
                    </Select>
                  </FormControl>
                  <Button variant="outlined" startIcon={<AutoGraphIcon />} onClick={() => void launchGraphRun()} disabled={busy}>
                    Générer les graphes
                  </Button>
                </Stack>
              </Paper>
            )}

            {activeTab === 0 && (
              <Box className="workGrid">
                <Paper className="panel">
                  <SectionHeader title="Runs pipeline" icon={<TerminalIcon fontSize="small" />} action={<RunPicker runs={runs} value={selectedRunId} onChange={setSelectedRunId} />} />
                  <RunTable runs={runs} selectedRunId={selectedRunId} onSelect={setSelectedRunId} />
                </Paper>
                <LogPanel title="Logs pipeline" text={logs} />
              </Box>
            )}

            {activeTab === 1 && (
              <Paper className="panel">
                <SectionHeader title="Base de données" icon={<DatasetIcon fontSize="small" />} />
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 2 }}>
                  <FormControl size="small" sx={{ minWidth: 160 }}>
                    <InputLabel>Couche</InputLabel>
                    <Select value={dataLayer} label="Couche" onChange={(event: SelectChangeEvent) => setDataLayer(event.target.value)}>
                      {Object.keys(datasets).map((name) => <MenuItem key={name} value={name}>{name}</MenuItem>)}
                    </Select>
                  </FormControl>
                  <FormControl size="small" sx={{ minWidth: 220 }}>
                    <InputLabel>Table</InputLabel>
                    <Select value={dataTable} label="Table" onChange={(event: SelectChangeEvent) => setDataTable(event.target.value)}>
                      {(datasets[dataLayer] ?? []).map((name) => <MenuItem key={name} value={name}>{name}</MenuItem>)}
                    </Select>
                  </FormControl>
                  <Button variant="outlined" startIcon={<RefreshIcon />} onClick={() => void loadPreview(dataLayer, dataTable, 0, false)}>Recharger</Button>
                </Stack>
                {preview && (
                  <Stack spacing={2}>
                    <DataPreview preview={preview} loadedRows={preview.rows.length} />
                    <Button
                      variant="outlined"
                      onClick={() => void loadPreview(dataLayer, dataTable, preview.rows.length, true)}
                      disabled={previewLoadingMore || preview.rows.length >= preview.total_rows}
                    >
                      {preview.rows.length >= preview.total_rows ? 'Toutes les lignes sont chargées' : `Charger ${TABLE_PAGE_SIZE.toLocaleString('fr-FR')} lignes de plus`}
                    </Button>
                  </Stack>
                )}
              </Paper>
            )}

            {activeTab === 2 && (
              <Paper className="panel">
                <SectionHeader title="CSV Gold générés" icon={<DatasetIcon fontSize="small" />} />
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 2, alignItems: { md: 'center' } }}>
                  <FormControl size="small" sx={{ minWidth: 340 }}>
                    <InputLabel>CSV Gold</InputLabel>
                    <Select value={selectedGoldCsv} label="CSV Gold" onChange={(event: SelectChangeEvent) => setSelectedGoldCsv(event.target.value)}>
                      {goldCsvs.map((csv) => <MenuItem key={csv.run_name} value={csv.run_name}>{csv.run_name}</MenuItem>)}
                    </Select>
                  </FormControl>
                  <Button variant="outlined" startIcon={<RefreshIcon />} onClick={() => void loadGoldCsv(selectedGoldCsv, 0, false)}>Recharger</Button>
                  <Button
                    component="a"
                    href={selectedGoldCsv ? `/api/gold-csvs/${encodeURIComponent(selectedGoldCsv)}/download` : undefined}
                    variant="contained"
                    disabled={!selectedGoldCsv}
                  >
                    Télécharger le CSV
                  </Button>
                  <Typography variant="body2" color="text.secondary" className="pathText">
                    {goldCsvs.find((csv) => csv.run_name === selectedGoldCsv)?.csv_path ?? 'Aucun CSV Gold'}
                  </Typography>
                </Stack>
                <GoldCsvTable csvs={goldCsvs} selected={selectedGoldCsv} onSelect={setSelectedGoldCsv} />
                {goldPreview && (
                  <Stack spacing={2} sx={{ mt: 2 }}>
                    <DataPreview preview={goldPreview} loadedRows={goldPreview.rows.length} />
                    <Button
                      variant="outlined"
                      onClick={() => void loadGoldCsv(selectedGoldCsv, goldPreview.rows.length, true)}
                      disabled={goldLoadingMore || goldPreview.rows.length >= goldPreview.total_rows}
                    >
                      {goldPreview.rows.length >= goldPreview.total_rows ? 'Toutes les lignes sont chargées' : `Charger ${CSV_PAGE_SIZE.toLocaleString('fr-FR')} lignes de plus`}
                    </Button>
                  </Stack>
                )}
              </Paper>
            )}

            {activeTab === 3 && (
              <Box className="workGrid">
                <Paper className="panel">
                  <SectionHeader title="Graphes" icon={<AutoGraphIcon fontSize="small" />} action={<GraphRunPicker runs={graphRuns} value={selectedGraphRunId} onChange={setSelectedGraphRunId} />} />
                  <GraphRunTable runs={graphRuns} selected={selectedGraphRunId} onSelect={setSelectedGraphRunId} />
                </Paper>
                <LogPanel title="Logs graphes" text={graphLogs} />
              </Box>
            )}
          </Stack>
        </Container>
      </Box>
    </ThemeProvider>
  )
}

function SideNavigation({ health, onRefresh }: { health: 'ok' | 'ko' | 'loading'; onRefresh: () => void }) {
  return (
    <Box component="nav" className="sideNav">
      <Box className="brandBlock">
        <Box>
          <Typography variant="subtitle1">InduSense</Typography>
          <Typography variant="caption" color="text.secondary">Ingestion</Typography>
        </Box>
      </Box>
      <Box className="navLinks">
        <Box component="a" href="#ingestion" className="navLink">
          <span>Ingestion</span>
        </Box>
      </Box>
      <Box className="navStatus">
        <Chip
          label={health === 'ok' ? 'API connectée' : health === 'loading' ? 'Connexion API' : 'API indisponible'}
          color={health === 'ok' ? 'success' : 'error'}
          size="small"
        />
        <Tooltip title="Rafraîchir">
          <IconButton onClick={onRefresh} size="small">
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  )
}

function MetricCard({ label, value, helper }: { label: string; value: string; helper: string }) {
  return (
    <Paper className="metricCard">
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography variant="h6">{value}</Typography>
      <Typography variant="caption" color="text.secondary">{helper}</Typography>
    </Paper>
  )
}

function SectionHeader({ title, action, icon }: { title: string; action?: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 2, alignItems: { md: 'center' }, justifyContent: 'space-between' }}>
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
        {icon}
        <Typography variant="h6">{title}</Typography>
      </Stack>
      {action}
    </Stack>
  )
}

function RunPicker({ runs, value, onChange }: { runs: RunInfo[]; value: string; onChange: (value: string) => void }) {
  return (
    <FormControl size="small" sx={{ minWidth: 260 }}>
      <InputLabel>Run</InputLabel>
      <Select value={value} label="Run" onChange={(event: SelectChangeEvent) => onChange(event.target.value)}>
        {runs.map((run) => <MenuItem key={run.run_id} value={run.run_id}>{run.run_id}</MenuItem>)}
      </Select>
    </FormControl>
  )
}

function GraphRunPicker({ runs, value, onChange }: { runs: GraphRunInfo[]; value: string; onChange: (value: string) => void }) {
  return (
    <FormControl size="small" sx={{ minWidth: 260 }}>
      <InputLabel>Run graphes</InputLabel>
      <Select value={value} label="Run graphes" onChange={(event: SelectChangeEvent) => onChange(event.target.value)}>
        {runs.map((run) => <MenuItem key={run.run_id} value={run.run_id}>{run.run_id}</MenuItem>)}
      </Select>
    </FormControl>
  )
}

function RunTable({ runs, selectedRunId, onSelect }: { runs: RunInfo[]; selectedRunId: string; onSelect: (id: string) => void }) {
  return (
    <TableContainer sx={{ maxHeight: 430 }}>
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Run</TableCell>
            <TableCell>Statut</TableCell>
            <TableCell>Couche</TableCell>
            <TableCell>Lignes</TableCell>
            <TableCell>Colonnes</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.run_id} hover selected={run.run_id === selectedRunId} onClick={() => onSelect(run.run_id)} sx={{ cursor: 'pointer' }}>
              <TableCell>{run.run_id}</TableCell>
              <TableCell><Chip label={run.status} color={statusColor[run.status]} size="small" /></TableCell>
              <TableCell>{run.layer}</TableCell>
              <TableCell>{run.rows?.toLocaleString('fr-FR') ?? '-'}</TableCell>
              <TableCell>{run.columns?.toLocaleString('fr-FR') ?? '-'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

function GraphRunTable({ runs, selected, onSelect }: { runs: GraphRunInfo[]; selected: string; onSelect: (id: string) => void }) {
  return (
    <TableContainer sx={{ maxHeight: 430 }}>
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Run</TableCell>
            <TableCell>Statut</TableCell>
            <TableCell>Source</TableCell>
            <TableCell>Graphes</TableCell>
            <TableCell>Télémétrie</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.run_id} hover selected={run.run_id === selected} onClick={() => onSelect(run.run_id)} sx={{ cursor: 'pointer' }}>
              <TableCell>{run.run_id}</TableCell>
              <TableCell><Chip label={run.status} color={statusColor[run.status]} size="small" /></TableCell>
              <TableCell>{run.source_layer}</TableCell>
              <TableCell>{run.graph_count?.toLocaleString('fr-FR') ?? '-'}</TableCell>
              <TableCell>{run.telemetry_rows?.toLocaleString('fr-FR') ?? '-'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

function GoldCsvTable({ csvs, selected, onSelect }: { csvs: GoldCsvInfo[]; selected: string; onSelect: (run: string) => void }) {
  return (
    <TableContainer className="tableBlock" sx={{ maxHeight: 280 }}>
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Run CSV</TableCell>
            <TableCell>Lignes</TableCell>
            <TableCell>Colonnes</TableCell>
            <TableCell>Taille</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {csvs.map((csv) => (
            <TableRow key={csv.run_name} hover selected={csv.run_name === selected} onClick={() => onSelect(csv.run_name)} sx={{ cursor: 'pointer' }}>
              <TableCell>{csv.run_name}</TableCell>
              <TableCell>{csv.rows?.toLocaleString('fr-FR') ?? '-'}</TableCell>
              <TableCell>{csv.columns?.toLocaleString('fr-FR') ?? '-'}</TableCell>
              <TableCell>{formatBytes(csv.size_bytes)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

function DataPreview({ preview, loadedRows }: { preview: TablePreview; loadedRows?: number }) {
  const visibleRows = loadedRows ?? preview.rows.length
  return (
    <Box className="tableBlock">
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
        <Typography variant="h6">{preview.layer}.{preview.table}</Typography>
        <Chip label={`${visibleRows.toLocaleString('fr-FR')} / ${preview.total_rows.toLocaleString('fr-FR')} lignes`} color="primary" size="small" />
      </Box>
      <TableContainer sx={{ maxHeight: 580 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              {preview.columns.map((column) => <TableCell key={column}>{column}</TableCell>)}
            </TableRow>
          </TableHead>
          <TableBody>
            {preview.rows.map((row, index) => (
              <TableRow key={index} hover>
                {preview.columns.map((column) => <TableCell key={column}>{formatCell(row[column])}</TableCell>)}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  )
}

function LogPanel({ title, text }: { title: string; text: string }) {
  return (
    <Paper className="panel">
      <SectionHeader title={title} icon={<TerminalIcon fontSize="small" />} />
      <Typography variant="caption" color="text.secondary">{text.split('\n').filter(Boolean).length.toLocaleString('fr-FR')} lignes</Typography>
      <Box component="pre" className="logBox">
        {text || 'Aucun log pour ce run.'}
      </Box>
    </Paper>
  )
}

function formatCell(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} o`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} Ko`
  return `${(value / 1024 / 1024).toFixed(1)} Mo`
}

function messageFrom(error: unknown): string {
  return error instanceof Error ? error.message : 'Erreur inconnue'
}
