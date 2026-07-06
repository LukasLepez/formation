import { Chip } from '@mui/material'
import type { RunStatus } from '../types'
import { statusColor, statusLabel } from '../lib/status'

export function StatusChip({ status }: { status: RunStatus }) {
  return <Chip label={statusLabel[status]} color={statusColor[status]} size="small" />
}
