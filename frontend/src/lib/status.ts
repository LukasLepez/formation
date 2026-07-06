import type { RunStatus } from '../types'

export const statusColor: Record<RunStatus, 'default' | 'success' | 'error' | 'warning' | 'info'> = {
  queued: 'info',
  running: 'warning',
  success: 'success',
  failed: 'error',
}

export const statusLabel: Record<RunStatus, string> = {
  queued: 'En attente',
  running: 'En cours',
  success: 'Succès',
  failed: 'Échec',
}

export const statusValueColor: Record<RunStatus, string> = {
  queued: '#93c5fd',
  running: '#fbbf24',
  success: '#22c55e',
  failed: '#f87171',
}
