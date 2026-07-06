import { Paper, Stack, Typography } from '@mui/material'
import type { ReactNode } from 'react'
import { InfoTooltip } from './InfoTooltip'

type MetricCardProps = {
  label: string
  value: string
  helper: string
  valueColor?: string
  info?: ReactNode
}

export function MetricCard({ label, value, helper, valueColor, info }: MetricCardProps) {
  return (
    <Paper className="metricCard">
      <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        {info && <InfoTooltip title={info} />}
      </Stack>
      <Typography variant="h6" sx={{ color: valueColor }}>{value}</Typography>
      <Typography variant="caption" color="text.secondary">{helper}</Typography>
    </Paper>
  )
}
