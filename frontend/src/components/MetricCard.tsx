import { Paper, Typography } from '@mui/material'

type MetricCardProps = {
  label: string
  value: string
  helper: string
  valueColor?: string
}

export function MetricCard({ label, value, helper, valueColor }: MetricCardProps) {
  return (
    <Paper className="metricCard">
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography variant="h6" sx={{ color: valueColor }}>{value}</Typography>
      <Typography variant="caption" color="text.secondary">{helper}</Typography>
    </Paper>
  )
}
