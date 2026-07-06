export function formatCell(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function formatBytes(value: number): string {
  if (value < 1024) return `${value} o`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} Ko`
  return `${(value / 1024 / 1024).toFixed(1)} Mo`
}

export function formatNumber(value: number | undefined | null): string {
  if (value == null || Number.isNaN(value)) return '-'
  return value.toLocaleString('fr-FR', { maximumFractionDigits: 4 })
}

export function formatPercent(value: number | undefined | null): string {
  if (value == null || Number.isNaN(value)) return '-'
  return value.toLocaleString('fr-FR', { style: 'percent', maximumFractionDigits: 2 })
}
