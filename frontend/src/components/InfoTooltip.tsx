import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import { IconButton, Tooltip } from '@mui/material'
import type { ReactNode } from 'react'

type InfoTooltipProps = {
  title: ReactNode
}

export function InfoTooltip({ title }: InfoTooltipProps) {
  return (
    <Tooltip title={title} arrow>
      <IconButton size="small" aria-label="Information" sx={{ p: 0.25 }}>
        <InfoOutlinedIcon fontSize="small" />
      </IconButton>
    </Tooltip>
  )
}
