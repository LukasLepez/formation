import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import { IconButton, Tooltip } from '@mui/material'

type InfoTooltipProps = {
  title: string
}

export function InfoTooltip({ title }: InfoTooltipProps) {
  return (
    <Tooltip title={title} arrow>
      <IconButton size="small" aria-label={title} sx={{ p: 0.25 }}>
        <InfoOutlinedIcon fontSize="small" />
      </IconButton>
    </Tooltip>
  )
}
