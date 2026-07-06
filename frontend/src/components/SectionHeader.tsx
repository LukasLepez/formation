import { Stack, Typography } from '@mui/material'

type SectionHeaderProps = {
  title: string
  action?: React.ReactNode
  icon?: React.ReactNode
}

export function SectionHeader({ title, action, icon }: SectionHeaderProps) {
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
