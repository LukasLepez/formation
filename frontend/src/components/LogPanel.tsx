import { Box, Paper, Typography } from '@mui/material'
import TerminalIcon from '@mui/icons-material/Terminal'
import { SectionHeader } from './SectionHeader'

export function LogPanel({ title, text }: { title: string; text: string }) {
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
