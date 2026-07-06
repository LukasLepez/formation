import { Box, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material'
import type { TablePreview } from '../types'
import { formatCell } from '../lib/format'

export function DataPreview({ preview, loadedRows }: { preview: TablePreview; loadedRows?: number }) {
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
