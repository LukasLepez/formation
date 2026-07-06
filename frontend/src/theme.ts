import { createTheme } from '@mui/material'

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#93c5fd' },
    secondary: { main: '#a7f3d0' },
    background: { default: '#0b0d12', paper: '#141821' },
    text: { primary: '#e5e7eb', secondary: '#9ca3af' },
    success: { main: '#22c55e' },
    error: { main: '#f87171' },
    warning: { main: '#fbbf24' },
  },
  shape: { borderRadius: 6 },
  typography: {
    fontFamily: '"Inter", "Segoe UI", system-ui, sans-serif',
    h4: { fontWeight: 750, letterSpacing: 0 },
    h6: { fontWeight: 750, letterSpacing: 0 },
    button: { textTransform: 'none', fontWeight: 700 },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          border: '1px solid #252b36',
          boxShadow: 'none',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: { borderColor: '#252b36', whiteSpace: 'nowrap' },
        head: { fontWeight: 750, color: '#d1d5db', backgroundColor: '#10141c' },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: { boxShadow: 'none' },
      },
    },
  },
})
