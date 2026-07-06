import { CssBaseline, ThemeProvider } from '@mui/material'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { IngestionPage } from './pages/IngestionPage'
import { MaintenanceMlPage } from './pages/MaintenanceMlPage'
import { theme } from './theme'
import './App.css'

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Navigate to="/ingestion" replace />} />
            <Route path="/ingestion" element={<IngestionPage />} />
            <Route path="/maintenance-ml" element={<MaintenanceMlPage />} />
            <Route path="*" element={<Navigate to="/ingestion" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}
