import { useEffect, useState } from 'react'
import { Box, Chip, IconButton, Tooltip, Typography } from '@mui/material'
import RefreshIcon from '@mui/icons-material/Refresh'
import { NavLink, Outlet } from 'react-router-dom'
import { api } from '../lib/api'

type HealthStatus = 'ok' | 'ko' | 'loading'

export function AppShell() {
  const [health, setHealth] = useState<HealthStatus>('loading')

  async function refreshHealth() {
    try {
      const healthData = await api<{ status: string }>('/health')
      setHealth(healthData.status === 'ok' ? 'ok' : 'ko')
    } catch {
      setHealth('ko')
    }
  }

  useEffect(() => {
    void refreshHealth()
  }, [])

  return (
    <Box className="appFrame">
      <Box component="nav" className="sideNav">
        <Box className="brandBlock">
          <Box>
            <Typography variant="subtitle1">InduSense</Typography>
            <Typography variant="caption" color="text.secondary">Application</Typography>
          </Box>
        </Box>
        <Box className="navLinks">
          <NavLink to="/ingestion" className={({ isActive }) => `navLink ${isActive ? 'navLinkActive' : ''}`}>
            <span>Ingestion</span>
          </NavLink>
          <NavLink to="/maintenance-ml" className={({ isActive }) => `navLink ${isActive ? 'navLinkActive' : ''}`}>
            <span>Maintenance ML</span>
          </NavLink>
        </Box>
        <Box className="navStatus">
          <Chip
            label={health === 'ok' ? 'API connectée' : health === 'loading' ? 'Connexion API' : 'API indisponible'}
            color={health === 'ok' ? 'success' : 'error'}
            size="small"
          />
          <Tooltip title="Rafraîchir">
            <IconButton onClick={() => void refreshHealth()} size="small">
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      <Outlet />
    </Box>
  )
}
