export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<T>
}

export async function apiText(path: string): Promise<string> {
  const response = await fetch(`/api${path}`)
  if (!response.ok) throw new Error(await response.text())
  return response.text()
}

export function messageFrom(error: unknown): string {
  return error instanceof Error ? error.message : 'Erreur inconnue'
}
