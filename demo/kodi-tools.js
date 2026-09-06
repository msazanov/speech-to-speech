// Optional same-origin device tools. The device server enforces its allowlist.
export function kodiToolDefinitions(config) {
  return Array.isArray(config.kodiTools) ? config.kodiTools.filter(t =>
    t?.type === 'function' && t?.name?.startsWith('kodi_') && t?.parameters?.type === 'object') : [];
}

export async function executeKodiTool(name, args, fetcher = fetch) {
  const response = await fetcher('api/kodi', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, arguments: args}),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `Kodi error (${response.status})`);
  return JSON.stringify(payload);
}
