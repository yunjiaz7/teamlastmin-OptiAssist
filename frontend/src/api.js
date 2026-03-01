const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function parseJsonOrThrow(response) {
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return data;
}

export async function uploadImage(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/upload-image`, {
    method: "POST",
    body: form
  });
  return parseJsonOrThrow(response);
}

export async function sendChatMessage(sessionId, message) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message })
  });
  return parseJsonOrThrow(response);
}

export function toAbsoluteAssetUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE}${path}`;
}
