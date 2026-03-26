import axios from "axios"

/**
 * Human-readable message for failed API calls (network, FastAPI detail, etc.)
 */
export function extractApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.code === "ERR_NETWORK" || error.message === "Network Error") {
      return "Cannot reach the API. Start the backend (port 8000) or set NEXT_PUBLIC_API_URL in frontend/.env.local to match your server."
    }
    const d = error.response?.data as { detail?: unknown } | undefined
    if (typeof d?.detail === "string") return d.detail
    if (Array.isArray(d?.detail)) {
      return d.detail
        .map((x: { msg?: string; loc?: unknown }) => x.msg ?? JSON.stringify(x))
        .join("; ")
    }
    if (error.response?.status) {
      return `Request failed (HTTP ${error.response.status})`
    }
  }
  if (error instanceof Error) return error.message
  return "Request failed"
}
