"use client"
import { useState } from "react"
import { useRouter } from "next/navigation"

export function SearchBar() {
  const [query, setQuery] = useState("")
  const router = useRouter()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = query.trim()
    if (!trimmed) return
    router.push(`/analyze/${encodeURIComponent(trimmed)}`)
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 w-full max-w-xl">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Star name or TIC ID — e.g. K2-18, TRAPPIST-1, TIC 234994474"
        className="input-field flex-1"
      />
      <button type="submit" className="btn-primary">
        Analyze
      </button>
    </form>
  )
}
