"use client"

import React from "react"

interface Props {
  children: React.ReactNode
  fallback?: React.ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      if (this.props.fallback) return this.props.fallback
      return (
        <div className="max-w-xl mx-auto mt-20 p-6 card-surface border-red-500/20 bg-red-500/5 text-center">
          <p className="text-sm text-red-400 mb-2">Something went wrong</p>
          <p className="text-xs text-muted-foreground break-all mb-4">{this.state.error.message}</p>
          <button
            onClick={() => this.setState({ error: null })}
            className="btn-secondary text-xs"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
