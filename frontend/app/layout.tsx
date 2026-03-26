import type { Metadata } from "next"
import { Inter, JetBrains_Mono } from "next/font/google"
import { APP_DESCRIPTION, APP_TITLE } from "@/lib/brand"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { ScanFloat } from "@/components/ScanFloat"
import { Sidebar } from "@/components/Sidebar"
import "./globals.css"

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" })
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" })

export const metadata: Metadata = {
  title: APP_TITLE,
  description: APP_DESCRIPTION,
  openGraph: {
    title: APP_TITLE,
    description: APP_DESCRIPTION,
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${mono.variable} font-sans antialiased h-screen flex overflow-hidden bg-background text-foreground`}>
        <Sidebar />
        <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
          <ErrorBoundary>{children}</ErrorBoundary>
        </main>
        <ScanFloat />
      </body>
    </html>
  )
}
