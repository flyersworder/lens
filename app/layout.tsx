import type { Metadata } from "next";
import "./globals.css";
import { StatsBar } from "./components/StatsBar";

export const metadata: Metadata = {
  title: "LENS — LLM Engineering Navigation System",
  description:
    "Search papers, resolve tradeoffs, and explain concepts grounded in a curated LLM-engineering corpus.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans">
        <header className="border-b border-ink-line/70 bg-ink/80 backdrop-blur sticky top-0 z-10">
          <div className="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between">
            <a
              href="/"
              className="font-mono text-sm tracking-tight text-white hover:text-accent-soft"
            >
              <span className="text-accent">▣</span> lens
            </a>
            <nav className="flex gap-6 text-sm text-zinc-400">
              <a className="hover:text-white" href="/">search</a>
              <a className="hover:text-white" href="/analyze">analyze</a>
              <a className="hover:text-white" href="/explain">explain</a>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-10">{children}</main>
        <footer className="mx-auto max-w-5xl px-6 py-12 mt-12">
          <StatsBar />
          <p className="mt-6 text-xs text-zinc-500">
            LENS is a research prototype. Read-only public API; non-commercial
            use under Vercel Hobby ToS. Backed by a 1536-dim hybrid index on
            Turso libSQL.
          </p>
        </footer>
      </body>
    </html>
  );
}
