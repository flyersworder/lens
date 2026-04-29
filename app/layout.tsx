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
          <div className="mx-auto max-w-5xl px-6 py-5 flex items-center justify-between">
            <a
              href="/"
              className="font-mono text-lg tracking-tight text-white hover:text-accent-soft"
            >
              <span className="text-accent">▣</span> lens
            </a>
            <nav className="flex items-center gap-8 text-base text-zinc-300">
              <a className="hover:text-white" href="/">search</a>
              <a className="hover:text-white" href="/analyze">analyze</a>
              <a className="hover:text-white" href="/explain">explain</a>
              <a className="hover:text-white" href="/usage">usage</a>
              <a
                className="flex items-center gap-1.5 hover:text-white"
                href="https://github.com/flyersworder/lens"
                target="_blank"
                rel="noreferrer"
                aria-label="GitHub repository"
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="h-5 w-5"
                  aria-hidden="true"
                >
                  <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56 0-.27-.01-1-.02-1.96-3.2.7-3.87-1.54-3.87-1.54-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.34.95.1-.74.4-1.24.72-1.52-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.78 0c2.21-1.49 3.17-1.18 3.17-1.18.63 1.58.24 2.75.12 3.04.74.81 1.18 1.84 1.18 3.1 0 4.42-2.69 5.4-5.25 5.68.41.36.78 1.06.78 2.13 0 1.54-.01 2.78-.01 3.16 0 .31.21.67.8.56C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z" />
                </svg>
                <span>github</span>
              </a>
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
