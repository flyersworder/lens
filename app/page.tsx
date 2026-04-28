"use client";

import { useState } from "react";
import { search, track, type SearchResult } from "@/lib/api";
import { useTrackOnce } from "@/lib/use-track-once";
import { SearchBox } from "./components/SearchBox";

export default function HomePage() {
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lastQ, setLastQ] = useState("");

  useTrackOnce("view_home");

  async function run(q: string) {
    setBusy(true);
    setErr(null);
    setLastQ(q);
    try {
      const r = await search(q, 10);
      setResults(r.results);
      track("search", q);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setResults(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-12">
      <section className="space-y-6 pt-8">
        <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">
          Navigate the LLM&nbsp;engineering literature.
        </h1>
        <p className="max-w-2xl text-base text-zinc-400 md:text-lg">
          LENS extracts parameters, principles, and tradeoffs from a curated
          corpus, then lets you query them with hybrid keyword + vector search,
          structured tradeoff resolution, and concept explanations grounded in
          source papers.
        </p>
        <SearchBox onSubmit={run} busy={busy} />
        <div className="flex flex-wrap gap-3 text-xs text-zinc-500">
          <span>Try:</span>
          {[
            "transformer attention",
            "reduce inference latency",
            "mixture of experts",
            "speculative decoding",
          ].map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => run(s)}
              className="rounded-full border border-ink-line/80 bg-ink-soft px-3 py-1 hover:border-accent/60 hover:text-white"
            >
              {s}
            </button>
          ))}
        </div>
      </section>

      {err && (
        <p className="text-sm text-red-400">Error: {err}</p>
      )}

      {results && (
        <section className="space-y-4">
          <h2 className="text-sm uppercase tracking-widest text-zinc-500">
            {results.length} papers · query{" "}
            <span className="font-mono text-zinc-300">"{lastQ}"</span>
          </h2>
          {results.length === 0 && (
            <p className="text-zinc-400">
              No matches. Try a broader query, or{" "}
              <a className="text-accent hover:underline" href="/explain">
                explain a concept
              </a>{" "}
              instead.
            </p>
          )}
          <ul className="space-y-3">
            {results.map((r) => {
              const year =
                r.date && r.date !== "1970-01-01" ? r.date.slice(0, 4) : "—";
              const arxivUrl = r.arxiv_id
                ? `https://arxiv.org/abs/${r.arxiv_id}`
                : null;
              return (
                <li
                  key={r.paper_id}
                  className="rounded-lg border border-ink-line bg-ink-soft/60 p-5 transition hover:border-accent/50"
                >
                  <div className="flex items-baseline justify-between gap-4">
                    <h3 className="text-base font-medium text-white">
                      {arxivUrl ? (
                        <a
                          href={arxivUrl}
                          target="_blank"
                          rel="noreferrer"
                          className="hover:text-accent-soft"
                        >
                          {r.title}
                        </a>
                      ) : (
                        r.title
                      )}
                    </h3>
                    <span className="shrink-0 font-mono text-xs text-zinc-500">
                      {year}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-zinc-500">
                    {r.authors_display}
                    {r.venue ? ` · ${r.venue}` : ""}
                    {r.arxiv_id ? ` · arXiv:${r.arxiv_id}` : ""}
                  </p>
                  {r.abstract_snippet && (
                    <p className="mt-3 text-sm text-zinc-300">
                      {r.abstract_snippet}
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
