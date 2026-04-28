"use client";

import { useState } from "react";
import { analyze, track } from "@/lib/api";
import { useTrackOnce } from "@/lib/use-track-once";
import { SearchBox } from "../components/SearchBox";

type AnalysisType = "tradeoff" | "architecture" | "agentic";

type Principle = {
  id?: string;
  name?: string;
  description?: string;
  score?: number;
};

type AnalyzeResponse = {
  improving?: string;
  worsening?: string;
  principles?: Principle[];
  // architecture / agentic variants return structurally similar shapes;
  // we render any "principles" / "candidates" array we find.
  candidates?: Principle[];
  [k: string]: unknown;
};

export default function AnalyzePage() {
  const [type, setType] = useState<AnalysisType>("tradeoff");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [lastQ, setLastQ] = useState("");

  useTrackOnce("view_analyze");

  async function run(q: string) {
    setBusy(true);
    setErr(null);
    setLastQ(q);
    try {
      const res = (await analyze({ query: q, type })) as AnalyzeResponse;
      setData(res);
      track("analyze", `${type}:${q}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setData(null);
    } finally {
      setBusy(false);
    }
  }

  const items = data?.principles ?? data?.candidates ?? [];

  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <h1 className="text-3xl font-semibold tracking-tight">Analyze</h1>
        <p className="max-w-2xl text-zinc-400">
          Resolve a tradeoff, surface candidate architectures, or map an agentic
          design pattern. Results are LLM-generated and grounded in the corpus.
        </p>

        <div className="flex flex-wrap gap-2 text-xs">
          {(["tradeoff", "architecture", "agentic"] as AnalysisType[]).map(
            (t) => (
              <button
                key={t}
                type="button"
                onClick={() => setType(t)}
                className={`rounded-full border px-3 py-1 transition ${
                  type === t
                    ? "border-accent bg-accent/20 text-white"
                    : "border-ink-line text-zinc-400 hover:border-accent/60"
                }`}
              >
                {t}
              </button>
            ),
          )}
        </div>

        <SearchBox
          placeholder={
            type === "tradeoff"
              ? "e.g. reduce inference latency without hurting accuracy"
              : type === "architecture"
                ? "e.g. retrieval-augmented chatbot with citations"
                : "e.g. multi-agent code review pipeline"
          }
          cta="Analyze"
          onSubmit={run}
          busy={busy}
        />
      </section>

      {err && <p className="text-sm text-red-400">Error: {err}</p>}

      {data && (
        <section className="space-y-4">
          <h2 className="text-sm uppercase tracking-widest text-zinc-500">
            Result · <span className="font-mono normal-case">{lastQ}</span>
          </h2>

          {(data.improving || data.worsening) && (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {data.improving && (
                <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-4">
                  <div className="text-xs uppercase tracking-widest text-emerald-400">
                    Improving
                  </div>
                  <div className="mt-1 text-white">{data.improving}</div>
                </div>
              )}
              {data.worsening && (
                <div className="rounded-lg border border-rose-500/30 bg-rose-500/5 p-4">
                  <div className="text-xs uppercase tracking-widest text-rose-400">
                    Worsening
                  </div>
                  <div className="mt-1 text-white">{data.worsening}</div>
                </div>
              )}
            </div>
          )}

          <ul className="space-y-3">
            {items.map((p, i) => (
              <li
                key={p.id ?? i}
                className="rounded-lg border border-ink-line bg-ink-soft/60 p-5"
              >
                <div className="flex items-baseline justify-between gap-4">
                  <h3 className="text-base font-medium text-white">
                    {p.name ?? p.id ?? `Candidate ${i + 1}`}
                  </h3>
                  {typeof p.score === "number" && (
                    <span className="font-mono text-xs text-accent">
                      {p.score.toFixed(2)}
                    </span>
                  )}
                </div>
                {p.description && (
                  <p className="mt-2 text-sm text-zinc-300">{p.description}</p>
                )}
                {p.id && (
                  <a
                    className="mt-3 inline-block text-xs text-accent hover:underline"
                    href={`/explain/${encodeURIComponent(p.id)}`}
                  >
                    explain →
                  </a>
                )}
              </li>
            ))}
          </ul>

          {items.length === 0 && (
            <details className="rounded-lg border border-ink-line bg-ink-soft/60 p-4 text-sm">
              <summary className="cursor-pointer text-zinc-400">
                Raw response
              </summary>
              <pre className="mt-3 overflow-x-auto text-xs text-zinc-300">
                {JSON.stringify(data, null, 2)}
              </pre>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
