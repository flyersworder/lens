"use client";

import { useState } from "react";
import { analyze, track } from "@/lib/api";
import { useTrackOnce } from "@/lib/use-track-once";
import { SearchBox } from "../components/SearchBox";

type AnalysisType = "tradeoff" | "architecture" | "agentic";

const MODES: Array<{
  id: AnalysisType;
  label: string;
  blurb: string;
  placeholder: string;
}> = [
  {
    id: "tradeoff",
    label: "Tradeoff",
    blurb: "Improve one parameter without breaking another.",
    placeholder: "e.g. reduce inference latency without hurting accuracy",
  },
  {
    id: "architecture",
    label: "Architecture",
    blurb: "Find candidate designs for a system requirement.",
    placeholder: "e.g. retrieval-augmented chatbot with citations",
  },
  {
    id: "agentic",
    label: "Agentic",
    blurb: "Map an LLM-agent design pattern to your problem.",
    placeholder: "e.g. multi-agent code review pipeline",
  },
];

type Principle = {
  // backend (serve/analyzer.py:analyze) emits `principle_id`, but architecture
  // and agentic variants may emit `id`. Accept either.
  principle_id?: string;
  id?: string;
  name?: string;
  description?: string;
  score?: number;
  count?: number;
  avg_confidence?: number;
  paper_ids?: string[];
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

        <div className="space-y-2">
          <p className="text-xs uppercase tracking-widest text-zinc-500">
            Pick a mode
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {MODES.map(({ id, label, blurb }) => (
              <button
                key={id}
                type="button"
                onClick={() => {
                  if (id === type) return;
                  setType(id);
                  setData(null);
                  setErr(null);
                  setLastQ("");
                }}
                className={`rounded-lg border p-4 text-left transition ${
                  type === id
                    ? "border-accent bg-accent/15 text-white"
                    : "border-ink-line bg-ink-soft/40 text-zinc-300 hover:border-accent/60 hover:bg-ink-soft/70"
                }`}
              >
                <div className="text-base font-medium capitalize">{label}</div>
                <div className="mt-1 text-xs text-zinc-400">{blurb}</div>
              </button>
            ))}
          </div>
        </div>

        <SearchBox
          placeholder={MODES.find((m) => m.id === type)!.placeholder}
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
            {items.map((p, i) => {
              const pid = p.principle_id ?? p.id;
              const meta: string[] = [];
              if (typeof p.count === "number")
                meta.push(`${p.count} tradeoff${p.count === 1 ? "" : "s"}`);
              if (typeof p.avg_confidence === "number")
                meta.push(`conf ${p.avg_confidence.toFixed(2)}`);
              if (p.paper_ids?.length)
                meta.push(`${p.paper_ids.length} paper${p.paper_ids.length === 1 ? "" : "s"}`);
              return (
                <li
                  key={pid ?? i}
                  className="rounded-lg border border-ink-line bg-ink-soft/60 p-5"
                >
                  <div className="flex items-baseline justify-between gap-4">
                    <h3 className="text-base font-medium text-white">
                      {p.name ?? pid ?? `Candidate ${i + 1}`}
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
                  {meta.length > 0 && (
                    <p className="mt-1 text-xs text-zinc-500">{meta.join(" · ")}</p>
                  )}
                  {pid && (
                    <a
                      className="mt-3 inline-block text-xs text-accent hover:underline"
                      href={`/explain/${encodeURIComponent(pid)}`}
                    >
                      explain →
                    </a>
                  )}
                </li>
              );
            })}
          </ul>

          {items.length === 0 && (
            <div className="rounded-lg border border-ink-line bg-ink-soft/60 p-4 text-sm text-zinc-400">
              No principles indexed for this tradeoff
              {data.improving && data.worsening && (
                <>
                  {" "}(
                  <span className="font-mono">{data.improving}</span> →{" "}
                  <span className="font-mono">{data.worsening}</span>)
                </>
              )}
              . Try rephrasing — LLM classification varies between runs.
            </div>
          )}
        </section>
      )}
    </div>
  );
}
