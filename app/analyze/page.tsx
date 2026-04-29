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

// Tradeoff mode: serve/analyzer.py:analyze
type Principle = {
  principle_id?: string;
  name?: string;
  score?: number;
  count?: number;
  avg_confidence?: number;
  paper_ids?: string[];
};

// Architecture mode: serve/analyzer.py:analyze_architecture
type Variant = {
  variant_name: string;
  slot?: string;
  properties?: string;
  replaces?: string | null;
  paper_ids?: string[];
  earliest_date?: string | null;
  vocab_id?: string;
  vocab_kind?: string;
  tradeoff_count?: number;
};

// Agentic mode: serve/analyzer.py:analyze_agentic
type ResolvedComponent = {
  name: string;
  vocab_id: string | null;
  vocab_kind: string | null;
};
type Pattern = {
  pattern_name: string;
  category?: string;
  structure?: string;
  use_case?: string;
  components?: ResolvedComponent[];
  paper_ids?: string[];
  earliest_date?: string | null;
};

type AnalyzeResponse = {
  query?: string;
  improving?: string;
  worsening?: string;
  principles?: Principle[];
  slot?: string;
  slot_id?: string | null;
  variants?: Variant[];
  category?: string;
  category_id?: string | null;
  patterns?: Pattern[];
};

function yearOf(date: string | null | undefined): string | null {
  if (!date || date === "1970-01-01") return null;
  return date.slice(0, 4);
}

const EMPTY_COPY: Record<AnalysisType, string> = {
  tradeoff:
    "No principles indexed for this tradeoff. Try rephrasing — LLM classification varies between runs.",
  architecture:
    "No architecture variants found for this request. Try rephrasing — LLM slot resolution varies between runs.",
  agentic:
    "No agentic patterns found for this request. Try rephrasing — LLM category resolution varies between runs.",
};

function MetaLine({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return <p className="mt-1 text-xs text-zinc-500">{items.join(" · ")}</p>;
}

function PaperRefs({ ids }: { ids?: string[] }) {
  if (!ids?.length) return null;
  return (
    <p className="mt-2 font-mono text-[11px] text-zinc-500">
      {ids.map((id, i) => (
        <span key={id}>
          {i > 0 && " · "}
          <a
            className="hover:text-accent hover:underline"
            href={`https://arxiv.org/abs/${id}`}
            target="_blank"
            rel="noreferrer"
          >
            arXiv:{id}
          </a>
        </span>
      ))}
    </p>
  );
}

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

  function selectMode(next: AnalysisType) {
    if (next === type) return;
    setType(next);
    setData(null);
    setErr(null);
    setLastQ("");
  }

  const principles = data?.principles ?? [];
  const variants = data?.variants ?? [];
  const patterns = data?.patterns ?? [];
  const isEmpty =
    data !== null &&
    principles.length === 0 &&
    variants.length === 0 &&
    patterns.length === 0;

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
                onClick={() => selectMode(id)}
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

        {/* `key={type}` forces SearchBox to remount on mode switch so its
        internal `q` state resets — clearing the previous query alongside
        the result panel. */}
        <SearchBox
          key={type}
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

          {/* Tradeoff: improving / worsening pair + ranked principles. */}
          {type === "tradeoff" && (data.improving || data.worsening) && (
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

          {type === "tradeoff" && principles.length > 0 && (
            <ul className="space-y-3">
              {principles.map((p, i) => {
                const pid = p.principle_id;
                const meta: string[] = [];
                if (typeof p.count === "number")
                  meta.push(`${p.count} tradeoff${p.count === 1 ? "" : "s"}`);
                if (typeof p.avg_confidence === "number")
                  meta.push(`conf ${p.avg_confidence.toFixed(2)}`);
                if (p.paper_ids?.length)
                  meta.push(
                    `${p.paper_ids.length} paper${p.paper_ids.length === 1 ? "" : "s"}`,
                  );
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
                    <MetaLine items={meta} />
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
          )}

          {/* Architecture: slot header + date-sorted variants with cross-links. */}
          {type === "architecture" && data.slot && (
            <div className="rounded-lg border border-accent/30 bg-accent/5 p-4">
              <div className="text-xs uppercase tracking-widest text-accent">
                Slot
              </div>
              <div className="mt-1 text-white">
                {data.slot_id ? (
                  <a
                    href={`/explain/${encodeURIComponent(data.slot_id)}`}
                    className="hover:underline"
                  >
                    {data.slot}
                  </a>
                ) : (
                  data.slot
                )}
              </div>
              {variants.length > 0 && (
                <p className="mt-2 text-xs text-zinc-400">
                  {variants.length} variant{variants.length === 1 ? "" : "s"} ·
                  ordered by year of first appearance
                </p>
              )}
            </div>
          )}

          {type === "architecture" && variants.length > 0 && (
            <ul className="space-y-3">
              {variants.map((v, i) => {
                const year = yearOf(v.earliest_date);
                return (
                  <li
                    key={`${v.variant_name}-${i}`}
                    className="rounded-lg border border-ink-line bg-ink-soft/60 p-5"
                  >
                    <div className="flex items-baseline justify-between gap-4">
                      <h3 className="text-base font-medium text-white">
                        {v.variant_name}
                      </h3>
                      {year && (
                        <span className="shrink-0 font-mono text-xs text-zinc-500">
                          {year}
                        </span>
                      )}
                    </div>
                    {v.replaces && (
                      <p className="mt-1 text-xs text-zinc-500">
                        ← replaces{" "}
                        <span className="text-zinc-300">{v.replaces}</span>
                      </p>
                    )}
                    {v.properties && (
                      <p className="mt-2 text-sm text-zinc-300">{v.properties}</p>
                    )}
                    {(v.vocab_id || typeof v.tradeoff_count === "number") && (
                      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
                        {typeof v.tradeoff_count === "number" && v.tradeoff_count > 0 && (
                          <span className="text-zinc-500">
                            involved in {v.tradeoff_count} tradeoff
                            {v.tradeoff_count === 1 ? "" : "s"}
                          </span>
                        )}
                        {v.vocab_id && (
                          <a
                            className="text-accent hover:underline"
                            href={`/explain/${encodeURIComponent(v.vocab_id)}`}
                          >
                            explain →
                          </a>
                        )}
                      </div>
                    )}
                    <PaperRefs ids={v.paper_ids} />
                  </li>
                );
              })}
            </ul>
          )}

          {/* Agentic: category header + date-sorted patterns with linked components. */}
          {type === "agentic" && data.category && (
            <div className="rounded-lg border border-accent/30 bg-accent/5 p-4">
              <div className="text-xs uppercase tracking-widest text-accent">
                Category
              </div>
              <div className="mt-1 text-white">
                {data.category_id ? (
                  <a
                    href={`/explain/${encodeURIComponent(data.category_id)}`}
                    className="hover:underline"
                  >
                    {data.category}
                  </a>
                ) : (
                  data.category
                )}
              </div>
              {patterns.length > 0 && (
                <p className="mt-2 text-xs text-zinc-400">
                  {patterns.length} pattern{patterns.length === 1 ? "" : "s"} ·
                  ordered by year of first appearance
                </p>
              )}
            </div>
          )}

          {type === "agentic" && patterns.length > 0 && (
            <ul className="space-y-3">
              {patterns.map((p, i) => {
                const year = yearOf(p.earliest_date);
                return (
                  <li
                    key={`${p.pattern_name}-${i}`}
                    className="rounded-lg border border-ink-line bg-ink-soft/60 p-5"
                  >
                    <div className="flex items-baseline justify-between gap-4">
                      <h3 className="text-base font-medium text-white">
                        {p.pattern_name}
                      </h3>
                      {year && (
                        <span className="shrink-0 font-mono text-xs text-zinc-500">
                          {year}
                        </span>
                      )}
                    </div>
                    {p.use_case && (
                      <p className="mt-1 text-xs uppercase tracking-widest text-zinc-500">
                        {p.use_case}
                      </p>
                    )}
                    {p.structure && (
                      <p className="mt-2 text-sm text-zinc-300">{p.structure}</p>
                    )}
                    {p.components && p.components.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {p.components.map((c, ci) =>
                          c.vocab_id ? (
                            <a
                              key={`${c.name}-${ci}`}
                              href={`/explain/${encodeURIComponent(c.vocab_id)}`}
                              className="rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-[11px] text-accent-soft hover:border-accent hover:text-white"
                            >
                              {c.name}
                            </a>
                          ) : (
                            <span
                              key={`${c.name}-${ci}`}
                              className="rounded-full border border-ink-line bg-ink-soft px-2 py-0.5 text-[11px] text-zinc-300"
                            >
                              {c.name}
                            </span>
                          ),
                        )}
                      </div>
                    )}
                    <PaperRefs ids={p.paper_ids} />
                  </li>
                );
              })}
            </ul>
          )}

          {isEmpty && (
            <div className="rounded-lg border border-ink-line bg-ink-soft/60 p-4 text-sm text-zinc-400">
              {EMPTY_COPY[type]}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
