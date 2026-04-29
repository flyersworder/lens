"use client";

import { useEffect, useRef, useState } from "react";
import { explain, track } from "@/lib/api";

type ExplainResult = {
  resolved_type: string;
  resolved_id: string;
  resolved_name: string;
  narrative: string;
  evolution?: string[];
  tradeoffs?: Array<Record<string, unknown>>;
  connections?: string[];
  paper_refs?: string[];
  alternatives?: Array<Record<string, unknown>>;
};

type Props = { decoded: string };

export function ConceptView({ decoded }: Props) {
  const [data, setData] = useState<ExplainResult | null>(null);
  const [busy, setBusy] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // Strict Mode runs effects twice on first mount in dev. The ref
  // de-dupes by `decoded`, so the same concept fires telemetry +
  // network exactly once per user navigation.
  const lastFetchKey = useRef<string | null>(null);

  useEffect(() => {
    if (lastFetchKey.current === decoded) return;
    lastFetchKey.current = decoded;
    // Two events on purpose:
    //   * view_explain_concept — fires on every concept-page load,
    //     including failures. Use this for "how often is the concept
    //     route hit at all".
    //   * explain — fires only on a successful resolution. Use this
    //     to compute resolution success rate (= explain / view_explain_concept).
    // The /explain index page emits "view_explain" separately, so
    // landing-page traffic stays distinct from concept-page traffic.
    track("view_explain_concept", decoded);
    setBusy(true);
    explain({ query: decoded })
      .then((r) => {
        setData((r.result as ExplainResult | null) ?? null);
        track("explain", decoded);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  }, [decoded]);

  return (
    <div className="space-y-8">
      <section>
        <a
          href="/explain"
          className="text-xs uppercase tracking-widest text-zinc-500 hover:text-white"
        >
          ← explain
        </a>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">
          {data?.resolved_name ?? decoded}
        </h1>
        {data && (
          <p className="mt-1 text-xs uppercase tracking-widest text-zinc-500">
            {data.resolved_type} ·{" "}
            <span className="font-mono">{data.resolved_id}</span>
          </p>
        )}
      </section>

      {busy && <p className="text-sm text-zinc-500">Resolving and synthesizing…</p>}
      {err && <p className="text-sm text-red-400">Error: {err}</p>}

      {!busy && !data && !err && (
        <div className="rounded-lg border border-ink-line bg-ink-soft/60 p-6 text-zinc-300">
          No match for <span className="font-mono">{decoded}</span>. Try a
          related term, or{" "}
          <a className="text-accent hover:underline" href="/">search papers</a>.
        </div>
      )}

      {data && (
        <>
          <section className="prose prose-invert prose-zinc max-w-none whitespace-pre-wrap rounded-lg border border-ink-line bg-ink-soft/60 p-6 text-sm leading-relaxed text-zinc-200">
            {data.narrative}
          </section>

          {data.evolution && data.evolution.length > 0 && (
            <section>
              <h2 className="mb-3 text-sm uppercase tracking-widest text-zinc-500">
                Evolution
              </h2>
              <ol className="space-y-2 border-l border-ink-line pl-5 text-sm">
                {data.evolution.map((e, i) => (
                  <li key={i} className="text-zinc-300">{e}</li>
                ))}
              </ol>
            </section>
          )}

          {data.connections && data.connections.length > 0 && (
            <section>
              <h2 className="mb-3 text-sm uppercase tracking-widest text-zinc-500">
                Connected concepts
              </h2>
              <div className="flex flex-wrap gap-2">
                {data.connections.map((c) => (
                  <a
                    key={c}
                    href={`/explain/${encodeURIComponent(c)}`}
                    className="rounded-full border border-ink-line bg-ink-soft px-3 py-1 text-xs hover:border-accent/60 hover:text-white"
                  >
                    {c}
                  </a>
                ))}
              </div>
            </section>
          )}

          {data.paper_refs && data.paper_refs.length > 0 && (
            <section>
              <h2 className="mb-3 text-sm uppercase tracking-widest text-zinc-500">
                Source papers ({data.paper_refs.length})
              </h2>
              <ul className="space-y-1 font-mono text-xs text-zinc-400">
                {data.paper_refs.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}
