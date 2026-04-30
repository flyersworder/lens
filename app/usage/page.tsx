"use client";

import { useEffect, useState } from "react";
import { usageSummary, type UsageSummary } from "@/lib/api";

const EVENT_LABELS: Record<string, string> = {
  view_home: "Home page view",
  view_analyze: "Analyze page view",
  view_explain: "Explain landing view",
  view_explain_concept: "Concept page view",
  search: "Search submitted",
  analyze: "Analyze submitted",
  explain: "Explain resolved",
};

function fmtDate(ts: number | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toISOString().slice(0, 10);
}

export default function UsagePage() {
  const [data, setData] = useState<UsageSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    usageSummary()
      .then(setData)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">Usage</h1>
        <p className="max-w-2xl text-sm sm:text-base text-zinc-400">
          Aggregate counts of feature interactions, recorded since the
          public deployment went live. Queries are SHA-256-hashed before
          storage — only the event type and timestamp are kept in the
          clear.
        </p>
        {data && (
          <p className="text-xs text-zinc-500">
            Tracking range:{" "}
            <span className="font-mono">{fmtDate(data.first_seen)}</span> →{" "}
            <span className="font-mono">{fmtDate(data.last_seen)}</span> · total{" "}
            <span className="font-mono">{data.total}</span> events
          </p>
        )}
      </section>

      {err && <p className="text-sm text-red-400">Error: {err}</p>}

      {data && data.events.length === 0 && !err && (
        <div className="rounded-lg border border-ink-line bg-ink-soft/60 p-6 text-zinc-300">
          No events tracked yet — usage tables are empty.
        </div>
      )}

      {data && data.events.length > 0 && (
        <ul className="divide-y divide-ink-line/60 rounded-lg border border-ink-line bg-ink-soft/60">
          {data.events.map((e) => {
            const pct = data.total > 0 ? (e.count / data.total) * 100 : 0;
            return (
              <li key={e.event} className="px-5 py-3">
                <div className="flex items-baseline justify-between gap-4">
                  <div>
                    <div className="text-sm text-white">
                      {EVENT_LABELS[e.event] ?? e.event}
                    </div>
                    <div className="font-mono text-[11px] text-zinc-500">
                      {e.event}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-base text-accent">
                      {e.count}
                    </div>
                    <div className="font-mono text-[11px] text-zinc-500">
                      {pct.toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-ink-line">
                  <div
                    className="h-full bg-accent/70"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
