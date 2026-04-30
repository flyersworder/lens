"use client";

import { useEffect, useRef, useState } from "react";
import { stats, type Stats } from "@/lib/api";

function fmt(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

export function StatsBar() {
  const [data, setData] = useState<Stats | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const fetched = useRef(false);

  useEffect(() => {
    if (fetched.current) return;
    fetched.current = true;
    stats()
      .then(setData)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  if (err) {
    return (
      <p className="text-xs text-zinc-500">
        Stats unavailable ({err}).
      </p>
    );
  }

  const cells: Array<[string, string]> = [
    ["papers", fmt(data?.papers ?? null)],
    ["concepts", fmt(data?.vocabulary.total ?? null)],
    ["principles", fmt(data?.vocabulary.principle ?? null)],
    ["parameters", fmt(data?.vocabulary.parameter ?? null)],
    ["tradeoffs", fmt(data?.matrix_cells ?? null)],
  ];

  return (
    <div
      role="region"
      aria-label="Corpus statistics"
      className="flex flex-wrap gap-x-5 gap-y-2 sm:gap-x-8 border-t border-ink-line/70 pt-6 text-xs uppercase tracking-wider text-zinc-500"
    >
      {cells.map(([label, value]) => (
        <div key={label} className="flex items-baseline gap-2">
          <span className="font-mono text-sm sm:text-base text-white normal-case tracking-normal">
            {value}
          </span>
          <span>{label}</span>
        </div>
      ))}
      {data?.taxonomy_version != null && (
        <div className="flex items-baseline gap-2 sm:ml-auto">
          <span className="font-mono text-zinc-300 normal-case tracking-normal">
            v{data.taxonomy_version}
          </span>
          <span>taxonomy</span>
        </div>
      )}
    </div>
  );
}
