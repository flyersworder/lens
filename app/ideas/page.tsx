"use client";

import { useEffect, useState } from "react";
import { ideas, type IdeasResponse } from "@/lib/api";
import { IdeaCard } from "../components/IdeaCard";
import { useTrackOnce } from "@/lib/use-track-once";

type Filter = "novel" | "overlaps" | "scooped" | null;

const CHIPS: Array<{ key: Exclude<Filter, null>; label: string }> = [
  { key: "novel", label: "Novel" },
  { key: "overlaps", label: "Overlaps" },
  { key: "scooped", label: "Scooped" },
];

export default function IdeasPage() {
  useTrackOnce("view_ideas");
  const [data, setData] = useState<IdeasResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>(null);

  useEffect(() => {
    ideas()
      .then(setData)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  const cards = data
    ? filter
      ? data.cards.filter((c) => c.novelty_status === filter)
      : data.cards
    : [];

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">
          Ideas
        </h1>
        <p className="max-w-2xl text-sm sm:text-base text-zinc-400">
          Research ideas generated from the LENS corpus, each checked against
          real prior art via OpenAlex. The verdict says whether the core idea
          looks novel, overlaps existing work, or is already covered.
        </p>
        {data && (
          <div className="flex flex-wrap gap-2">
            {CHIPS.map((chip) => {
              const n = data.counts[chip.key];
              const active = filter === chip.key;
              return (
                <button
                  key={chip.key}
                  onClick={() => setFilter(active ? null : chip.key)}
                  className={`rounded-full border px-3 py-1 font-mono text-xs transition ${
                    active
                      ? "border-accent bg-accent/15 text-accent"
                      : "border-ink-line text-zinc-400 hover:text-white"
                  }`}
                >
                  {chip.label} {n}
                </button>
              );
            })}
          </div>
        )}
      </section>

      {err && <p className="text-sm text-red-400">Error: {err}</p>}

      {data && cards.length === 0 && !err && (
        <div className="rounded-lg border border-ink-line bg-ink-soft/60 p-6 text-zinc-300">
          No vetted ideas yet.
        </div>
      )}

      <div className="space-y-4">
        {cards.map((card) => (
          <IdeaCard key={card.id} card={card} />
        ))}
      </div>
    </div>
  );
}
