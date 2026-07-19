"use client";

import { useState } from "react";
import { track, type IdeaCard as IdeaCardT } from "@/lib/api";

const BADGE: Record<string, { label: string; dot: string; text: string }> = {
  novel: { label: "NOVEL", dot: "🟢", text: "text-green-400" },
  overlaps: { label: "OVERLAPS", dot: "🟡", text: "text-yellow-400" },
  scooped: { label: "SCOOPED", dot: "🔴", text: "text-red-400" },
};

export function IdeaCard({ card }: { card: IdeaCardT }) {
  const [open, setOpen] = useState<string | null>(null);
  const badge = BADGE[card.novelty_status] ?? BADGE.novel;
  const priorCount = card.prior_art.length;

  const toggle = (section: string) => {
    setOpen((cur) => (cur === section ? null : section));
    track("expand_idea", String(card.id));
  };

  return (
    <article className="rounded-lg border border-ink-line bg-ink-soft/60 p-5 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <span className={`font-mono text-xs tracking-wide ${badge.text}`}>
          {badge.dot} {badge.label}
        </span>
        {priorCount > 0 && (
          <span className="text-xs text-zinc-500">
            ✓ checked against {priorCount} paper{priorCount === 1 ? "" : "s"}
          </span>
        )}
      </div>

      <h3 className="text-lg font-semibold text-white leading-snug">
        {card.title}
      </h3>
      {card.hook && <p className="text-sm text-zinc-400">{card.hook}</p>}

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {card.mechanism && (
          <button
            className="text-accent hover:text-accent-soft"
            onClick={() => toggle("mechanism")}
          >
            ▸ mechanism
          </button>
        )}
        {card.falsification && (
          <button
            className="text-accent hover:text-accent-soft"
            onClick={() => toggle("falsify")}
          >
            ▸ how to falsify
          </button>
        )}
        {priorCount > 0 && (
          <button
            className="text-accent hover:text-accent-soft"
            onClick={() => toggle("prior")}
          >
            ▸ prior art
          </button>
        )}
      </div>

      {open === "mechanism" && (
        <p className="text-sm text-zinc-300 border-l-2 border-ink-line pl-3">
          {card.mechanism}
        </p>
      )}
      {open === "falsify" && (
        <p className="text-sm text-zinc-300 border-l-2 border-ink-line pl-3">
          {card.falsification}
        </p>
      )}
      {open === "prior" && (
        <div className="text-sm text-zinc-300 border-l-2 border-ink-line pl-3 space-y-1">
          {card.novelty_note && (
            <p className="italic text-zinc-400">{card.novelty_note}</p>
          )}
          <ul className="space-y-1">
            {card.prior_art.map((p, i) => (
              <li key={i}>
                {p.url ? (
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-accent hover:text-accent-soft"
                  >
                    {p.title}
                  </a>
                ) : (
                  <span>{p.title}</span>
                )}
                {p.year ? <span className="text-zinc-500"> ({p.year})</span> : null}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="font-mono text-[11px] text-zinc-500">
        grounded in {card.grounded_paper_count} paper
        {card.grounded_paper_count === 1 ? "" : "s"}
        {card.signature_terms.length > 0 &&
          ` · ${card.signature_terms.join(", ")}`}
      </div>
    </article>
  );
}
