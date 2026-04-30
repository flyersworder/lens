"use client";

import { useState, type FormEvent } from "react";

type Props = {
  initial?: string;
  placeholder?: string;
  cta?: string;
  onSubmit: (q: string) => void | Promise<void>;
  busy?: boolean;
};

export function SearchBox({
  initial = "",
  placeholder = "Search papers, concepts, or paste a tradeoff…",
  cta = "Search",
  onSubmit,
  busy = false,
}: Props) {
  const [q, setQ] = useState(initial);

  function submit(e: FormEvent) {
    e.preventDefault();
    const trimmed = q.trim();
    if (!trimmed || busy) return;
    onSubmit(trimmed);
  }

  return (
    <form onSubmit={submit} className="flex gap-2 sm:gap-3">
      <input
        autoFocus
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={placeholder}
        className="min-w-0 flex-1 rounded-lg border border-ink-line bg-ink-soft px-3 sm:px-4 py-3 text-base sm:text-sm placeholder:text-zinc-500 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/40"
      />
      <button
        type="submit"
        disabled={busy || !q.trim()}
        className="shrink-0 rounded-lg bg-accent px-4 sm:px-5 py-3 text-sm font-medium text-white transition disabled:opacity-40 hover:bg-accent-soft hover:text-ink"
      >
        {busy ? "…" : cta}
      </button>
    </form>
  );
}
