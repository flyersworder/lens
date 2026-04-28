"use client";

import { useRouter } from "next/navigation";
import { useTrackOnce } from "@/lib/use-track-once";
import { SearchBox } from "../components/SearchBox";

export default function ExplainIndex() {
  const router = useRouter();
  useTrackOnce("view_explain");

  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <h1 className="text-3xl font-semibold tracking-tight">Explain a concept</h1>
        <p className="max-w-2xl text-zinc-400">
          Type any concept (architecture slot, principle, parameter). LENS
          disambiguates against the vocabulary and synthesizes a narrative
          backed by source papers.
        </p>
        <SearchBox
          placeholder="e.g. grouped query attention"
          cta="Explain"
          onSubmit={(q) => router.push(`/explain/${encodeURIComponent(q)}`)}
        />
      </section>
    </div>
  );
}
