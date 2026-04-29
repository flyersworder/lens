"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  explainStream,
  track,
  type ExplainEventMeta,
} from "@/lib/api";

type Status = "loading" | "streaming" | "done" | "empty" | "error";

type State = {
  meta: ExplainEventMeta | null;
  narrative: string;
  status: Status;
  error: string | null;
};

const INITIAL: State = {
  meta: null,
  narrative: "",
  status: "loading",
  error: null,
};

type Props = { decoded: string };

export function ConceptView({ decoded }: Props) {
  const [s, setS] = useState<State>(INITIAL);

  // Strict Mode runs effects twice on first mount in dev. The ref
  // de-dupes by `decoded`, so the same concept fires telemetry +
  // network exactly once per user navigation.
  const lastFetchKey = useRef<string | null>(null);

  useEffect(() => {
    if (lastFetchKey.current === decoded) return;
    lastFetchKey.current = decoded;

    const ctrl = new AbortController();
    setS(INITIAL);
    track("view_explain_concept", decoded);

    (async () => {
      let sawMeta = false;
      try {
        for await (const ev of explainStream({ query: decoded }, ctrl.signal)) {
          if (ev.t === "meta") {
            sawMeta = true;
            setS((p) => ({ ...p, meta: ev, status: "streaming" }));
          } else if (ev.t === "token") {
            setS((p) => ({ ...p, narrative: p.narrative + ev.v }));
          } else if (ev.t === "empty") {
            setS((p) => ({ ...p, status: "empty" }));
            return;
          } else if (ev.t === "error") {
            setS((p) => ({ ...p, status: "error", error: ev.msg }));
            return;
          }
        }
        if (sawMeta) {
          setS((p) => ({ ...p, status: "done" }));
          track("explain", decoded);
        }
      } catch (e) {
        if ((e as { name?: string })?.name === "AbortError") return;
        setS((p) => ({
          ...p,
          status: "error",
          error: e instanceof Error ? e.message : String(e),
        }));
      }
    })();

    return () => ctrl.abort();
  }, [decoded]);

  const data = s.meta;

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

      {s.status === "loading" && (
        <p className="text-sm text-zinc-500">
          Resolving candidates… synthesis can take 30–60s.
        </p>
      )}
      {s.status === "error" && (
        <p className="text-sm text-red-400">Error: {s.error}</p>
      )}
      {s.status === "empty" && (
        <div className="rounded-lg border border-ink-line bg-ink-soft/60 p-6 text-zinc-300">
          No match for <span className="font-mono">{decoded}</span>. Try a
          related term, or{" "}
          <a className="text-accent hover:underline" href="/">
            search papers
          </a>
          .
        </div>
      )}

      {data && (
        <>
          <section
            className={[
              "prose prose-invert prose-zinc max-w-none",
              "prose-headings:font-semibold prose-headings:text-white prose-headings:tracking-tight",
              "prose-h1:text-3xl prose-h1:mt-6 prose-h1:mb-3",
              "prose-h2:text-2xl prose-h2:mt-7 prose-h2:mb-3 prose-h2:border-b prose-h2:border-ink-line prose-h2:pb-2",
              "prose-h3:text-xl prose-h3:mt-5 prose-h3:mb-2",
              "prose-h4:text-lg prose-h4:mt-4 prose-h4:mb-2",
              "prose-p:my-3 prose-p:text-zinc-200",
              "prose-strong:text-white prose-em:text-zinc-100",
              "prose-code:text-accent-soft prose-code:before:content-none prose-code:after:content-none prose-code:rounded prose-code:bg-ink-soft prose-code:px-1 prose-code:py-0.5",
              "prose-li:my-1",
              "prose-table:text-sm",
              "prose-a:text-accent hover:prose-a:underline",
              "rounded-lg border border-ink-line bg-ink-soft/60 p-6 leading-relaxed",
            ].join(" ")}
          >
            {s.narrative ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {s.narrative}
              </ReactMarkdown>
            ) : (
              <span className="text-zinc-500">Synthesizing…</span>
            )}
            {s.status === "streaming" && (
              <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-accent align-middle" />
            )}
          </section>

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
