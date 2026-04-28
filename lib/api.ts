// Tiny, framework-agnostic fetch helpers.
// Same-origin: in production these resolve to /api/* on the same Vercel
// deployment; in `next dev` we proxy via NEXT_PUBLIC_API_BASE so a
// developer can point at a remote prod API while iterating on UI.

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function jpost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ---------------- response shapes (mirror api/index.py) ---------------- //

export type SearchResult = {
  paper_id: string;
  title: string;
  date: string;
  authors_display: string;
  abstract_snippet: string;
  arxiv_id: string;
  venue: string | null;
  score?: number;
};

export type SearchResponse = { results: SearchResult[]; count: number };

export type Stats = {
  papers: number | null;
  vocabulary: {
    total: number | null;
    parameter: number | null;
    principle: number | null;
    arch_slot: number | null;
    agentic: number | null;
  };
  matrix_cells: number | null;
  tradeoffs: number | null;
  taxonomy_version: number | null;
};

export type AnalyzeRequest = {
  query: string;
  type?: "tradeoff" | "architecture" | "agentic";
};

export type ExplainRequest = { query: string; focus?: string };

// ----------------------------- endpoints ------------------------------- //

export const search = (q: string, limit = 10) =>
  jget<SearchResponse>(`/api/search?q=${encodeURIComponent(q)}&limit=${limit}`);

export const stats = () => jget<Stats>("/api/stats");

export const analyze = (req: AnalyzeRequest) =>
  jpost<Record<string, unknown>>("/api/analyze", req);

export const explain = (req: ExplainRequest) =>
  jpost<{ result: Record<string, unknown> | null }>("/api/explain", req);

// ----------------------------- tracking -------------------------------- //
//
// Fire-and-forget. Errors are swallowed so a failing track call never
// degrades user experience. Uses sendBeacon when available (survives
// page unload), falling back to fetch with keepalive.

export function track(event: string, query?: string): void {
  if (typeof window === "undefined") return;
  const body = JSON.stringify({ event, query: query ?? null });
  try {
    if (navigator.sendBeacon) {
      const blob = new Blob([body], { type: "application/json" });
      navigator.sendBeacon(`${BASE}/api/track`, blob);
      return;
    }
    fetch(`${BASE}/api/track`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* ignore */
  }
}
