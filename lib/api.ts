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

export type UsageSummary = {
  events: Array<{ event: string; count: number }>;
  total: number;
  first_seen: number | null;
  last_seen: number | null;
};

export const usageSummary = () =>
  jget<UsageSummary>("/api/usage-summary");

export const analyze = (req: AnalyzeRequest) =>
  jpost<Record<string, unknown>>("/api/analyze", req);

// /api/explain returns an NDJSON stream — one JSON object per line —
// not a single JSON envelope. See `explainStream` below.

export type ExplainEventMeta = {
  t: "meta";
  resolved_type: string;
  resolved_id: string;
  resolved_name: string;
  tradeoffs?: Array<Record<string, unknown>>;
  connections?: string[];
  paper_refs?: string[];
  evolution?: string[];
  alternatives?: Array<Record<string, unknown>>;
};
export type ExplainEventToken = { t: "token"; v: string };
export type ExplainEventEmpty = { t: "empty" };
export type ExplainEventError = { t: "error"; msg: string };
export type ExplainEvent =
  | ExplainEventMeta
  | ExplainEventToken
  | ExplainEventEmpty
  | ExplainEventError;

export async function* explainStream(
  req: ExplainRequest,
  signal?: AbortSignal,
): AsyncGenerator<ExplainEvent> {
  const res = await fetch(`${BASE}/api/explain`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
    signal,
    cache: "no-store",
  });
  if (!res.ok || !res.body) throw new Error(`/api/explain → ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl = buf.indexOf("\n");
    while (nl !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) yield JSON.parse(line) as ExplainEvent;
      nl = buf.indexOf("\n");
    }
  }
  const tail = buf.trim();
  if (tail) yield JSON.parse(tail) as ExplainEvent;
}

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
