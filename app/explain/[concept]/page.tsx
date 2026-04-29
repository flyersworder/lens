// Minimal diagnostic version to isolate why this dynamic route 500s on Vercel
// while working locally (next start).
import { ConceptView } from "./ConceptView";

function safeDecode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

type Props = { params: Promise<{ concept: string }> };

export default async function ExplainConceptPage({ params }: Props) {
  try {
    const { concept } = await params;
    return <ConceptView decoded={safeDecode(concept)} />;
  } catch (err) {
    // Surface the actual SSR error in the response body so we can read
    // it via `curl` instead of needing Vercel runtime log access.
    const msg = err instanceof Error ? `${err.name}: ${err.message}\n${err.stack}` : String(err);
    return (
      <pre style={{ whiteSpace: "pre-wrap", padding: 24, color: "#f87171", fontFamily: "monospace" }}>
        {`SSR ERROR\n${msg}`}
      </pre>
    );
  }
}
