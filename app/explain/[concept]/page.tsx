// Absolute-minimum diagnostic — no imports, no client component, no decode.
// If THIS still 500s, the issue is with async dynamic-route SSR itself, not
// our component logic.

type Props = { params: Promise<{ concept: string }> };

export default async function ExplainConceptPage({ params }: Props) {
  const { concept } = await params;
  return <div style={{ padding: 24, fontFamily: "monospace" }}>diagnostic ok: {concept}</div>;
}
