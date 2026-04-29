// Server component wrapper that resolves the dynamic params Promise
// before handing off to the interactive client component. The split
// avoids a known SSR pitfall where calling React 19's `use(params)`
// inside a "use client" component without an explicit Suspense
// boundary surfaces as an HTTP 500 in production builds. The server
// component awaits cleanly and passes the decoded segment as a plain
// prop, so ConceptView never has to suspend on the params promise.

import { ConceptView } from "./ConceptView";

// Browser URL bars routinely produce malformed percent-escapes
// (e.g. a stray "%" character pasted into the path). The default
// `decodeURIComponent` would throw a URIError, which Next surfaces
// as a 500 error boundary — fall back to the raw segment instead so
// the user still gets the "no match" UI.
function safeDecode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

type Props = { params: Promise<{ concept: string }> };

export default async function ExplainConceptPage({ params }: Props) {
  const { concept } = await params;
  return <ConceptView decoded={safeDecode(concept)} />;
}
