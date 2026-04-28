"use client";

import { useEffect, useRef } from "react";
import { track } from "./api";

/**
 * Fire a usage event exactly once per *logical* mount, keyed by `key`.
 *
 * Why this exists: React 19 + Next 16 dev/Strict Mode deliberately
 * runs effects twice on first mount to surface accidental side-effect
 * coupling. For naive `useEffect(() => track("view_*"), [])` that
 * means every page-view is double-counted in dev. The same pattern
 * also handles route remounts that share a `key` — when the user
 * navigates `/explain/foo` → `/explain/bar` the effect re-runs with
 * a new key, but `/explain/foo` → re-render of the same page should
 * not re-fire.
 */
export function useTrackOnce(event: string, key?: string): void {
  const lastKey = useRef<string | null>(null);
  useEffect(() => {
    const seen = key ?? "<no-key>";
    if (lastKey.current === seen) return;
    lastKey.current = seen;
    track(event, key);
  }, [event, key]);
}
