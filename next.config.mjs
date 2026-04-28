import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Silence "multiple lockfiles" warning — pin Turbopack's root to
  // this repo, not the parent home directory's stray package-lock.json.
  turbopack: { root: __dirname },
  // Trim the build so we don't try to bundle the Python/data tree.
  // Vercel's Next.js builder respects this; local `next build` does too.
  outputFileTracingExcludes: {
    "*": [
      "src/**",
      "tests/**",
      "scripts/**",
      "docs/**",
      "api/**",
      ".venv/**",
    ],
  },
  // Locally (both `next dev` and `next start`), proxy /api/* to a
  // sibling uvicorn process. On Vercel, the platform auto-detects
  // api/*.py as serverless functions on the same domain — so we
  // skip the rewrite when the ``VERCEL`` env var is set (Vercel
  // injects this on build + runtime, see vercel.com/docs/projects/environment-variables/system-environment-variables).
  // The result: same-origin contract holds in every environment
  // without needing NEXT_PUBLIC_API_BASE.
  // Override the upstream port via LENS_API_PORT (default 8000).
  async rewrites() {
    if (process.env.VERCEL) return [];
    const port = process.env.LENS_API_PORT || "8000";
    return [
      {
        source: "/api/:path*",
        destination: `http://127.0.0.1:${port}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
