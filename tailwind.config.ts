import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      colors: {
        ink: { DEFAULT: "#0b0d12", soft: "#1a1d27", line: "#272a36" },
        accent: { DEFAULT: "#7c5cff", soft: "#b6a6ff" },
      },
    },
  },
  plugins: [],
};

export default config;
