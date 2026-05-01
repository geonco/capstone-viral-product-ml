import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0d12",
        panel: "#11141b",
        panel2: "#161a23",
        border: "#222734",
        accent: "#7c5cff",
        accent2: "#22d3ee",
        good: "#34d399",
        bad: "#f87171",
        warn: "#fbbf24",
        text: "#e5e7eb",
        sub: "#9aa3b2",
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "Pretendard", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
