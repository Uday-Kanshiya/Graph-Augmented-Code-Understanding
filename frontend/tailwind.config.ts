import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#151718",
        panel: "#f7f8f6",
        line: "#dfe3df",
        accent: "#0f766e",
        rust: "#a44b2f",
        plum: "#6d4b7d",
      },
      boxShadow: {
        soft: "0 16px 40px rgba(21, 23, 24, 0.08)",
      },
    },
  },
  plugins: [],
};

export default config;

