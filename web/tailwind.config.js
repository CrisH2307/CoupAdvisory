/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        poker: {
          bg1: "#17192a",
          bg2: "#101427",
          bg3: "#0b0f1f",
          ink: "#edf3ff",
          muted: "#9eaed1",
          panel: "#131a31",
          panel2: "#18213b",
          line: "#344974",
          green1: "#0f8553",
          green2: "#0a6f44",
          green3: "#085a37",
          red: "#c73b3b",
        },
      },
      fontFamily: {
        display: ["Orbitron", "ui-sans-serif", "system-ui"],
        body: ["Rajdhani", "ui-sans-serif", "system-ui"],
      },
    },
  },
  plugins: [],
};
