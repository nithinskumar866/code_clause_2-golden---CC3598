/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'rgb(10, 10, 12)',
        card: 'rgb(20, 20, 25)',
        primary: {
          DEFAULT: 'rgb(99, 102, 241)', // indigo-500
          hover: 'rgb(79, 70, 229)',    // indigo-600
        },
        success: 'rgb(34, 197, 94)',
        error: 'rgb(239, 68, 68)',
        warning: 'rgb(234, 179, 8)',
        border: 'rgba(255, 255, 255, 0.08)',
        textMuted: 'rgb(156, 163, 175)',
      },
      fontFamily: {
        sans: ['Outfit', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
