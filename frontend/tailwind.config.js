/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dreamy': '#eaebed',
        'ocean': '#006989',
      }
    },
  },
  plugins: [],
}