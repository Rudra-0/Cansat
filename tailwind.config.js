/** Tailwind build for the Mission Control dashboard (replaces the cdn.tailwindcss.com runtime).
 *  Rebuild after changing classes in dashboard.html / js/dashboard.js:  npm run build:css
 *  The generated frontend/css/tailwind.css is committed so the server needs no Node toolchain. */
module.exports = {
  content: ['./frontend/dashboard.html', './frontend/js/dashboard.js'],
  theme: { extend: {} },
  plugins: [],
};
