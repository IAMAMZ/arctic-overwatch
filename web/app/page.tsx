import Link from "next/link";

export default function Home() {
  return (
    <main className="relative h-screen w-screen overflow-hidden bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white">
      {/* Subtle aurora/ice glow accents */}
      <div className="pointer-events-none absolute -top-32 -left-24 h-72 w-72 rounded-full bg-cyan-500/10 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-40 -right-24 h-96 w-96 rounded-full bg-indigo-500/10 blur-3xl" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      {/* Centered content */}
      <section className="relative z-10 flex h-full items-center justify-center">
        <div className="max-w-3xl px-6 text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs uppercase tracking-widest text-white/70">
            Arctic Overwatch
          </div>

          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            Maritime Anomaly Detection for Arctic Waters
          </h1>

          <p className="mx-auto mt-4 max-w-2xl text-balance text-sm text-white/70 sm:text-base">
            A focused surface picture for sanction evasion, dark vessels, and illegal activity.
            Built for clarity, speed, and trust.
          </p>

          <div className="mx-auto mt-10 flex w-full max-w-md flex-col items-center justify-center gap-3 sm:flex-row">
            <Link
              href="/map"
              className="w-full rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 px-6 py-3 text-center text-sm font-medium shadow-lg shadow-cyan-900/20 transition hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-cyan-400/40 sm:w-auto"
            >
              Open Live Map
            </Link>

            <Link
              href="https://github.com/" // replace or remove if not needed
              className="w-full rounded-xl border border-white/15 bg-white/5 px-6 py-3 text-center text-sm font-medium text-white/80 transition hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white/20 sm:w-auto"
            >
              Docs / About
            </Link>
          </div>
        </div>
      </section>

      {/* Footer line (fixed, subtle) */}
      <footer className="pointer-events-none absolute bottom-0 left-0 right-0 z-10 flex items-center justify-between px-6 py-3 text-xs text-white/40">
        <span>© {new Date().getFullYear()} Arctic Overwatch</span>
        <span className="hidden sm:block">No unnecessary UI. Just signal.</span>
      </footer>
    </main>
  );
}
