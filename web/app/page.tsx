import Link from "next/link";
import TextType from "@/components/TextType";
import TrueFocus from "@/components/TrueFocus";

export default function Home() {
  return (
    <main className="relative h-screen w-screen overflow-hidden bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white">
      {/* Smaller aurora/ice glow accents */}
      <div className="pointer-events-none absolute -top-24 -left-16 h-64 w-64 rounded-full bg-cyan-500/10 blur-[100px]" />
      <div className="pointer-events-none absolute -bottom-24 -right-16 h-80 w-80 rounded-full bg-indigo-500/10 blur-[120px]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      {/* Centered content */}
      <section className="relative z-10 flex h-full items-center justify-center">
        <div className="max-w-2xl px-4 text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-[10px] uppercase tracking-widest text-white/70">
            Arctic Overwatch
          </div>

          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl leading-snug">
            Maritime Anomaly Detection <br className="hidden sm:block" />
            <TextType
              text={[
                "for sanction evasion",
                "for dark vessels",
                "for Arctic waters",
              ]}
              typingSpeed={75}
              pauseDuration={1500}
              showCursor={true}
              cursorCharacter="|"
            />
          </h1>

          <p className="mx-auto mt-4 max-w-xl text-balance text-sm text-white/70 sm:text-base">
            A focused surface picture built for
          </p>

          {/* TrueFocus effect */}
          <div className="mt-4">
            <TrueFocus
              sentence="Clarity, Speed, Trust"
              manualMode={false}
              blurAmount={5}
              borderColor="red"
              animationDuration={2}
              pauseBetweenAnimations={1}
            />
          </div>

          <div className="mx-auto mt-8 flex w-full max-w-sm flex-col items-center justify-center gap-3 sm:flex-row">
            <Link
              href="/globe"
              className="w-full rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 px-5 py-3 text-center text-sm font-semibold shadow-md shadow-cyan-900/40 transition hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-cyan-400/40 sm:w-auto"
            >
              🚢 Open Live Map
            </Link>

            <Link
              href="https://github.com/"
              className="w-full rounded-xl border border-white/15 bg-white/5 px-5 py-3 text-center text-sm font-medium text-white/80 transition hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white/20 sm:w-auto"
            >
              📖 Docs / About
            </Link>
          </div>
        </div>
      </section>

      {/* Footer line */}
      <footer className="pointer-events-none absolute bottom-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-2 text-[10px] text-white/40">
        <span>© {new Date().getFullYear()} Arctic Overwatch</span>
        <span className="hidden sm:block">No unnecessary UI. Just signal.</span>
      </footer>
    </main>
  );
}
