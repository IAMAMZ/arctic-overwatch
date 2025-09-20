"use client";

import { useEffect, useRef, useState } from "react";
import {
  BarChart3,
  Ship,
  Eye,
  Zap,
  Mic,
  Volume2,
  VolumeX,
  Send,
} from "lucide-react";
import { GoogleGenerativeAI } from "@google/generative-ai";

/* ---------------- Types ---------------- */

type Stats = {
  total_detections: number;
  ais_matches: number;
  dark_vessels: number;
  avg_intensity: number;
};

type ChatItem = {
  role: "assistant" | "user";
  content: string;
};

/* ---------------- Component ---------------- */

export default function MapPage() {
  const [stats, setStats] = useState<Stats>({
    total_detections: 0,
    ais_matches: 0,
    dark_vessels: 0,
    avg_intensity: 0,
  });

  const [chat, setChat] = useState<ChatItem[]>([
    {
      role: "assistant",
      content:
        "Type **analyze Fram Strait** (or any region) to run a regional detection analysis. Otherwise, ask anything about dark shipping and sanctions in the Arctic.",
    },
  ]);

  const [input, setInput] = useState("");
  const [recording, setRecording] = useState(false);
  const [speaking, setSpeaking] = useState(true);

  // Speech APIs
  const recognitionRef = useRef<any>(null);
  const selectedVoiceRef = useRef<SpeechSynthesisVoice | null>(null);

  // Gemini
  const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY || "";
  const genAI = new GoogleGenerativeAI(apiKey);

  /* ---------------- Load Stats once ---------------- */
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("http://localhost:5000/api/stats");
        const data = await res.json();
        setStats(data);
      } catch (err) {
        console.error("Failed to load stats:", err);
      }
    })();
  }, []);

  /* ---------------- Web Speech: STT ---------------- */
  useEffect(() => {
    if (typeof window === "undefined") return;
    const w: any = window as any;
    const SR = w.webkitSpeechRecognition || w.SpeechRecognition;
    if (!SR) return;

    const recognition = new SR();
    recognition.lang = "en-GB"; // British English if available
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (e: any) => {
      const transcript = e.results[0][0].transcript;
      setInput(transcript);
    };

    recognition.onerror = () => {
      setRecording(false);
    };
    recognition.onend = () => setRecording(false);

    recognitionRef.current = recognition;
  }, []);

  const handleMicDown = () => {
    if (!recognitionRef.current) return;
    try {
      setRecording(true);
      recognitionRef.current.start();
    } catch {
      setRecording(false);
    }
  };

  const handleMicUp = () => {
    if (!recognitionRef.current) return;
    try {
      recognitionRef.current.stop();
    } catch {
      // ignore
    }
  };

  /* ---------------- Web Speech: TTS ---------------- */
  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    const pickVoice = () => {
      const voices = window.speechSynthesis.getVoices();
      // Prefer British English; else first English; else first voice.
      const enGB =
        voices.find((v) => v.lang === "en-GB") ||
        voices.find((v) => v.name.toLowerCase().includes("uk")) ||
        voices.find((v) => v.name.toLowerCase().includes("british"));
      const en =
        enGB ||
        voices.find((v) => v.lang?.toLowerCase().startsWith("en")) ||
        null;
      selectedVoiceRef.current = en ?? voices[0] ?? null;
    };

    pickVoice();
    window.speechSynthesis.onvoiceschanged = pickVoice;
  }, []);

  const speak = (text: string) => {
    if (!speaking || typeof window === "undefined" || !window.speechSynthesis)
      return;
    if (!text) return;
    const u = new SpeechSynthesisUtterance(text);
    if (selectedVoiceRef.current) u.voice = selectedVoiceRef.current;
    u.rate = 1;
    u.pitch = 1;
    window.speechSynthesis.speak(u);
  };

  const stopSpeaking = () => {
    if (typeof window !== "undefined" && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
    }
  };

  const toggleSpeaking = () => {
    setSpeaking((s) => {
      if (s) stopSpeaking();
      return !s;
    });
  };

  /* ---------------- Chat send ---------------- */
  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;

    setChat((p) => [...p, { role: "user", content: text }]);
    setInput("");

    // Detect “analyze …”
    const isAnalyze = /(^|\s)analyze\s+/i.test(text);

    try {
      if (isAnalyze) {
        // 1) Pull detections from backend
        const detRes = await fetch("http://localhost:5000/api/detections");
        const detections = await detRes.json();

        // 2) Ask Gemini for a richly formatted analysis (text output)
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        const analysisPrompt = `
You are an Arctic maritime analyst. Analyze the detections below and the user request.
Return a concise, serious, formatted output with:
- **Most active region** and detection count
- Rationale (distances/cluster logic at a high level — no raw coords)
- Top 3–5 regions and counts (comma-separated)
- Operational note on dark vessels / sanction-evasion risk

User message: ${text}
Detections JSON (subset or full): ${JSON.stringify(
          detections?.detections ?? []
        ).slice(0, 12000)}
Tone: professional, operational. Avoid fluff.`;

        const analysis = await model.generateContent(analysisPrompt);
        const formatted = analysis.response.text().trim();

        // 3) Ask Gemini to simplify that formatted text into a short TTS script
        const ttsPrompt = `Rewrite the following analysis for spoken audio in 2–3 short sentences, plain English, coherent and natural, no lists, no headings, no markup:

${formatted}

Constraints:
- Say the best region and count first.
- Mention one short operational implication about dark vessels/sanctions.
- Keep under 320 characters.`;

        const speechRes = await model.generateContent(ttsPrompt);
        const speechText = speechRes.response.text().trim();

        setChat((p) => [...p, { role: "assistant", content: formatted }]);
        speak(speechText);
      } else {
        // Domain-expert general Q&A
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        const qaPrompt = `
You are an Arctic maritime domain expert on sanctions, dark shipping, AIS spoofing, and Arctic waters.
User: ${text}
Answer with highly specific and accurate knowledge (operational tone).`;

        const res = await model.generateContent(qaPrompt);
        const fullText = res.response.text().trim();

        // Also create a short spoken version for clarity
        const ttsModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        const ttsPrompt = `Summarize the following answer into 1–2 short sentences for speech. Plain language, no lists/markup, max 250 characters:\n\n${fullText}`;
        const ttsRes = await ttsModel.generateContent(ttsPrompt);
        const voiceText = ttsRes.response.text().trim();

        setChat((p) => [...p, { role: "assistant", content: fullText }]);
        speak(voiceText);
      }
    } catch (err) {
      console.error(err);
      stopSpeaking();
      setChat((p) => [
        ...p,
        {
          role: "assistant",
          content:
            "⚠️ Analysis service error. Check the backend and your Gemini API key.",
        },
      ]);
    }
  };

  /* ---------------- Layout ---------------- */

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      {/* Top bar */}
      <header className="sticky top-0 z-20 border-b border-white/10 bg-black/30 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-r from-cyan-600 to-indigo-600 shadow-lg">
              <Ship className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">
                Arctic Overwatch
              </h1>
              <p className="text-xs text-white/60">
                SAR Vessel Detection System
              </p>
            </div>
          </div>

          <button
            onClick={toggleSpeaking}
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm hover:bg-white/10"
          >
            {speaking ? (
              <Volume2 className="h-4 w-4 text-emerald-400" />
            ) : (
              <VolumeX className="h-4 w-4 text-rose-400" />
            )}
            <span className="text-white/80">
              {speaking ? "Voice On" : "Muted"}
            </span>
          </button>
        </div>
      </header>

      {/* Main grid */}
      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-[1fr_400px]">
        {/* LEFT: Map + Stats */}
        <div className="space-y-6">
          {/* Stats (compact) */}
          <section className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatCard
              label="Detections"
              value={stats.total_detections.toLocaleString()}
              icon={<BarChart3 className="h-5 w-5 text-cyan-400" />}
              color="from-cyan-600/20 to-cyan-700/20 border-cyan-500/30 text-cyan-300"
            />
            <StatCard
              label="AIS Matches"
              value={stats.ais_matches.toLocaleString()}
              icon={<Eye className="h-5 w-5 text-emerald-400" />}
              color="from-emerald-600/20 to-emerald-700/20 border-emerald-500/30 text-emerald-300"
            />
            <StatCard
              label="Dark Vessels"
              value={stats.dark_vessels.toLocaleString()}
              icon={<Zap className="h-5 w-5 text-rose-400" />}
              color="from-rose-600/20 to-rose-700/20 border-rose-500/30 text-rose-300"
            />
            <StatCard
              label="Avg dB"
              value={stats.avg_intensity.toFixed(1)}
              icon={<BarChart3 className="h-5 w-5 text-yellow-400" />}
              color="from-yellow-600/20 to-yellow-700/20 border-yellow-500/30 text-yellow-300"
            />
          </section>

          {/* Map (slightly reduced height) */}
          <section className="overflow-hidden rounded-2xl border border-white/10 bg-white/5 shadow-xl">
            <div className="border-b border-white/10 bg-gradient-to-r from-cyan-700/50 to-indigo-700/50 px-5 py-3 text-sm font-medium">
              Detection Map
            </div>
            <div className="h-[520px]">
              <iframe
                src="http://localhost:5000/"
                className="h-full w-full border-none"
                title="Detection Map"
              />
            </div>
          </section>
        </div>

        {/* RIGHT: Chat panel */}
        <aside className="flex h-[calc(100vh-140px)] flex-col overflow-hidden rounded-2xl border border-white/10 bg-black/30 backdrop-blur-md shadow-2xl">
          <div className="border-b border-white/10 px-5 py-3 text-sm text-white/70">
            Analyst (Gemini)
            <span className="ml-2 text-xs text-white/40">
              Use <code className="rounded bg-white/10 px-1">analyze REGION</code>
            </span>
          </div>

          {/* Messages */}
          <div className="flex-1 space-y-3 overflow-y-auto p-4">
            {chat.map((msg, i) => (
              <div
                key={i}
                className={`max-w-[85%] rounded-xl px-4 py-3 text-sm shadow ${
                  msg.role === "user"
                    ? "ml-auto bg-cyan-700/25 ring-1 ring-cyan-600/30"
                    : "bg-white/5 ring-1 ring-white/10"
                }`}
              >
                {/* Allow lightweight markdown headings/bullets */}
                <div className="[&_strong]:font-semibold [&_li]:ml-4 [&_li]:list-disc [&_ul]:space-y-1 whitespace-pre-wrap">
                  {msg.content}
                </div>
              </div>
            ))}
          </div>

          {/* Composer */}
          <div className="border-t border-white/10 p-3">
            <div className="flex items-center gap-2">
              {/* Mic: press & hold */}
              <button
                onMouseDown={handleMicDown}
                onMouseUp={handleMicUp}
                onTouchStart={handleMicDown}
                onTouchEnd={handleMicUp}
                className={`inline-flex h-10 w-10 items-center justify-center rounded-full ring-1 transition ${
                  recording
                    ? "bg-rose-600 ring-rose-500"
                    : "bg-white/5 ring-white/10 hover:bg-white/10"
                }`}
                title="Hold to speak"
              >
                <Mic className="h-5 w-5" />
              </button>

              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder='Try: "analyze Fram Strait"'
                className="flex-1 rounded-xl bg-white/5 px-4 py-2 text-sm outline-none ring-1 ring-white/10 placeholder:text-white/30 focus:ring-2 focus:ring-cyan-500/40"
              />

              <button
                onClick={handleSend}
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 px-4 py-2 text-sm font-medium shadow-lg shadow-cyan-900/20 hover:brightness-110"
              >
                <Send className="h-4 w-4" />
                Send
              </button>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

/* ---------------- UI bits ---------------- */

function StatCard({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <div
      className={`rounded-xl border bg-gradient-to-br ${color} p-4 ring-1 ring-white/10`}
    >
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-wider text-white/60">
          {label}
        </span>
        {icon}
      </div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
    </div>
  );
}
