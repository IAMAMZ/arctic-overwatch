"use client";

import React, { useMemo, useRef, useState } from "react";
import { Send, Bot, User, Map, BarChart3 } from "lucide-react";
import type { AnalyzeResponse } from "@/types";

type ChatItem =
  | { role: "assistant" | "user"; content: string }
  | { role: "analysis"; content: string; data: AnalyzeResponse };

export default function GeminiChat() {
  const [input, setInput] = useState("");
  const [items, setItems] = useState<ChatItem[]>([
    {
      role: "assistant",
      content:
        "Ask me anything. Try: “analyze Fram Strait” or “analyze Northeast Greenland”.",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  const onSend = async () => {
    const text = input.trim();
    if (!text) return;

    setItems((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      if (/^analyze\b/i.test(text)) {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });
        const data = (await res.json()) as AnalyzeResponse;
        if ((data as any).error) {
          setItems((prev) => [
            ...prev,
            { role: "assistant", content: "Analyze error: " + (data as any).detail },
          ]);
        } else {
          setItems((prev) => [
            ...prev,
            { role: "analysis", content: data.narrative, data },
          ]);
        }
      } else {
        const res = await fetch("/api/gemini", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });
        const data = await res.json();
        setItems((prev) => [
          ...prev,
          { role: "assistant", content: String(data.reply || data.error || "…") },
        ]);
      }
    } catch (err: any) {
      setItems((prev) => [
        ...prev,
        { role: "assistant", content: "Error: " + String(err?.message || err) },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  };

  return (
    <div className="flex h-full w-full flex-col">
      {/* Messages */}
      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {items.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] rounded-xl bg-white/10 px-4 py-3 text-sm shadow">
                <div className="mb-1 flex items-center justify-end gap-2 text-white/70">
                  <span className="text-xs">You</span>
                  <User className="h-4 w-4" />
                </div>
                <div className="whitespace-pre-wrap">{m.content}</div>
              </div>
            </div>
          ) : m.role === "assistant" ? (
            <div key={i} className="flex justify-start">
              <div className="max-w-[85%] rounded-xl bg-black/30 px-4 py-3 text-sm shadow">
                <div className="mb-1 flex items-center gap-2 text-white/60">
                  <Bot className="h-4 w-4" />
                  <span className="text-xs">Assistant</span>
                </div>
                <div className="whitespace-pre-wrap">{m.content}</div>
              </div>
            </div>
          ) : (
            <AnalysisBubble key={i} data={m.data} />
          )
        )}
        <div ref={bottomRef} />
      </div>

      {/* Composer */}
      <div className="border-t border-white/10 p-3">
        <div className="flex items-center gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && canSend && onSend()}
            placeholder='Try: analyze Fram Strait'
            className="flex-1 rounded-xl bg-white/5 px-4 py-3 text-sm outline-none ring-1 ring-white/10 placeholder:text-white/30 focus:ring-2 focus:ring-cyan-500/40"
          />
          <button
            disabled={!canSend}
            onClick={onSend}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 px-4 py-3 text-sm font-medium shadow-lg shadow-cyan-900/20 disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function AnalysisBubble({ data }: { data: AnalyzeResponse }) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-xl bg-black/30 px-4 py-3 text-sm shadow">
        <div className="mb-2 flex items-center gap-2 text-white/60">
          <BarChart3 className="h-4 w-4" />
          <span className="text-xs">Analysis</span>
        </div>

        <div className="whitespace-pre-wrap">{data.narrative}</div>

        <div className="mt-3 grid grid-cols-1 gap-3 rounded-lg bg-white/5 p-3 sm:grid-cols-2">
          <InfoRow label="Requested" value={data.requestedRegion || "—"} />
          <InfoRow label="Resolved" value={data.resolvedRegion || "—"} />
          <InfoRow label="Best Region" value={`${data.bestRegion} (${data.bestRegionCount})`} />
          <InfoRow label="Total Detections" value={String(data.totalDetections)} />
        </div>

        <div className="mt-3 rounded-lg bg-white/5 p-3">
          <div className="mb-2 flex items-center gap-2 text-white/60">
            <Map className="h-4 w-4" />
            <span className="text-xs">Top Regions</span>
          </div>
          <ul className="space-y-1">
            {data.topRegions.map((r, idx) => (
              <li key={idx} className="flex justify-between text-xs text-white/80">
                <span>{r.region}</span>
                <span>{r.count} ships</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-md bg-black/20 px-3 py-2">
      <span className="text-white/60">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
