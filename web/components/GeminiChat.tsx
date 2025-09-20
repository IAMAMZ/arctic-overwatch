"use client";

import React, { useMemo, useRef, useState } from "react";
import { Send, Bot, User, Map, BarChart3, Ship, Shield, AlertTriangle } from "lucide-react";
import type { AnalyzeResponse } from "@/types";

// Helper function to format text with ** for bold and handle table formatting
function formatText(text: string): string {
  let formatted = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Convert simple table format to HTML table
  if (formatted.includes('|') && formatted.includes('---')) {
    const lines = formatted.split('\n');
    let tableHtml = '<table class="w-full mt-3 mb-3 border-collapse bg-black/20 rounded-lg overflow-hidden"><tbody>';
    let isHeader = true;
    
    for (const line of lines) {
      if (line.trim() === '' || line.includes('---')) continue;
      
      if (line.includes('|')) {
        const cells = line.split('|').map(cell => cell.trim()).filter(cell => cell !== '');
        tableHtml += '<tr>';
        
        for (const cell of cells) {
          if (isHeader) {
            tableHtml += `<th class="border border-white/30 px-3 py-2 bg-white/20 text-left font-bold text-xs text-cyan-200">${cell}</th>`;
          } else {
            tableHtml += `<td class="border border-white/20 px-3 py-2 text-xs text-white/90">${cell}</td>`;
          }
        }
        
        tableHtml += '</tr>';
        isHeader = false;
      }
    }
    
    tableHtml += '</tbody></table>';
    
    // Replace the table part in the original text
    const tableStart = formatted.indexOf('|');
    const tableEnd = formatted.lastIndexOf('|') + formatted.substring(formatted.lastIndexOf('|')).indexOf('\n');
    if (tableStart !== -1) {
      formatted = formatted.substring(0, tableStart) + tableHtml + formatted.substring(tableEnd === -1 ? formatted.length : tableEnd);
    }
  }
  
  return formatted;
}

type ChatItem =
  | { role: "assistant" | "user"; content: string }
  | { role: "analysis"; content: string; data: AnalyzeResponse };

export default function GeminiChat() {
  const [input, setInput] = useState("");
  const [items, setItems] = useState<ChatItem[]>([
    {
      role: "assistant",
      content:
        "Ask me anything about Arctic ship detections. Try: <strong>analyze Fram Strait</strong> for regional analysis or <strong>analyze ships in Fram Strait</strong> for detailed vessel listings with confidence ratings.",
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
        // Check if the user is asking for ship details in a region
        const analyzeRegionMatch = text.match(/analyze.*ships?.*in\s+(.+)/i) || 
                                  text.match(/show.*ships?.*in\s+(.+)/i) ||
                                  text.match(/list.*ships?.*in\s+(.+)/i);
        
        if (analyzeRegionMatch) {
          const region = analyzeRegionMatch[1].trim();
          const prompt = `Based on the Arctic ship detection data, analyze and list all vessel detections in the "${region}" region. 

Format your response with:

1. A brief overview of naval activity in the region
2. A detailed table showing ship detection information:

| Vessel Name | Type | Confidence | Engine | Flag | Dimensions | Status |
|---|---|---|---|---|---|---|
| vessel_name | ship_type | confidence% | engine_type | flag_state | length×beam | AIS_status |

Requirements:
- Use the enhanced detection data that includes ship names, types, confidence levels, engine types, flag states, and dimensions
- Color-code confidence levels in your description (High: 75%+, Medium: 50-74%, Low: <50%)
- Indicate AIS match status (Dark Vessel for no AIS match)
- Include technical specifications like engine type and vessel dimensions
- Focus on ships detected in regions matching or similar to "${region}"
- Mention any vessels of particular interest (military, research, unknown contacts)

Present the information as if you're a naval intelligence analyst reviewing recent Arctic surveillance data.`;

          const res = await fetch("/api/gemini", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: prompt }),
          });
          const data = await res.json();
          setItems((prev) => [
            ...prev,
            { role: "assistant", content: String(data.reply || data.error || "…") },
          ]);
        } else {
          // Regular chat - enhance context for ship-related queries
          let enhancedPrompt = text;
          if (text.toLowerCase().includes('ship') || text.toLowerCase().includes('vessel') || text.toLowerCase().includes('arctic')) {
            enhancedPrompt = `${text}

Context: You have access to Arctic ship detection data that includes:
- Vessel names, types (cargo, fishing, research, military, etc.)
- ML confidence ratings (0-100%)
- Technical specs (engine types, dimensions, flag states)
- AIS matching status (identifying "dark vessels")
- Geographic regions (High Arctic, Arctic Ocean, etc.)

Provide detailed, realistic responses about Arctic maritime activity, ship types, and detection capabilities.`;
          }

          const res = await fetch("/api/gemini", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: enhancedPrompt }),
          });
          const data = await res.json();
          setItems((prev) => [
            ...prev,
            { role: "assistant", content: String(data.reply || data.error || "…") },
          ]);
        }
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
                <div className="whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: formatText(m.content) }} />
              </div>
            </div>
          ) : m.role === "assistant" ? (
            <div key={i} className="flex justify-start">
              <div className="max-w-[85%] rounded-xl bg-black/30 px-4 py-3 text-sm shadow">
                <div className="mb-1 flex items-center gap-2 text-white/60">
                  <Bot className="h-4 w-4" />
                  <span className="text-xs">Arctic Intelligence</span>
                </div>
                <div className="whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: formatText(m.content) }} />
              </div>
            </div>
          ) : m.role === "analysis" ? (
            <AnalysisBubble key={i} data={m.data} />
          ) : null
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
            placeholder='Try: "analyze ships in Fram Strait" or "what is Arctic shipping activity?"'
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
          <span className="text-xs">Regional Analysis</span>
        </div>

        <div className="whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: formatText(data.narrative) }} />

        <div className="mt-3 grid grid-cols-1 gap-3 rounded-lg bg-white/5 p-3 sm:grid-cols-2">
          <InfoRow label="Requested" value={data.requestedRegion || "—"} />
          <InfoRow label="Resolved" value={data.resolvedRegion || "—"} />
          <InfoRow label="Best Region" value={`${data.bestRegion} (${data.bestRegionCount})`} />
          <InfoRow label="Total Detections" value={String(data.totalDetections)} />
        </div>

        <div className="mt-3 rounded-lg bg-white/5 p-3">
          <div className="mb-2 flex items-center gap-2 text-white/60">
            <Ship className="h-4 w-4" />
            <span className="text-xs">Top Regions by Activity</span>
          </div>
          <ul className="space-y-2">
            {data.topRegions.map((r, idx) => (
              <li key={idx} className="flex justify-between items-center text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                  <span className="text-white/80">{r.region}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Ship className="h-3 w-3 text-white/60" />
                  <span className="text-white/90 font-medium">{r.count}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>

        {/* Confidence Status Indicators */}
        <div className="mt-3 grid grid-cols-3 gap-2">
          <div className="flex items-center gap-1 text-xs bg-green-900/30 rounded px-2 py-1">
            <Shield className="h-3 w-3 text-green-400" />
            <span className="text-green-300">High Conf.</span>
          </div>
          <div className="flex items-center gap-1 text-xs bg-yellow-900/30 rounded px-2 py-1">
            <AlertTriangle className="h-3 w-3 text-yellow-400" />
            <span className="text-yellow-300">Med Conf.</span>
          </div>
          <div className="flex items-center gap-1 text-xs bg-red-900/30 rounded px-2 py-1">
            <AlertTriangle className="h-3 w-3 text-red-400" />
            <span className="text-red-300">Low Conf.</span>
          </div>
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