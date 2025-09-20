"use client";

import React, { useMemo, useRef, useState } from "react";
import { Send, Bot, User, BarChart3, Ship, Shield, AlertTriangle, Radar, Waves, Target } from "lucide-react";
import type { AnalyzeResponse } from "@/types";

/* ---------------- Types ---------------- */

type DeepAnalysisResponse = AnalyzeResponse & {
  deepAnalysis?: {
    vesselTypes: { [key: string]: number };
    wavePatterns: { [key: string]: number };
    darkVesselRisk: { high: number; medium: number; low: number };
    confidenceDistribution: { [key: string]: number };
    suspiciousActivity: string[];
  };
};

type ChatItem =
  | { role: "assistant" | "user"; content: string }
  | { role: "analysis"; content: string; data: DeepAnalysisResponse }
  | { role: "deep_analysis"; content: string; data: DeepAnalysisResponse };

/* ---------------- Helpers ---------------- */

/** Convert **bold** to <strong>, and simple Markdown tables to HTML */
function formatText(text: string): string {
  let formatted = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // naive check for a markdown table
  if (formatted.includes("|") && formatted.includes("---")) {
    const lines = formatted.split("\n");
    let tableHtml =
      '<table class="w-full mt-3 mb-3 border-collapse bg-black/20 rounded-lg overflow-hidden"><tbody>';
    let isHeader = true;
    const rows: string[] = [];
    for (const line of lines) {
      if (!line.trim() || line.includes("---")) continue;
      if (line.includes("|")) {
        const cells = line
          .split("|")
          .map((c) => c.trim())
          .filter((c) => c !== "");
        if (cells.length) {
          let row = "<tr>";
          for (const cell of cells) {
            if (isHeader) {
              row += `<th class="border border-white/30 px-3 py-2 bg-white/20 text-left font-bold text-xs text-cyan-200">${cell}</th>`;
            } else {
              row += `<td class="border border-white/20 px-3 py-2 text-xs text-white/90">${cell}</td>`;
            }
          }
          row += "</tr>";
          rows.push(row);
          isHeader = false;
        }
      }
    }
    tableHtml += rows.join("") + "</tbody></table>";

    // replace the first detected table-ish block with HTML table
    const firstBar = formatted.indexOf("|");
    if (firstBar !== -1) {
      const after = formatted.slice(firstBar);
      // end at the first blank line after the table-ish block
      const endIdx = after.indexOf("\n\n");
      const tableBlock = endIdx === -1 ? after : after.slice(0, endIdx);
      formatted =
        formatted.slice(0, firstBar) +
        tableHtml +
        (endIdx === -1 ? "" : after.slice(endIdx));
    }
  }

  return formatted;
}

/* ---------------- Component ---------------- */

export default function GeminiChat() {
  const [input, setInput] = useState("");
  const [items, setItems] = useState<ChatItem[]>([
    {
      role: "assistant",
      content:
        'Ask me anything about Arctic ship detections. Try: <strong>analyze Fram Strait</strong> for regional analysis, <strong>deep analysis Northern Star</strong> for a comprehensive vessel assessment, or <strong>analyze ships in Fram Strait</strong> for detailed vessel listings.',
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
      // IMPORTANT: handle both "northern start" (typo) and "northern star"
      const mentionsNorthernStarOrStart = /northern\s*sta(?:r|rt)\b/i.test(text);

      // If user mentions Northern Star/Start anywhere, DO NOT call Gemini or Analyze.
      if (mentionsNorthernStarOrStart) {
        const predetermined = buildNorthernStarFingerprintReport();
        setItems((prev) => [...prev, { role: "assistant", content: predetermined }]);
        return;
      }

      // --- Otherwise, proceed with your normal logic ---

      const isDeepAnalysis = /(analysis|analyze|deep|complete|full)/i.test(text);

      if (/^(deep\s+)?analyze\b/i.test(text)) {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, deepAnalysis: isDeepAnalysis }),
        });
        const data = (await res.json()) as DeepAnalysisResponse;
        if ((data as any).error) {
          setItems((prev) => [
            ...prev,
            { role: "assistant", content: "Analyze error: " + (data as any).detail },
          ]);
        } else {
          setItems((prev) => [
            ...prev,
            {
              role: isDeepAnalysis ? "deep_analysis" : "analysis",
              content: data.narrative,
              data,
            },
          ]);
        }
        return;
      }

      // region or vessel phrasing (no hardcoded vessel hit)
      const analyzeRegionMatch =
        text.match(/(deep\s+analysis|analyze|show|list).*ships?.*in\s+(.+)/i);

      if (analyzeRegionMatch) {
        const isDeep = /deep\s+analysis/i.test(analyzeRegionMatch[1]);
        const region = analyzeRegionMatch[2].trim();

        let prompt = `Based on the Arctic ship detection data, analyze and list all vessel detections in the "${region}" region.`;

        if (isDeep) {
          prompt += `

Perform DEEP ANALYSIS including:

**Vessel Classification & Risk Assessment:**
- Classify each vessel by type (cargo, tanker, fishing, research, military, unknown)
- Analyze ML confidence scores to assess dark vessel probability
- Generate wave pattern signatures for each detection
- Evaluate suspicious behavior indicators

**Deep Analysis Format:**

1. **Executive Summary** - Overview of maritime activity and threat assessment

2. **Vessel Detection Table:**
| Vessel ID | Type | Confidence | Dark Vessel Risk | Wave Pattern | Engine Signature | Flag | Status |
|---|---|---|---|---|---|---|---|
| DET-001 | Cargo | 87% | LOW | V-Wake Strong | Diesel-2x | RUS | AIS Match |
| DET-002 | Unknown | 34% | HIGH | Minimal Wake | Electric/Hybrid | ??? | Dark Vessel |

3. **Risk Assessment Matrix:**
- HIGH RISK (Confidence <50%): Likely dark vessels, no AIS correlation
- MEDIUM RISK (50-75%): Partial AIS match, unusual patterns
- LOW RISK (>75%): Full AIS correlation, normal operations

4. **Wave Pattern Analysis:**
- V-Wake signatures indicating vessel size/speed
- Kelvin wake angles for speed estimation
- Turbulence patterns for engine type identification

5. **Intelligence Summary:**
- Count of probable dark vessels
- Suspicious activity patterns
- Recommended surveillance priorities`;
        }

        prompt += `

Requirements:
- Use enhanced detection data with ship names, types, confidence levels
- Color-code risk levels in descriptions
- Include technical specifications and wave analysis
- Focus on "${region}" detections
- Present as naval intelligence brief`;

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
        return;
      }

      // regular chat fallback to Gemini
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
                <div
                  className="whitespace-pre-wrap"
                  dangerouslySetInnerHTML={{ __html: formatText(m.content) }}
                />
              </div>
            </div>
          ) : m.role === "assistant" ? (
            <div key={i} className="flex justify-start">
              <div className="max-w-[85%] rounded-xl bg-black/30 px-4 py-3 text-sm shadow">
                <div className="mb-1 flex items-center gap-2 text-white/60">
                  <Bot className="h-4 w-4" />
                  <span className="text-xs">Arctic Intelligence</span>
                </div>
                <div
                  className="whitespace-pre-wrap"
                  dangerouslySetInnerHTML={{ __html: formatText(m.content) }}
                />
              </div>
            </div>
          ) : m.role === "analysis" ? (
            <AnalysisBubble key={i} data={(m as any).data} />
          ) : m.role === "deep_analysis" ? (
            <DeepAnalysisBubble key={i} data={(m as any).data} />
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
            placeholder='Try: "deep analysis Northern Star" for detailed vessel threat assessment'
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

/* ---------------- Subcomponents ---------------- */

function AnalysisBubble({ data }: { data: DeepAnalysisResponse }) {
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

function DeepAnalysisBubble({ data }: { data: DeepAnalysisResponse }) {
  const deepData = data.deepAnalysis;

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-xl bg-gradient-to-br from-red-900/20 to-orange-900/20 border border-red-500/30 px-4 py-3 text-sm shadow-lg">
        <div className="mb-2 flex items-center gap-2 text-red-300">
          <Target className="h-4 w-4" />
          <span className="text-xs font-semibold">DEEP THREAT ANALYSIS</span>
          <div className="ml-auto px-2 py-1 bg-red-900/50 rounded text-xs border border-red-500/50">
            CLASSIFIED
          </div>
        </div>

        <div className="whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: formatText(data.narrative) }} />

        <div className="mt-3 grid grid-cols-1 gap-3 rounded-lg bg-black/20 p-3 sm:grid-cols-2">
          <InfoRow label="Region" value={data.resolvedRegion || "—"} />
          <InfoRow label="Total Contacts" value={String(data.totalDetections)} />
          <InfoRow label="Primary Threat" value={data.bestRegion} />
          <InfoRow label="Activity Level" value={`${data.bestRegionCount} vessels`} />
        </div>

        {deepData?.darkVesselRisk && (
          <div className="mt-3 rounded-lg bg-black/20 p-3">
            <div className="mb-2 flex items-center gap-2 text-red-300">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-xs font-medium">DARK VESSEL RISK MATRIX</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="flex items-center gap-1 text-xs bg-red-900/40 rounded px-2 py-1 border border-red-500/30">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                <span className="text-red-300">High: {deepData.darkVesselRisk.high}</span>
              </div>
              <div className="flex items-center gap-1 text-xs bg-yellow-900/40 rounded px-2 py-1 border border-yellow-500/30">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                <span className="text-yellow-300">Med: {deepData.darkVesselRisk.medium}</span>
              </div>
              <div className="flex items-center gap-1 text-xs bg-green-900/40 rounded px-2 py-1 border border-green-500/30">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-green-300">Low: {deepData.darkVesselRisk.low}</span>
              </div>
            </div>
          </div>
        )}

        <div className="mt-3 rounded-lg bg-black/20 p-3">
          <div className="mb-2 flex items-center gap-2 text-cyan-300">
            <Waves className="h-4 w-4" />
            <span className="text-xs font-medium">WAVE SIGNATURE ANALYSIS</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-black/30 rounded px-2 py-1">
              <span className="text-white/60">V-Wake Patterns:</span>
              <span className="ml-2 text-cyan-300">{deepData?.wavePatterns?.vWake || "N/A"}</span>
            </div>
            <div className="bg-black/30 rounded px-2 py-1">
              <span className="text-white/60">Kelvin Wakes:</span>
              <span className="ml-2 text-cyan-300">{deepData?.wavePatterns?.kelvin || "N/A"}</span>
            </div>
          </div>
        </div>

        <div className="mt-3 rounded-lg bg-black/20 p-3">
          <div className="mb-2 flex items-center gap-2 text-white/60">
            <Ship className="h-4 w-4" />
            <span className="text-xs font-medium">VESSEL TYPE DISTRIBUTION</span>
          </div>
          <div className="space-y-1">
            {Object.entries(deepData?.vesselTypes || {}).map(([type, count]) => (
              <div key={type} className="flex justify-between items-center text-xs">
                <span className="text-white/80 capitalize">{type}</span>
                <span className="text-white/90 font-medium">{count}</span>
              </div>
            ))}
          </div>
        </div>

        {deepData?.suspiciousActivity && deepData.suspiciousActivity.length > 0 && (
          <div className="mt-3 rounded-lg bg-red-900/20 border border-red-500/30 p-3">
            <div className="mb-2 flex items-center gap-2 text-red-300">
              <Radar className="h-4 w-4" />
              <span className="text-xs font-medium">SUSPICIOUS ACTIVITY ALERTS</span>
            </div>
            <ul className="space-y-1 text-xs">
              {deepData.suspiciousActivity.map((activity, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <AlertTriangle className="h-3 w-3 text-red-400 mt-0.5 flex-shrink-0" />
                  <span className="text-red-200">{activity}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-3 rounded-lg bg-orange-900/20 border border-orange-500/30 p-3">
          <div className="text-xs text-orange-200">
            <strong>INTELLIGENCE RECOMMENDATION:</strong> Maintain enhanced surveillance on high-risk contacts.
            Correlate with allied naval intelligence. Priority monitoring required for vessels with &lt;50% confidence ratings.
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

/* ---------------- Hardcoded Northern Star/Start Report ---------------- */

function buildNorthernStarFingerprintReport(): string {
  // Uses only the stats you provided and centers on the wake/fingerprint mismatch
  return `**NORTHERN STAR — FINGERPRINT MISMATCH REPORT (NO LLM CALL)**

**Vessel Summary**
- **Name (queried):** Northern Star / Northern Start
- **Declared Class/Model:** Fishing Vessel (Trawler)
- **Dimensions:** Length **65.8m**, Beam **12.4m**, Draft **5.2m**
- **Tonnage:** **1,850 GT**
- **Region:** **High Arctic**
- **Location:** **East of Greenland**, approaching **Fram Strait**
- **Jurisdiction:** **International Waters** (near Greenland)

**Wake Pattern Analysis (Observed)**
- **Match Confidence:** **89.1%**
- **Kelvin Angle:** **19.2°**
- **Wake Length:** **~2.3 km**
- **Wake Width:** **~180.0 m**
- **Images:** Image not found • Image not found

**Model Fingerprint vs Observed Signature**
| Parameter | Expected (65.8m Trawler Model) | Observed (Detection) | Assessment |
|---|---|---|---|
| Operational Speed | ~8–12 knots (trawling profile) | ~16–17 knots (derived from 19.2°) | **Excessive** for trawler ops |
| Wake Width | ~75–95 m | **~180 m** | **89–140% wider than expected** |
| Wake Energy Pattern | Low-to-moderate displacement wake | High-energy V-wake with long trail (~2.3 km) | **Inconsistent** with trawler |
| Implied Displacement/Size | Consistent with ~66 m LOA | Consistent with **85–100 m** LOA | **Suggests larger vessel class** |

**Fingerprint Mismatch — Key Points**
1. **Wake Width Oversize:** The ~180 m wake beam is **well beyond** the expected ~75–95 m for a 65.8 m trawler; this alone indicates a hull/propulsion profile larger than declared.
2. **Speed Profile Misfit:** A trawler operating in fishing mode does **not** sustain ~16–17 knots; the Kelvin angle (~19.2°) indicates a **high transit speed**, not fishing behavior.
3. **Energy/Trail Signature:** A **2.3 km** persistent wake with strong V-arms implies **higher power** and **greater displacement** than the trawler model fingerprint.
4. **Size Inference:** Hydrodynamic cues (width, angle, trail) align more closely with **~85–100 m** vessels, not a 65.8 m fishing trawler.

**Assessment**
- The observed hydrodynamic fingerprint is **incompatible** with the declared 65.8 m trawler model.
- Evidence indicates **under-reported size/displacement and/or different propulsion configuration** than a standard marine-diesel trawler.
- **Conclusion:** Strong likelihood of **misclassification or identity obfuscation**. Prioritize verification of registry, recent refits, or identity masking.

**Recommended Next Steps (Technical)**
- Perform multi-pass wake sampling on additional frames (if available) to confirm width/angle stability.
- Cross-validate with any available AIS/ELINT (if permissible) and revisit hull length estimates using wake-inferred speed and beam heuristics.
- Task supplementary SAR/optical collection to resolve hull form and propulsion signatures.
`;
}
