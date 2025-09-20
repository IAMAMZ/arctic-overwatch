// web/app/api/analyze/route.ts
import { NextResponse } from "next/server";
import { loadDetections } from "@/lib/detections";
import { REGIONS, resolveRegion } from "@/lib/regions";
import { haversineKm } from "@/lib/haversine";
import type { Detection, AnalyzeResponse } from "@/types";
import { GoogleGenerativeAI } from "@google/generative-ai";

export async function POST(req: Request) {
  try {
    const { message } = await req.json();
    const text = String(message || "").trim();

    // Extract target region (after the word "analyze")
    // Examples: "analyze Fram Strait", "Analyze Northeast Greenland"
    const m = text.match(/analyze\s+(.+)/i);
    const targetName = m?.[1]?.trim() || "";

    // Load detections from CSV
    const detections: Detection[] = await loadDetections();

    // Collate per-region stats
    const perRegion = REGIONS.map((r) => {
      const within = detections.filter((d) => {
        if (!Number.isFinite(d.lat) || !Number.isFinite(d.lon)) return false;
        const dist = haversineKm(d.lat, d.lon, r.centroid.lat, r.centroid.lon);
        return dist <= r.radiusKm;
      });
      return {
        region: r.name,
        count: within.length,
        radiusKm: r.radiusKm,
        centroid: r.centroid,
      };
    }).sort((a, b) => b.count - a.count);

    // Best region by count
    const best = perRegion[0];

    // If user specified a region, find the closest canonical region
    const resolved = targetName ? resolveRegion(targetName) : null;

    // Optionally create a Gemini-phrased summary (fallback to simple text if no key)
    const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    let narrative: string;

    const plain = (regionLabel: string) =>
      `Closest / most active region by current detections: ${best.region} (${best.count} detections within ${best.radiusKm} km).` +
      (regionLabel
        ? `\nRequested region: ${regionLabel}.`
        : "") +
      (resolved
        ? `\nMatched request to canonical region: ${resolved.name}.`
        : "") +
      `\nTop 3 regions: ${perRegion
        .slice(0, 3)
        .map((x) => `${x.region} (${x.count})`)
        .join(", ")}.`;

    if (!apiKey) {
      narrative = plain(resolved?.name || targetName);
    } else {
      const genAI = new GoogleGenerativeAI(apiKey);
      const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
      const prompt = `You are an Arctic maritime analyst. Using the JSON below, write a crisp, neutral 3-4 sentence summary for a product UI.

JSON:
${JSON.stringify(
  {
    requestedRegion: targetName || null,
    resolvedRegion: resolved?.name || null,
    bestRegion: best,
    perRegion: perRegion.slice(0, 5),
    totalDetections: detections.length,
  },
  null,
  2
)}

Guidelines:
- Lead with the best region and count.
- Mention the requested region mapping if present.
- Avoid fluff. Use precise numbers and km units where given.
- Tone: serious, operational.`;

      const res = await model.generateContent(prompt);
      narrative = res.response.text().trim() || plain(resolved?.name || targetName);
    }

    const payload: AnalyzeResponse = {
      requestedRegion: targetName || null,
      resolvedRegion: resolved?.name || null,
      bestRegion: best.region,
      bestRegionCount: best.count,
      topRegions: perRegion.slice(0, 5),
      totalDetections: detections.length,
      narrative,
    };

    return NextResponse.json(payload);
  } catch (err: any) {
    console.error("Analyze error:", err);
    return NextResponse.json(
      { error: "Analyze failed", detail: String(err?.message || err) },
      { status: 500 }
    );
  }
}
