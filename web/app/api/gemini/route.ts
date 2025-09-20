// web/app/api/gemini/route.ts
import { NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";

export async function POST(req: Request) {
  try {
    const { message } = await req.json();
    const text = String(message || "").trim();

    const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    if (!apiKey) {
      // Fallback when no API key configured
      return NextResponse.json({
        reply:
          "Gemini API key is not set. Add GOOGLE_GENERATIVE_AI_API_KEY to .env.local.\nEcho: " +
          text,
      });
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });
    const res = await model.generateContent(text);
    const reply = res.response.text().trim();

    return NextResponse.json({ reply });
  } catch (err: any) {
    console.error("Gemini route error:", err);
    return NextResponse.json(
      { error: "Gemini request failed", detail: String(err?.message || err) },
      { status: 500 }
    );
  }
}
