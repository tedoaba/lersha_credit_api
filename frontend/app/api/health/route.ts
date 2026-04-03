import { NextResponse } from "next/server";

const API_BASE = process.env.API_BASE_URL ?? "http://localhost:8000";
const API_KEY = process.env.API_KEY ?? "";

export async function GET() {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      headers: { "X-API-Key": API_KEY },
      cache: "no-store",
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ status: "unreachable" }, { status: 503 });
  }
}
