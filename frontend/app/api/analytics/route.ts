import { NextResponse } from "next/server";

const API_BASE = process.env.API_BASE_URL ?? "http://localhost:8006";
const API_KEY = process.env.API_KEY ?? "";

export async function GET() {
  const res = await fetch(`${API_BASE}/v1/analytics/summary`, {
    headers: { "X-API-Key": API_KEY },
    cache: "no-store",
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
