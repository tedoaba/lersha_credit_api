import { NextResponse } from "next/server";

const API_BASE = process.env.API_BASE_URL ?? "http://localhost:8006";
const API_KEY = process.env.API_KEY ?? "";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  const params = new URLSearchParams();
  for (const key of ["q", "limit"]) {
    const val = searchParams.get(key);
    if (val) params.set(key, val);
  }

  const res = await fetch(`${API_BASE}/v1/farmers/search?${params}`, {
    headers: { "X-API-Key": API_KEY },
    cache: "no-store",
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
