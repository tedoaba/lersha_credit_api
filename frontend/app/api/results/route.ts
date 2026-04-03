import { NextResponse } from "next/server";

const API_BASE = process.env.API_BASE_URL ?? "http://localhost:8000";
const API_KEY = process.env.API_KEY ?? "";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") ?? "500";
  const model_name = searchParams.get("model_name");

  const params = new URLSearchParams({ limit });
  if (model_name) params.set("model_name", model_name);

  const res = await fetch(`${API_BASE}/v1/results/?${params}`, {
    headers: { "X-API-Key": API_KEY },
    cache: "no-store",
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
