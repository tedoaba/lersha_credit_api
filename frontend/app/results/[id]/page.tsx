"use client";

import { use } from "react";
import Link from "next/link";
import { useResults } from "@/lib/queries";
import EvaluationCard from "@/components/EvaluationCard";
import FeatureContribChart from "@/components/FeatureContribChart";

interface PageProps {
  params: Promise<{ id: string }>;
}

function parseId(id: string): { farmerUid: string; index: number; modelName: string } | null {
  try {
    const decoded = decodeURIComponent(id);
    const [farmerUid, indexStr, modelName] = decoded.split("__");
    const index = parseInt(indexStr, 10);
    if (!farmerUid || isNaN(index) || !modelName) return null;
    return { farmerUid, index, modelName };
  } catch {
    return null;
  }
}

const BackLink = () => (
  <Link
    href="/results"
    className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
  >
    ← Back to Results
  </Link>
);

export default function ResultDetailPage({ params }: PageProps) {
  const { id } = use(params);
  const parsed = parseId(id);
  const { data: results, isLoading: resultsLoading } = useResults({ limit: 500 });

  const record =
    parsed && results
      ? results.records.find((r, i) => r.farmer_uid === parsed.farmerUid && i === parsed.index)
      : null;

  if (!parsed) {
    return (
      <div className="max-w-xl space-y-4">
        <h1 className="text-2xl font-bold">Result not found</h1>
        <p className="text-muted-foreground text-sm">The result ID in the URL is invalid.</p>
        <BackLink />
      </div>
    );
  }

  if (resultsLoading) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="h-6 w-48 bg-muted rounded" />
        <div className="h-32 bg-muted rounded-lg" />
        <div className="h-48 bg-muted rounded-lg" />
      </div>
    );
  }

  if (!record) {
    return (
      <div className="max-w-xl space-y-4">
        <h1 className="text-2xl font-bold">Result not found</h1>
        <p className="text-muted-foreground text-sm">
          No record found for farmer{" "}
          <code className="font-mono">{parsed.farmerUid}</code>.
        </p>
        <BackLink />
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div className="flex items-center gap-3">
        <BackLink />
        <h1 className="text-2xl font-bold tracking-tight">Farmer Detail</h1>
      </div>

      <EvaluationCard record={record} />

      {record.top_feature_contributions.length > 0 && (
        <div className="rounded-lg border p-5">
          <FeatureContribChart contributions={record.top_feature_contributions} />
        </div>
      )}

      <div className="rounded-lg border p-5 space-y-2">
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
          AI Explanation
        </h3>
        <p className="text-sm leading-relaxed">{record.rag_explanation}</p>
      </div>
    </div>
  );
}
