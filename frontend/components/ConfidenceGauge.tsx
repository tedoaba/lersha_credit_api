"use client";

import { cn } from "@/lib/utils";

interface ConfidenceGaugeProps {
  score: number | null;
  className?: string;
}

function getColor(score: number): string {
  if (score >= 0.8) return "bg-emerald-500";
  if (score >= 0.5) return "bg-amber-500";
  return "bg-red-500";
}

function getTextColor(score: number): string {
  if (score >= 0.8) return "text-emerald-700 dark:text-emerald-400";
  if (score >= 0.5) return "text-amber-700 dark:text-amber-400";
  return "text-red-700 dark:text-red-400";
}

export default function ConfidenceGauge({ score, className }: ConfidenceGaugeProps) {
  if (score == null) {
    return <span className="text-xs text-muted-foreground">{"\u2014"}</span>;
  }

  const pct = Math.round(score * 100);

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="relative h-2 w-full max-w-[120px] rounded-full bg-muted overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", getColor(score))}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={cn("text-xs font-semibold tabular-nums", getTextColor(score))}>
        {pct}%
      </span>
    </div>
  );
}
