"use client";

import { cn } from "@/lib/utils";

interface ProbabilityBreakdownProps {
  probabilities: Record<string, number> | null;
  className?: string;
}

const CLASS_COLORS: Record<string, { bar: string; text: string }> = {
  Eligible: {
    bar: "bg-emerald-500",
    text: "text-emerald-700 dark:text-emerald-400",
  },
  Review: {
    bar: "bg-amber-500",
    text: "text-amber-700 dark:text-amber-400",
  },
  Ineligible: {
    bar: "bg-red-500",
    text: "text-red-700 dark:text-red-400",
  },
  "Not Eligible": {
    bar: "bg-red-500",
    text: "text-red-700 dark:text-red-400",
  },
};

const DEFAULT_COLOR = {
  bar: "bg-slate-500",
  text: "text-slate-700 dark:text-slate-400",
};

export default function ProbabilityBreakdown({ probabilities, className }: ProbabilityBreakdownProps) {
  if (!probabilities || Object.keys(probabilities).length === 0) {
    return (
      <div className="text-xs text-muted-foreground">
        No probability data available.
      </div>
    );
  }

  // Sort: Eligible first, then Review, then others
  const order = ["Eligible", "Review", "Ineligible", "Not Eligible"];
  const entries = Object.entries(probabilities).sort(([a], [b]) => {
    const ia = order.indexOf(a);
    const ib = order.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });

  return (
    <div className={cn("space-y-2", className)}>
      <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
        Class Probabilities
      </h4>
      {entries.map(([label, prob]) => {
        const pct = Math.round(prob * 100);
        const colors = CLASS_COLORS[label] ?? DEFAULT_COLOR;
        return (
          <div key={label} className="flex items-center gap-2">
            <span className="text-xs w-20 shrink-0 truncate">{label}</span>
            <div className="relative h-2 flex-1 rounded-full bg-muted overflow-hidden">
              <div
                className={cn("h-full rounded-full transition-all", colors.bar)}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className={cn("text-xs font-semibold tabular-nums w-10 text-right", colors.text)}>
              {pct}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
