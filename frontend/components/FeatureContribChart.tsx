"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { FeatureContribution } from "@/lib/types";

interface FeatureContribChartProps {
  contributions: FeatureContribution[];
}

export default function FeatureContribChart({ contributions }: FeatureContribChartProps) {
  const sorted = [...contributions].sort(
    (a, b) => Math.abs(b.value) - Math.abs(a.value),
  );

  return (
    <div className="w-full">
      <h3 className="text-sm font-medium mb-3 text-muted-foreground uppercase tracking-wide">
        Risk Factor Contributions (SHAP)
      </h3>
      <ResponsiveContainer width="100%" height={Math.max(sorted.length * 36, 120)}>
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
        >
          <XAxis
            type="number"
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={140}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(value: number) => [value.toFixed(4), "SHAP value"]}
          />
          <ReferenceLine x={0} stroke="hsl(var(--border))" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, index) => (
              <Cell
                key={index}
                fill={
                  entry.value >= 0
                    ? "hsl(38 92% 50%)"   // amber — risk-increasing
                    : "hsl(172 66% 40%)"  // teal  — risk-reducing
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-muted-foreground mt-2">
        <span className="text-amber-500 font-medium">■ Amber</span> = increases credit risk &nbsp;|&nbsp;
        <span className="text-teal-600 font-medium">■ Teal</span> = reduces credit risk
      </p>
    </div>
  );
}
