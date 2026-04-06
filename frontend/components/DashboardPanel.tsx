"use client";

import { useMemo } from "react";
import { useAnalytics } from "@/lib/queries";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, AlertTriangle, XCircle, Users } from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Label,
} from "recharts";

const DECISION_COLORS: Record<string, string> = {
  Eligible: "#2d8a4e",
  Review: "#c9a84c",
  "Not Eligible": "#dc3545",
};

const AGREEMENT_COLORS: Record<string, string> = {
  Agreed: "#2d8a4e",
  Mixed: "#94a3b8",
};

const MODEL_COLORS = ["#2d8a4e", "#c9a84c", "#4caf50", "#8b6f3a", "#5a9e6f"];

const SHAP_COLORS = {
  increases_risk: "#e2903a",
  reduces_risk: "#2d8a4e",
};

const CONFIDENCE_COLORS: Record<string, string> = {
  Low: "#dc3545",
  Medium: "#c9a84c",
  High: "#2d8a4e",
};

function formatModelName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function DotLegend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="flex items-center justify-center gap-3 mt-1">
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1 text-[10px] text-muted-foreground">
          <span
            className="inline-block w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: item.color }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Card className={`transition-shadow duration-200 hover:shadow-lg ${className}`}>
      <CardHeader className="pb-0 pt-3 px-4">
        <CardTitle className="text-xs font-semibold leading-none">{title}</CardTitle>
        <p className="text-[10px] text-muted-foreground leading-snug">{subtitle}</p>
      </CardHeader>
      <CardContent className="px-2 pb-2 pt-1">
        {children}
      </CardContent>
    </Card>
  );
}

function DonutCenterLabel({ viewBox, line1, line2 }: { viewBox?: { cx: number; cy: number }; line1: string; line2: string }) {
  if (!viewBox) return null;
  const { cx, cy } = viewBox;
  return (
    <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central">
      <tspan x={cx} dy="-0.4em" className="fill-foreground text-lg font-bold">{line1}</tspan>
      <tspan x={cx} dy="1.3em" className="fill-muted-foreground text-[9px]">{line2}</tspan>
    </text>
  );
}

export default function DashboardPanel() {
  const { data: analytics, isLoading, error } = useAnalytics();

  const byModel = analytics?.by_model ?? {};
  const modelNames = Object.keys(byModel).sort();

  const byConsensus = analytics?.by_consensus ?? {};
  const totalFarmers = analytics?.total_farmers ?? 0;
  const consensusEligible = byConsensus["Eligible"] ?? 0;
  const consensusReview = byConsensus["Review"] ?? 0;
  const consensusNotEligible = byConsensus["Not Eligible"] ?? 0;
  const consensusMixed = byConsensus["Mixed"] ?? 0;
  const pieData = [
    { name: "Eligible", value: consensusEligible },
    { name: "Review", value: consensusReview },
    { name: "Not Eligible", value: consensusNotEligible },
  ].filter((d) => d.value > 0);

  const agreementCount = totalFarmers - consensusMixed;
  const agreementRate = totalFarmers > 0 ? Math.round((agreementCount / totalFarmers) * 100) : 0;
  const agreementPieData = [
    { name: "Agreed", value: agreementCount },
    { name: "Mixed", value: consensusMixed },
  ].filter((d) => d.value > 0);

  const decisions = ["Eligible", "Review", "Not Eligible"];
  const comparisonData = decisions.map((d) => {
    const row: Record<string, string | number> = { decision: d };
    for (const m of modelNames) {
      row[formatModelName(m)] = byModel[m]?.[d] ?? 0;
    }
    return row;
  });

  const modelStats = modelNames.map((m) => {
    const d = byModel[m] ?? {};
    return {
      label: formatModelName(m),
      eligible: d["Eligible"] ?? 0,
      review: d["Review"] ?? 0,
      notEligible: d["Not Eligible"] ?? 0,
    };
  });

  const byGender = analytics?.by_gender ?? {};
  const genderData = Object.entries(byGender).map(([gender, decs]) => ({
    gender,
    Eligible: decs["Eligible"] ?? 0,
    Review: decs["Review"] ?? 0,
    "Not Eligible": decs["Not Eligible"] ?? 0,
  }));

  const confidenceData = analytics?.confidence_distribution ?? [];

  const topRiskFactors = useMemo(() => {
    const factors = analytics?.top_risk_factors ?? [];
    return factors.map((f) => ({
      ...f,
      feature: formatFeatureName(f.feature),
      value: f.direction === "reduces_risk" ? -f.mean_abs_shap : f.mean_abs_shap,
    }));
  }, [analytics?.top_risk_factors]);

  const topEligiblePct = totalFarmers > 0 ? Math.round((consensusEligible / totalFarmers) * 100) : 0;
  const shapSlice = topRiskFactors.slice(0, 12);

  return (
    <div className="space-y-4">
      {isLoading && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="pt-4 pb-4">
                <div className="h-6 bg-muted rounded animate-pulse" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          Unable to load analytics. Check the backend connection.
        </div>
      )}

      {analytics && modelNames.length > 0 && (
        <>
          {/* ─── KPIs ──────────────────────────────────────────────────────── */}
          <div>
            <h2 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-2">
              Overview
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <KpiTile
                label="Farmers Scored"
                value={totalFarmers}
                icon={<Users className="h-3.5 w-3.5 text-muted-foreground" />}
                colorClass="text-foreground"
              />
              <KpiTile
                label="Eligible"
                value={consensusEligible}
                subtitle={totalFarmers > 0 ? `${((consensusEligible / totalFarmers) * 100).toFixed(1)}%` : undefined}
                details={modelStats.map((m) => `${m.label}: ${m.eligible}`)}
                icon={<CheckCircle className="h-3.5 w-3.5 text-emerald-600" />}
                colorClass="text-emerald-600 dark:text-emerald-400"
              />
              <KpiTile
                label="Review"
                value={consensusReview}
                subtitle={totalFarmers > 0 ? `${((consensusReview / totalFarmers) * 100).toFixed(1)}%` : undefined}
                details={modelStats.map((m) => `${m.label}: ${m.review}`)}
                icon={<AlertTriangle className="h-3.5 w-3.5 text-amber-500" />}
                colorClass="text-amber-600 dark:text-amber-400"
              />
              <KpiTile
                label="Not Eligible"
                value={consensusNotEligible}
                subtitle={totalFarmers > 0 ? `${((consensusNotEligible / totalFarmers) * 100).toFixed(1)}%` : undefined}
                details={modelStats.map((m) => `${m.label}: ${m.notEligible}`)}
                icon={<XCircle className="h-3.5 w-3.5 text-red-500" />}
                colorClass="text-red-600 dark:text-red-400"
              />
            </div>
          </div>

          {/* ─── Analytics — 3-col grid, SHAP spans 2 rows in col 3 ─────── */}
          <div>
            <h2 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-2">
              Analytics
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-[0.8fr_1.2fr_1.5fr] gap-2.5">

              {/* Row 1, Col 1 — Consensus */}
              <ChartCard
                title="Farmer Consensus"
                subtitle={`Decision across ${modelNames.map(formatModelName).join(" & ")}`}
              >
                {pieData.length === 0 ? (
                  <p className="text-[10px] text-muted-foreground text-center py-4">No data yet</p>
                ) : (
                  <>
                    <ResponsiveContainer width="100%" height={150}>
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={38}
                          outerRadius={62}
                          dataKey="value"
                          paddingAngle={2}
                          stroke="none"
                        >
                          {pieData.map((entry) => (
                            <Cell key={entry.name} fill={DECISION_COLORS[entry.name] ?? "#94a3b8"} />
                          ))}
                          <Label content={<DonutCenterLabel line1={`${topEligiblePct}%`} line2="eligible" />} position="center" />
                        </Pie>
                        <Tooltip cursor={false} formatter={(value: number, name: string) => [`${value} farmer${value !== 1 ? "s" : ""}`, name]} />
                      </PieChart>
                    </ResponsiveContainer>
                    <DotLegend items={pieData.map((d) => ({ label: d.name, color: DECISION_COLORS[d.name] ?? "#94a3b8" }))} />
                  </>
                )}
              </ChartCard>

              {/* Row 1, Col 2 — Gender */}
              <ChartCard title="Gender Breakdown" subtitle="Decision distribution by gender">
                {genderData.length === 0 ? (
                  <p className="text-[10px] text-muted-foreground text-center py-4">No data yet</p>
                ) : (
                  <>
                    <ResponsiveContainer width="100%" height={150}>
                      <BarChart data={genderData} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="gender" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 9 }} width={28} axisLine={false} tickLine={false} />
                        <Tooltip cursor={{ fill: "rgba(0,0,0,0.04)" }} />
                        <Bar dataKey="Eligible" fill={DECISION_COLORS["Eligible"]} stackId="a" />
                        <Bar dataKey="Review" fill={DECISION_COLORS["Review"]} stackId="a" />
                        <Bar dataKey="Not Eligible" fill={DECISION_COLORS["Not Eligible"]} stackId="a" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                    <DotLegend items={decisions.map((d) => ({ label: d, color: DECISION_COLORS[d] }))} />
                  </>
                )}
              </ChartCard>

              {/* Row 1-2, Col 3 — SHAP (spans 2 rows) */}
              <ChartCard
                title="Top Risk Factors"
                subtitle="SHAP feature impact on credit decisions"
                className="xl:row-span-3"
              >
                {topRiskFactors.length === 0 ? (
                  <p className="text-[10px] text-muted-foreground text-center py-4">No data yet</p>
                ) : (
                  <>
                    <ResponsiveContainer width="100%" height={shapSlice.length * 44 + 50}>
                      <BarChart
                        data={shapSlice}
                        layout="vertical"
                        margin={{ left: 0, right: 20, top: 5, bottom: 15 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f0f0f0" />
                        <XAxis
                          type="number"
                          tick={{ fontSize: 10 }}
                          axisLine={false}
                          tickLine={false}
                          label={{ value: "Mean |SHAP value|", position: "bottom", offset: -2, style: { fontSize: 10, fill: "#888" } }}
                        />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          tick={{ fontSize: 11 }}
                          width={130}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          cursor={{ fill: "rgba(0,0,0,0.04)" }}
                          formatter={(value: number) => [
                            `${Math.abs(value).toFixed(4)}`,
                            value > 0 ? "Increases risk" : "Reduces risk",
                          ]}
                        />
                        <ReferenceLine x={0} stroke="#d4d4d8" />
                        <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                          {shapSlice.map((entry) => (
                            <Cell
                              key={entry.feature}
                              fill={entry.value > 0 ? SHAP_COLORS.increases_risk : SHAP_COLORS.reduces_risk}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    <DotLegend items={[
                      { label: "Increases risk", color: SHAP_COLORS.increases_risk },
                      { label: "Reduces risk", color: SHAP_COLORS.reduces_risk },
                    ]} />
                  </>
                )}
              </ChartCard>

              {/* Row 2, Col 1 — Agreement */}
              <ChartCard title="Model Agreement" subtitle="How often all models agree">
                <ResponsiveContainer width="100%" height={150}>
                  <PieChart>
                    <Pie
                      data={agreementPieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={38}
                      outerRadius={62}
                      dataKey="value"
                      paddingAngle={2}
                      stroke="none"
                    >
                      {agreementPieData.map((entry) => (
                        <Cell key={entry.name} fill={AGREEMENT_COLORS[entry.name] ?? "#94a3b8"} />
                      ))}
                      <Label content={<DonutCenterLabel line1={`${agreementRate}%`} line2="agreed" />} position="center" />
                    </Pie>
                    <Tooltip cursor={false} formatter={(value: number, name: string) => [`${value} farmer${value !== 1 ? "s" : ""}`, name]} />
                  </PieChart>
                </ResponsiveContainer>
                <DotLegend items={[{ label: "Agreed", color: AGREEMENT_COLORS.Agreed }, { label: "Mixed", color: AGREEMENT_COLORS.Mixed }]} />
              </ChartCard>

              {/* Row 2, Col 2 — Confidence */}
              <ChartCard title="Confidence Distribution" subtitle="Model prediction certainty">
                {confidenceData.length === 0 ? (
                  <p className="text-[10px] text-muted-foreground text-center py-4">No data yet</p>
                ) : (
                  <>
                    <ResponsiveContainer width="100%" height={150}>
                      <BarChart data={confidenceData} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="range" tick={{ fontSize: 8 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 9 }} width={24} axisLine={false} tickLine={false} />
                        <Tooltip cursor={{ fill: "rgba(0,0,0,0.04)" }} formatter={(value: number) => [`${value}`, "Count"]} />
                        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                          {confidenceData.map((entry, i) => {
                            const ratio = i / Math.max(confidenceData.length - 1, 1);
                            const color = ratio < 0.4 ? CONFIDENCE_COLORS.Low : ratio < 0.7 ? CONFIDENCE_COLORS.Medium : CONFIDENCE_COLORS.High;
                            return <Cell key={entry.range} fill={color} />;
                          })}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    <DotLegend items={Object.entries(CONFIDENCE_COLORS).map(([l, c]) => ({ label: l, color: c }))} />
                  </>
                )}
              </ChartCard>

              {/* Row 3, Col 1-2 — Model Comparison */}
              <ChartCard title="Model Comparison" subtitle="Side-by-side decisions per model" className="sm:col-span-2 xl:col-span-2">
                <ResponsiveContainer width="100%" height={150}>
                  <BarChart data={comparisonData} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                    <XAxis dataKey="decision" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 9 }} width={24} axisLine={false} tickLine={false} />
                    <Tooltip cursor={{ fill: "rgba(0,0,0,0.04)" }} />
                    {modelNames.map((m, i) => (
                      <Bar key={m} dataKey={formatModelName(m)} fill={MODEL_COLORS[i % MODEL_COLORS.length]} radius={[3, 3, 0, 0]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
                <DotLegend items={modelNames.map((m, i) => ({ label: formatModelName(m), color: MODEL_COLORS[i % MODEL_COLORS.length] }))} />
              </ChartCard>
            </div>
          </div>
        </>
      )}

      {analytics && modelNames.length === 0 && (
        <div className="rounded-md border border-dashed p-8 text-center text-muted-foreground text-sm">
          No scored records yet. Submit a prediction to get started.
        </div>
      )}
    </div>
  );
}

function KpiTile({
  label,
  value,
  subtitle,
  details,
  icon,
  colorClass,
}: {
  label: string;
  value: number | string;
  subtitle?: string;
  details?: string[];
  icon: React.ReactNode;
  colorClass: string;
}) {
  return (
    <Card className="transition-shadow duration-200 hover:shadow-md">
      <CardHeader className="flex flex-row items-center justify-between pb-0 pt-2.5 px-3">
        <CardTitle className="text-xs font-bold text-muted-foreground uppercase tracking-widest">
          {label}
        </CardTitle>
        {icon}
      </CardHeader>
      <CardContent className="px-3 pb-2.5 pt-0">
        <p className={`text-2xl font-bold tabular-nums leading-tight ${colorClass}`}>{value}</p>
        {subtitle && (
          <p className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</p>
        )}
        {details && details.length > 0 && (
          <div className="mt-1 space-y-0">
            {details.map((d) => (
              <p key={d} className="text-[10px] text-muted-foreground leading-relaxed">{d}</p>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
