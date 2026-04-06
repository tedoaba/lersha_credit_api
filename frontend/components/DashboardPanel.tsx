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
  Legend,
  CartesianGrid,
  ReferenceLine,
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

/** Recharts custom legend renderer for pie charts with color mapping */
function renderPieLegend(colorMap: Record<string, string>) {
  return function PieLegend({ payload }: { payload?: Array<{ value: string }> }) {
    if (!payload) return null;
    return (
      <div className="flex items-center justify-center gap-4 mt-2">
        {payload.map((entry) => (
          <span key={entry.value} className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span
              className="inline-block w-3 h-3 rounded-full shrink-0"
              style={{ backgroundColor: colorMap[entry.value] ?? "#94a3b8" }}
            />
            {entry.value}
          </span>
        ))}
      </div>
    );
  };
}

export default function DashboardPanel() {
  const { data: analytics, isLoading, error } = useAnalytics();

  const byModel = analytics?.by_model ?? {};
  const modelNames = Object.keys(byModel).sort();

  // Consensus = farmer-level (deduplicated, models agree or "Mixed")
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

  // Model agreement rate
  const agreementCount = totalFarmers - consensusMixed;
  const agreementPieData = [
    { name: "Agreed", value: agreementCount },
    { name: "Mixed", value: consensusMixed },
  ].filter((d) => d.value > 0);

  // Model comparison data
  const decisions = ["Eligible", "Review", "Not Eligible"];
  const comparisonData = decisions.map((d) => {
    const row: Record<string, string | number> = { decision: d };
    for (const m of modelNames) {
      row[formatModelName(m)] = byModel[m]?.[d] ?? 0;
    }
    return row;
  });

  // Per-model stats for KPI detail lines
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

  // Confidence distribution
  const confidenceData = analytics?.confidence_distribution ?? [];

  // Top risk factors
  const topRiskFactors = useMemo(() => {
    const factors = analytics?.top_risk_factors ?? [];
    return factors.map((f) => ({
      ...f,
      feature: formatFeatureName(f.feature),
      value: f.direction === "reduces_risk" ? -f.mean_abs_shap : f.mean_abs_shap,
    }));
  }, [analytics?.top_risk_factors]);

  const DecisionLegend = useMemo(() => renderPieLegend(DECISION_COLORS), []);
  const AgreementLegend = useMemo(() => renderPieLegend(AGREEMENT_COLORS), []);

  return (
    <div className="space-y-6">
      {isLoading && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="pt-6">
                <div className="h-8 bg-muted rounded animate-pulse" />
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
          {/* ─── Chapter 1: At a Glance ─────────────────────────────────── */}
          <div>
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-4">
              Overview
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <KpiTile
                label="Farmers Scored"
                value={totalFarmers}
                icon={<Users className="h-4 w-4 text-muted-foreground" />}
                colorClass="text-foreground"
              />
              <KpiTile
                label="Eligible"
                value={consensusEligible}
                subtitle={totalFarmers > 0 ? `${((consensusEligible / totalFarmers) * 100).toFixed(1)}%` : undefined}
                details={modelStats.map((m) => `${m.label}: ${m.eligible}`)}
                icon={<CheckCircle className="h-4 w-4 text-emerald-600" />}
                colorClass="text-emerald-600 dark:text-emerald-400"
              />
              <KpiTile
                label="Review"
                value={consensusReview}
                subtitle={totalFarmers > 0 ? `${((consensusReview / totalFarmers) * 100).toFixed(1)}%` : undefined}
                details={modelStats.map((m) => `${m.label}: ${m.review}`)}
                icon={<AlertTriangle className="h-4 w-4 text-amber-500" />}
                colorClass="text-amber-600 dark:text-amber-400"
              />
              <KpiTile
                label="Not Eligible"
                value={consensusNotEligible}
                subtitle={totalFarmers > 0 ? `${((consensusNotEligible / totalFarmers) * 100).toFixed(1)}%` : undefined}
                details={modelStats.map((m) => `${m.label}: ${m.notEligible}`)}
                icon={<XCircle className="h-4 w-4 text-red-500" />}
                colorClass="text-red-600 dark:text-red-400"
              />
            </div>
          </div>

          {/* ─── Chapter 2: Decision Landscape ──────────────────────────── */}
          <div>
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-4">
              Decision Landscape
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Consensus donut — the headline */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Farmer Consensus</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    Agreed decision across {modelNames.map(formatModelName).join(" & ")}
                  </p>
                </CardHeader>
                <CardContent>
                  {pieData.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-8">No data yet</p>
                  ) : (
                    <ResponsiveContainer width="100%" height={220}>
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={85}
                          dataKey="value"
                          paddingAngle={2}
                          label={({ name, percent }) =>
                            `${name} ${(percent * 100).toFixed(0)}%`
                          }
                        >
                          {pieData.map((entry) => (
                            <Cell
                              key={entry.name}
                              fill={DECISION_COLORS[entry.name] ?? "#94a3b8"}
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          cursor={false}
                          formatter={(value: number, name: string) => [
                            `${value} farmer${value !== 1 ? "s" : ""}`,
                            name,
                          ]}
                        />
                        <Legend content={<DecisionLegend />} />
                      </PieChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>

              {/* Gender breakdown — who is being scored */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Gender Breakdown</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    Decision distribution by gender, combined across all models
                  </p>
                </CardHeader>
                <CardContent>
                  {genderData.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-8">No data yet</p>
                  ) : (
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={genderData}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="gender" tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 11 }} label={{ value: "Predictions", angle: -90, position: "insideLeft", style: { fontSize: 11, fill: "#888" } }} />
                        <Tooltip cursor={false} />
                        <Legend />
                        <Bar dataKey="Eligible" fill={DECISION_COLORS["Eligible"]} stackId="a" />
                        <Bar dataKey="Review" fill={DECISION_COLORS["Review"]} stackId="a" />
                        <Bar dataKey="Not Eligible" fill={DECISION_COLORS["Not Eligible"]} stackId="a" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>

          {/* ─── Chapter 3: Model Trust ─────────────────────────────────── */}
          <div>
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-4">
              Model Trust
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Model Agreement */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Model Agreement</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    How often all models agree on a farmer&apos;s decision
                  </p>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie
                        data={agreementPieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={85}
                        dataKey="value"
                        paddingAngle={2}
                        label={({ name, percent }) =>
                          `${name} ${(percent * 100).toFixed(0)}%`
                        }
                      >
                        {agreementPieData.map((entry) => (
                          <Cell key={entry.name} fill={AGREEMENT_COLORS[entry.name] ?? "#94a3b8"} />
                        ))}
                      </Pie>
                      <Tooltip
                        cursor={false}
                        formatter={(value: number, name: string) => [
                          `${value} farmer${value !== 1 ? "s" : ""}`,
                          name,
                        ]}
                      />
                      <Legend content={<AgreementLegend />} />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Confidence Distribution */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Confidence Distribution</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    How certain the models are about their predictions
                  </p>
                </CardHeader>
                <CardContent>
                  {confidenceData.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-8">No data yet</p>
                  ) : (
                    <>
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={confidenceData}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 11 }} label={{ value: "Predictions", angle: -90, position: "insideLeft", style: { fontSize: 11, fill: "#888" } }} />
                          <Tooltip
                            cursor={false}
                            formatter={(value: number) => [`${value} predictions`, "Count"]}
                          />
                          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                            {confidenceData.map((entry, i) => {
                              const ratio = i / Math.max(confidenceData.length - 1, 1);
                              const color = ratio < 0.4 ? CONFIDENCE_COLORS.Low : ratio < 0.7 ? CONFIDENCE_COLORS.Medium : CONFIDENCE_COLORS.High;
                              return <Cell key={entry.range} fill={color} />;
                            })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <div className="flex items-center justify-center gap-4 mt-2">
                        {Object.entries(CONFIDENCE_COLORS).map(([label, color]) => (
                          <span key={label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                            <span className="inline-block w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: color }} />
                            {label}
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              {/* Model Comparison */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Model Comparison</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    Side-by-side decisions per model
                  </p>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={comparisonData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="decision" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} label={{ value: "Count", angle: -90, position: "insideLeft", style: { fontSize: 11, fill: "#888" } }} />
                      <Tooltip cursor={false} />
                      <Legend />
                      {modelNames.map((m, i) => (
                        <Bar
                          key={m}
                          dataKey={formatModelName(m)}
                          fill={MODEL_COLORS[i % MODEL_COLORS.length]}
                        />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* ─── Chapter 4: What Drives Decisions ───────────────────────── */}
          <div>
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-4">
              What Drives Decisions
            </h2>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Top Risk Factors</CardTitle>
                <p className="text-xs text-muted-foreground">
                  Average SHAP feature impact across all predictions — which factors most influence credit decisions
                </p>
              </CardHeader>
              <CardContent>
                {topRiskFactors.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">No data yet</p>
                ) : (
                  <>
                    <ResponsiveContainer width="100%" height={360}>
                      <BarChart
                        data={topRiskFactors.slice(0, 12)}
                        layout="vertical"
                        margin={{ left: 10, right: 20, top: 5, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                        <XAxis
                          type="number"
                          tick={{ fontSize: 11 }}
                          label={{ value: "Mean |SHAP value|", position: "bottom", offset: 10, style: { fontSize: 12, fill: "#888" } }}
                        />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          tick={{ fontSize: 12 }}
                          width={150}
                        />
                        <Tooltip
                          cursor={false}
                          formatter={(value: number) => [
                            `${Math.abs(value).toFixed(4)}`,
                            value > 0 ? "Increases credit risk" : "Reduces credit risk",
                          ]}
                        />
                        <ReferenceLine x={0} stroke="#666" />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                          {topRiskFactors.slice(0, 12).map((entry) => (
                            <Cell
                              key={entry.feature}
                              fill={entry.value > 0 ? SHAP_COLORS.increases_risk : SHAP_COLORS.reduces_risk}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    <div className="flex items-center justify-center gap-6 mt-3">
                      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                        <span className="inline-block w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: SHAP_COLORS.increases_risk }} />
                        Increases credit risk
                      </span>
                      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                        <span className="inline-block w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: SHAP_COLORS.reduces_risk }} />
                        Reduces credit risk
                      </span>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
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
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-1">
        <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          {label}
        </CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <p className={`text-3xl font-bold tabular-nums ${colorClass}`}>{value}</p>
        {subtitle && (
          <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
        )}
        {details && details.length > 0 && (
          <div className="mt-2 space-y-0.5">
            {details.map((d) => (
              <p key={d} className="text-xs text-muted-foreground">{d}</p>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
