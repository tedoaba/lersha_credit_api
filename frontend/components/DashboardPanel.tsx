"use client";

import { useAnalytics } from "@/lib/queries";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, AlertTriangle, XCircle, Users, HelpCircle } from "lucide-react";
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
} from "recharts";

const DECISION_COLORS: Record<string, string> = {
  Eligible: "#10b981",
  Review: "#f59e0b",
  "Not Eligible": "#ef4444",
  Mixed: "#8b5cf6",
};

const MODEL_COLORS = ["#3b82f6", "#f97316", "#8b5cf6", "#06b6d4", "#ec4899"];

function formatModelName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
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

  return (
    <div className="space-y-8">
      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {[...Array(5)].map((_, i) => (
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
          {/* KPI tiles — farmer-level consensus with per-model breakdown */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
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
            <KpiTile
              label="Mixed"
              value={consensusMixed}
              subtitle={totalFarmers > 0 && consensusMixed > 0 ? `${((consensusMixed / totalFarmers) * 100).toFixed(1)}%` : undefined}
              icon={<HelpCircle className="h-4 w-4 text-purple-500" />}
              colorClass="text-purple-600 dark:text-purple-400"
            />
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Consensus donut */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Farmer Consensus</CardTitle>
              </CardHeader>
              <CardContent>
                {pieData.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">No data yet</p>
                ) : (
                  <ResponsiveContainer width="100%" height={280}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
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
                        formatter={(value: number, name: string) => [
                          `${value} farmer${value !== 1 ? "s" : ""}`,
                          name,
                        ]}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            {/* Model comparison */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Model Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="decision" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
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

            {/* Gender breakdown */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Gender Breakdown</CardTitle>
              </CardHeader>
              <CardContent>
                {genderData.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">No data yet</p>
                ) : (
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={genderData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="gender" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="Eligible" fill="#10b981" stackId="a" />
                      <Bar dataKey="Review" fill="#f59e0b" stackId="a" />
                      <Bar dataKey="Not Eligible" fill="#ef4444" stackId="a" />
                    </BarChart>
                  </ResponsiveContainer>
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
  value: number;
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
