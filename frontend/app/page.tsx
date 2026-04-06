"use client";

import Link from "next/link";
import { useAnalytics } from "@/lib/queries";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import DecisionBadge from "@/components/DecisionBadge";
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
} from "recharts";

const DECISION_COLORS: Record<string, string> = {
  Eligible: "#10b981",
  Review: "#f59e0b",
  "Not Eligible": "#ef4444",
};

function formatTimeAgo(timestamp: string | null): string {
  if (!timestamp) return "\u2014";
  const diff = Date.now() - new Date(timestamp).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function DashboardPage() {
  const { data: analytics, isLoading, error } = useAnalytics();

  const byDecision = analytics?.by_decision ?? {};
  const eligible = byDecision["Eligible"] ?? 0;
  const review = byDecision["Review"] ?? 0;
  const notEligible = byDecision["Not Eligible"] ?? 0;
  const total = analytics?.total ?? 0;

  const pieData = [
    { name: "Eligible", value: eligible },
    { name: "Review", value: review },
    { name: "Not Eligible", value: notEligible },
  ].filter((d) => d.value > 0);

  // Build gender bar chart data
  const byGender = analytics?.by_gender ?? {};
  const genderData = Object.entries(byGender).map(([gender, decisions]) => ({
    gender,
    Eligible: decisions["Eligible"] ?? 0,
    Review: decisions["Review"] ?? 0,
    "Not Eligible": decisions["Not Eligible"] ?? 0,
  }));

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Credit scoring analytics overview.
          </p>
        </div>
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
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

      {analytics && (
        <>
          {/* KPI Tiles */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <KpiTile
              label="Total Scored"
              value={total}
              icon={<Users className="h-4 w-4 text-muted-foreground" />}
              colorClass="text-foreground"
            />
            <KpiTile
              label="Eligible"
              value={eligible}
              subtitle={total > 0 ? `${((eligible / total) * 100).toFixed(1)}%` : undefined}
              icon={<CheckCircle className="h-4 w-4 text-emerald-600" />}
              colorClass="text-emerald-600 dark:text-emerald-400"
            />
            <KpiTile
              label="Review"
              value={review}
              subtitle={total > 0 ? `${((review / total) * 100).toFixed(1)}%` : undefined}
              icon={<AlertTriangle className="h-4 w-4 text-amber-500" />}
              colorClass="text-amber-600 dark:text-amber-400"
            />
            <KpiTile
              label="Not Eligible"
              value={notEligible}
              subtitle={total > 0 ? `${((notEligible / total) * 100).toFixed(1)}%` : undefined}
              icon={<XCircle className="h-4 w-4 text-red-500" />}
              colorClass="text-red-600 dark:text-red-400"
            />
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Eligibility Distribution Donut */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Eligibility Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {pieData.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">No data yet</p>
                ) : (
                  <ResponsiveContainer width="100%" height={260}>
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
                          `${value} farmers`,
                          name,
                        ]}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            {/* Gender Breakdown Bar Chart */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Gender Breakdown</CardTitle>
              </CardHeader>
              <CardContent>
                {genderData.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">No data yet</p>
                ) : (
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={genderData}>
                      <XAxis dataKey="gender" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} />
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

          {/* Recent Activity */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold">Recent Activity</h2>
              <Link
                href="/farmers"
                className="text-sm text-primary hover:underline"
              >
                View all farmers
              </Link>
            </div>

            {analytics.recent.length === 0 ? (
              <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
                No scored records yet. Submit a prediction to get started.
              </div>
            ) : (
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Farmer</TableHead>
                      <TableHead className="hidden sm:table-cell">Gender</TableHead>
                      <TableHead>Decision</TableHead>
                      <TableHead className="hidden sm:table-cell">Model</TableHead>
                      <TableHead className="hidden md:table-cell">Scored</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {analytics.recent.map((record, i) => {
                      const name =
                        [record.first_name, record.last_name]
                          .filter(Boolean)
                          .join(" ") || record.farmer_uid;

                      return (
                        <TableRow key={i}>
                          <TableCell className="font-medium">{name}</TableCell>
                          <TableCell className="hidden sm:table-cell text-sm text-muted-foreground">
                            {record.gender ?? "\u2014"}
                          </TableCell>
                          <TableCell>
                            <DecisionBadge decision={record.predicted_class_name} />
                          </TableCell>
                          <TableCell className="hidden sm:table-cell text-xs text-muted-foreground">
                            {record.model_name}
                          </TableCell>
                          <TableCell className="hidden md:table-cell text-xs text-muted-foreground">
                            {formatTimeAgo(record.timestamp)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function KpiTile({
  label,
  value,
  subtitle,
  icon,
  colorClass,
}: {
  label: string;
  value: number;
  subtitle?: string;
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
      </CardContent>
    </Card>
  );
}
