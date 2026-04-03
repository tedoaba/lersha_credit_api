import { getServerSideJobSummary } from "@/lib/server-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";

// ISR — revalidate every 60 seconds
export const revalidate = 60;

export default async function DashboardPage() {
  const summary = await getServerSideJobSummary();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground text-sm mt-1">
          System overview — refreshed every 60 seconds.
        </p>
      </div>

      {/* Summary tiles */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryTile
          label="Total Scored"
          value={summary?.total ?? "—"}
          colorClass="text-foreground"
        />
        <SummaryTile
          label="Completed Jobs"
          value={summary?.completed ?? "—"}
          colorClass="text-emerald-600 dark:text-emerald-400"
        />
      </div>

      {/* Recent activity */}
      <div className="space-y-3">
        <h2 className="text-base font-semibold">Recent Activity</h2>

        {!summary && (
          <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
            Unable to load activity. Check the backend connection and API key configuration.
          </div>
        )}

        {summary && summary.recent.length === 0 && (
          <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
            No scored records yet. Submit a prediction to get started.
          </div>
        )}

        {summary && summary.recent.length > 0 && (
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Farmer</TableHead>
                  <TableHead>Decision</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead className="hidden sm:table-cell">Scored at</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {summary.recent.map((record, i) => {
                  const name =
                    [record.first_name, record.last_name].filter(Boolean).join(" ") ||
                    record.farmer_uid;
                  const isEligible = record.predicted_class_name === "Eligible";

                  return (
                    <TableRow key={i}>
                      <TableCell>{name}</TableCell>
                      <TableCell>
                        <Badge
                          className={
                            isEligible
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-red-100 text-red-800"
                          }
                        >
                          {record.predicted_class_name}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {record.model_name}
                      </TableCell>
                      <TableCell className="hidden sm:table-cell text-xs text-muted-foreground">
                        {record.timestamp
                          ? new Date(record.timestamp).toLocaleString()
                          : "—"}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      <div className="flex gap-3">
        <Link
          href="/predict"
          className="text-sm font-medium text-primary hover:underline"
        >
          Submit new prediction →
        </Link>
        <Link
          href="/results"
          className="text-sm font-medium text-primary hover:underline"
        >
          View all results →
        </Link>
      </div>
    </div>
  );
}

function SummaryTile({
  label,
  value,
  colorClass,
}: {
  label: string;
  value: number | string;
  colorClass: string;
}) {
  return (
    <Card>
      <CardHeader className="pb-1">
        <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className={`text-3xl font-bold tabular-nums ${colorClass}`}>{value}</p>
      </CardContent>
    </Card>
  );
}
