"use client";

import Link from "next/link";
import { useResults } from "@/lib/queries";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ResultsRecord } from "@/lib/types";

function encodeResultId(record: ResultsRecord, index: number): string {
  // We encode job context into the URL via query string on the detail page
  // Here we use the farmer_uid + index as a pseudo-id for the row link
  return encodeURIComponent(`${record.farmer_uid}__${index}__${record.model_name}`);
}

export default function ResultsPage() {
  const { data, isLoading, error } = useResults({ limit: 500 });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Results</h1>
          <p className="text-muted-foreground text-sm mt-1">
            All scored farmer records.
          </p>
        </div>
        {data && (
          <Badge variant="secondary" className="text-sm">
            {data.total} record{data.total !== 1 ? "s" : ""}
          </Badge>
        )}
      </div>

      {isLoading && (
        <div className="text-sm text-muted-foreground animate-pulse">Loading results…</div>
      )}

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          Failed to load results. Check your API key in Settings.
        </div>
      )}

      {data?.total === 0 && (
        <div className="rounded-md border border-dashed p-8 text-center text-muted-foreground text-sm">
          No results yet. Submit a prediction job to get started.
        </div>
      )}

      {data && data.total > 0 && (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Farmer</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Decision</TableHead>
                <TableHead className="hidden sm:table-cell">Scored at</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.records.map((record, i) => {
                const name = [record.first_name, record.last_name]
                  .filter(Boolean)
                  .join(" ") || record.farmer_uid;
                const isEligible = record.predicted_class_name === "Eligible";

                return (
                  <TableRow key={`${record.farmer_uid}-${i}`} className="cursor-pointer hover:bg-muted/50">
                    <TableCell className="font-medium">{name}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">{record.model_name}</TableCell>
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
                    <TableCell className="hidden sm:table-cell text-xs text-muted-foreground">
                      {record.timestamp
                        ? new Date(record.timestamp).toLocaleString()
                        : "—"}
                    </TableCell>
                    <TableCell>
                      <Link
                        href={`/results/${encodeResultId(record, i)}`}
                        className="text-xs text-primary hover:underline"
                      >
                        View →
                      </Link>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
