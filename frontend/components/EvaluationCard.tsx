import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { ResultsRecord } from "@/lib/types";

interface EvaluationCardProps {
  record: ResultsRecord;
}

function formatName(record: ResultsRecord): string {
  const parts = [record.first_name, record.middle_name, record.last_name].filter(Boolean);
  return parts.length > 0 ? parts.join(" ") : record.farmer_uid;
}

const CLASS_COLORS: Record<string, string> = {
  Eligible: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  "Not Eligible": "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};

export default function EvaluationCard({ record }: EvaluationCardProps) {
  const colorClass =
    CLASS_COLORS[record.predicted_class_name] ??
    "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200";

  const formattedDate = record.timestamp
    ? new Date(record.timestamp).toLocaleString()
    : "—";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-start justify-between gap-2">
          <span>{formatName(record)}</span>
          <Badge className={colorClass}>{record.predicted_class_name}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted-foreground">Farmer UID</dt>
          <dd className="font-mono text-xs">{record.farmer_uid}</dd>

          <dt className="text-muted-foreground">Model</dt>
          <dd>{record.model_name}</dd>

          <dt className="text-muted-foreground">Scored at</dt>
          <dd>{formattedDate}</dd>
        </dl>
      </CardContent>
    </Card>
  );
}
