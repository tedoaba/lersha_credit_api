import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import DecisionBadge from "@/components/DecisionBadge";
import ConfidenceGauge from "@/components/ConfidenceGauge";
import type { ResultsRecord } from "@/lib/types";

interface EvaluationCardProps {
  record: ResultsRecord;
}

function formatName(record: ResultsRecord): string {
  const parts = [record.first_name, record.middle_name, record.last_name].filter(Boolean);
  return parts.length > 0 ? parts.join(" ") : record.farmer_uid;
}

export default function EvaluationCard({ record }: EvaluationCardProps) {
  const formattedDate = record.timestamp
    ? new Date(record.timestamp).toLocaleString()
    : "\u2014";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-start justify-between gap-2">
          <span>{formatName(record)}</span>
          <DecisionBadge decision={record.predicted_class_name} confidence={record.confidence_score} />
        </CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted-foreground">Farmer UID</dt>
          <dd className="font-mono text-xs">{record.farmer_uid}</dd>

          <dt className="text-muted-foreground">Model</dt>
          <dd>{record.model_name}</dd>

          <dt className="text-muted-foreground">Confidence</dt>
          <dd><ConfidenceGauge score={record.confidence_score} /></dd>

          <dt className="text-muted-foreground">Scored at</dt>
          <dd>{formattedDate}</dd>
        </dl>
      </CardContent>
    </Card>
  );
}
