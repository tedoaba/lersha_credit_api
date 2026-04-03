import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { ExplainResponse } from "@/lib/types";

interface ExplanationPanelProps {
  explanation: ExplainResponse | undefined;
  isLoading: boolean;
  error?: string;
}

export default function ExplanationPanel({
  explanation,
  isLoading,
  error,
}: ExplanationPanelProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          AI Explanation
          {explanation?.cache_hit && (
            <Badge variant="outline" className="text-xs">Cached</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading && (
          <p className="text-sm text-muted-foreground animate-pulse">Generating explanation…</p>
        )}

        {error && (
          <p className="text-sm text-muted-foreground italic">
            Explanation unavailable — {error}
          </p>
        )}

        {explanation && (
          <>
            <p className="text-sm leading-relaxed">{explanation.explanation}</p>

            <div className="flex flex-wrap gap-2 pt-1">
              <span className="text-xs text-muted-foreground">Source docs:</span>
              {explanation.retrieved_doc_ids.map((id) => (
                <Badge key={id} variant="secondary" className="text-xs">
                  #{id}
                </Badge>
              ))}
            </div>

            <dl className="grid grid-cols-3 gap-x-4 gap-y-0.5 text-xs text-muted-foreground border-t pt-2">
              <dt>Prompt version</dt>
              <dd className="col-span-2 font-mono">{explanation.prompt_version}</dd>

              <dt>Latency</dt>
              <dd className="col-span-2">{explanation.latency_ms} ms</dd>

              <dt>Prediction</dt>
              <dd className="col-span-2 font-medium text-foreground">{explanation.prediction}</dd>
            </dl>
          </>
        )}
      </CardContent>
    </Card>
  );
}
