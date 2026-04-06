"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";
import DecisionBadge from "@/components/DecisionBadge";
import FeatureContribChart from "@/components/FeatureContribChart";
import type { ResultsRecord } from "@/lib/types";

interface FarmerDetailDrawerProps {
  record: ResultsRecord | null;
  open: boolean;
  onClose: () => void;
}

function formatName(record: ResultsRecord): string {
  const parts = [record.first_name, record.middle_name, record.last_name].filter(Boolean);
  return parts.length > 0 ? parts.join(" ") : record.farmer_uid;
}

export default function FarmerDetailDrawer({ record, open, onClose }: FarmerDetailDrawerProps) {
  if (!record) return null;

  const formattedDate = record.timestamp
    ? new Date(record.timestamp).toLocaleString()
    : "\u2014";

  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent className="w-full sm:max-w-2xl lg:max-w-3xl overflow-y-auto">
        <SheetHeader className="pb-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <SheetTitle className="text-lg">{formatName(record)}</SheetTitle>
              <p className="text-sm text-muted-foreground font-mono mt-0.5">
                {record.farmer_uid}
              </p>
            </div>
            <DecisionBadge decision={record.predicted_class_name} />
          </div>
        </SheetHeader>

        {/* Summary info */}
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-sm mb-6">
          <div>
            <dt className="text-muted-foreground text-xs">Gender</dt>
            <dd className="font-medium">{record.gender ?? "\u2014"}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Model</dt>
            <dd className="font-medium">{record.model_name}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Decision</dt>
            <dd className="font-medium">{record.predicted_class_name}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Scored at</dt>
            <dd className="font-medium">{formattedDate}</dd>
          </div>
        </dl>

        <Separator className="mb-6" />

        {/* Two-column: SHAP chart + RAG explanation */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* SHAP Chart */}
          <div className="min-w-0">
            {record.top_feature_contributions.length > 0 ? (
              <FeatureContribChart contributions={record.top_feature_contributions} />
            ) : (
              <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
                No feature contributions available.
              </div>
            )}
          </div>

          {/* RAG Explanation */}
          <div>
            <h3 className="text-sm font-medium mb-3 text-muted-foreground uppercase tracking-wide">
              AI Explanation
            </h3>
            <div className="rounded-lg border bg-muted/30 p-4">
              <p className="text-sm leading-relaxed whitespace-pre-line">
                {record.rag_explanation || "No explanation available."}
              </p>
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
