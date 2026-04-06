"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import DecisionBadge from "@/components/DecisionBadge";
import ConfidenceGauge from "@/components/ConfidenceGauge";
import ProbabilityBreakdown from "@/components/ProbabilityBreakdown";
import FeatureContribChart from "@/components/FeatureContribChart";
import type { GroupedFarmer, ResultsRecord } from "@/lib/types";

interface FarmerDetailDrawerProps {
  farmer: GroupedFarmer | null;
  modelName?: string | null;
  open: boolean;
  onClose: () => void;
}

function formatName(farmer: GroupedFarmer): string {
  const parts = [farmer.first_name, farmer.middle_name, farmer.last_name].filter(Boolean);
  return parts.length > 0 ? parts.join(" ") : farmer.farmer_uid;
}

function formatModelName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function FarmerDetailDrawer({ farmer, modelName, open, onClose }: FarmerDetailDrawerProps) {
  if (!farmer) return null;

  const record = modelName
    ? farmer.models.find((m) => m.model_name === modelName) ?? farmer.models[0]
    : farmer.models[0];

  const formattedDate = farmer.timestamp
    ? new Date(farmer.timestamp).toLocaleString()
    : "\u2014";

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-3xl lg:max-w-5xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-start justify-between gap-3 pr-6">
            <div>
              <DialogTitle className="text-lg">{formatName(farmer)}</DialogTitle>
              <DialogDescription className="font-mono mt-0.5">
                {farmer.farmer_uid}
              </DialogDescription>
            </div>
            <DecisionBadge decision={record.predicted_class_name} confidence={record.confidence_score} />
          </div>
        </DialogHeader>

        {/* Summary info */}
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-sm">
          <div>
            <dt className="text-muted-foreground text-xs">Gender</dt>
            <dd className="font-medium">{farmer.gender ?? "\u2014"}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Model</dt>
            <dd className="font-medium">{formatModelName(record.model_name)}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Decision</dt>
            <dd className="font-medium">{record.predicted_class_name}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Confidence</dt>
            <dd><ConfidenceGauge score={record.confidence_score} /></dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Scored at</dt>
            <dd className="font-medium">{formattedDate}</dd>
          </div>
        </dl>

        <Separator />

        {/* Probability breakdown */}
        {record.class_probabilities && Object.keys(record.class_probabilities).length > 0 && (
          <div className="rounded-lg border bg-muted/30 p-4">
            <ProbabilityBreakdown probabilities={record.class_probabilities} />
          </div>
        )}

        {/* SHAP chart + RAG explanation */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="min-w-0">
            {record.top_feature_contributions.length > 0 ? (
              <FeatureContribChart contributions={record.top_feature_contributions} />
            ) : (
              <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
                No feature contributions available.
              </div>
            )}
          </div>

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
      </DialogContent>
    </Dialog>
  );
}
