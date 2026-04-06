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
import FarmerAvatar from "@/components/FarmerAvatar";
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
      <DialogContent className="sm:max-w-4xl lg:max-w-6xl max-h-[90vh] overflow-hidden">
        {/* Profile header: Avatar + Identity + Stats */}
        <div className="flex flex-col sm:flex-row gap-6 items-start">
          {/* Avatar */}
          <div className="shrink-0 self-center sm:self-start">
            <FarmerAvatar
              firstName={farmer.first_name}
              middleName={farmer.middle_name}
              uid={farmer.farmer_uid}
              size="lg"
            />
          </div>

          {/* Identity + summary stats */}
          <div className="flex-1 min-w-0 space-y-4">
            <DialogHeader className="p-0">
              <div className="flex items-start justify-between gap-3 pr-8">
                <div>
                  <DialogTitle className="text-xl">{formatName(farmer)}</DialogTitle>
                  <DialogDescription className="font-mono mt-1">
                    {farmer.farmer_uid}
                  </DialogDescription>
                </div>
                <DecisionBadge decision={record.predicted_class_name} confidence={record.confidence_score} />
              </div>
            </DialogHeader>

            {/* Stats row */}
            <dl className="grid grid-cols-2 sm:grid-cols-5 gap-x-6 gap-y-3 text-sm rounded-lg border bg-muted/30 p-4">
              <div>
                <dt className="text-muted-foreground text-xs uppercase tracking-wide">Gender</dt>
                <dd className="font-medium mt-0.5">{farmer.gender ?? "\u2014"}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground text-xs uppercase tracking-wide">Model</dt>
                <dd className="font-medium mt-0.5">{formatModelName(record.model_name)}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground text-xs uppercase tracking-wide">Decision</dt>
                <dd className="font-medium mt-0.5">{record.predicted_class_name}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground text-xs uppercase tracking-wide">Confidence</dt>
                <dd className="mt-0.5"><ConfidenceGauge score={record.confidence_score} /></dd>
              </div>
              <div>
                <dt className="text-muted-foreground text-xs uppercase tracking-wide">Scored at</dt>
                <dd className="font-medium mt-0.5">{formattedDate}</dd>
              </div>
            </dl>
          </div>
        </div>

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

          <div className="flex flex-col min-h-0">
            <h3 className="text-sm font-medium mb-3 text-muted-foreground uppercase tracking-wide shrink-0">
              AI Explanation
            </h3>
            <div className="rounded-lg border bg-muted/30 p-4 overflow-y-auto max-h-80">
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
