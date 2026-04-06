"use client";

import { useEffect } from "react";
import { useJobStore } from "@/lib/stores";
import { useJobStatus, queryClient } from "@/lib/queries";
import JobStatusBadge from "@/components/JobStatusBadge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Eye } from "lucide-react";
import type { JobStatus } from "@/lib/types";

const PROGRESS_MAP: Record<JobStatus, number> = {
  pending: 15,
  processing: 60,
  completed: 100,
  failed: 100,
};

export default function JobProgressBanner() {
  const { activeJobId, clearActiveJobId, predictionModalOpen, openPredictionModal } = useJobStore();
  const { data: job } = useJobStatus(activeJobId);

  // Handle job completion: refresh data and clear active job
  useEffect(() => {
    if (!activeJobId || !job) return;
    if (job.status === "completed" || job.status === "failed") {
      queryClient.invalidateQueries({ queryKey: ["results-paginated"] });
      queryClient.invalidateQueries({ queryKey: ["analytics"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      // Auto-clear after a short delay so the banner briefly shows "Complete"
      const timer = setTimeout(() => clearActiveJobId(), 3000);
      return () => clearTimeout(timer);
    }
  }, [job?.status, activeJobId, clearActiveJobId]);

  // Don't render if no active job, modal is open, or job is in terminal state without data
  if (!activeJobId || predictionModalOpen || !job) return null;
  if (job.status === "completed" || job.status === "failed") {
    // Show briefly before auto-clear
  }

  const progress = PROGRESS_MAP[job.status];

  return (
    <div className="rounded-lg border bg-card px-4 py-3 flex items-center gap-4">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 mb-1">
          <span className="text-sm font-medium">Job in progress</span>
          <JobStatusBadge status={job.status} />
        </div>
        <div className="flex items-center gap-3">
          <Progress value={progress} className="h-1.5 flex-1" />
          <span className="text-xs font-mono text-muted-foreground truncate max-w-32">
            {activeJobId.slice(0, 12)}...
          </span>
        </div>
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={openPredictionModal}
        className="gap-1.5 shrink-0"
      >
        <Eye className="h-3.5 w-3.5" />
        View
      </Button>
    </div>
  );
}
