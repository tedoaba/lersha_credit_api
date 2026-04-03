import { Badge } from "@/components/ui/badge";
import type { JobStatus } from "@/lib/types";

interface JobStatusBadgeProps {
  status: JobStatus;
}

const STATUS_CONFIG: Record<JobStatus, { label: string; variant: "default" | "secondary" | "destructive" | "outline"; className: string }> = {
  pending: {
    label: "Pending",
    variant: "secondary",
    className: "bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-200",
  },
  processing: {
    label: "Processing…",
    variant: "default",
    className: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 animate-pulse",
  },
  completed: {
    label: "Completed",
    variant: "default",
    className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  },
  failed: {
    label: "Failed",
    variant: "destructive",
    className: "",
  },
};

export default function JobStatusBadge({ status }: JobStatusBadgeProps) {
  const config = STATUS_CONFIG[status];
  return (
    <Badge variant={config.variant} className={config.className}>
      {config.label}
    </Badge>
  );
}
