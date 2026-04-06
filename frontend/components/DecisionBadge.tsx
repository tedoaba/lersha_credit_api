import { Badge } from "@/components/ui/badge";
import { CheckCircle, AlertTriangle, XCircle } from "lucide-react";

const DECISION_CONFIG: Record<string, { color: string; Icon: typeof CheckCircle }> = {
  Eligible: {
    color: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
    Icon: CheckCircle,
  },
  Review: {
    color: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
    Icon: AlertTriangle,
  },
  "Not Eligible": {
    color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    Icon: XCircle,
  },
};

interface DecisionBadgeProps {
  decision: string;
  showIcon?: boolean;
  className?: string;
}

export default function DecisionBadge({ decision, showIcon = true, className = "" }: DecisionBadgeProps) {
  const config = DECISION_CONFIG[decision] ?? {
    color: "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200",
    Icon: AlertTriangle,
  };

  return (
    <Badge className={`${config.color} ${className} gap-1 font-medium`}>
      {showIcon && <config.Icon className="h-3 w-3" />}
      {decision}
    </Badge>
  );
}
