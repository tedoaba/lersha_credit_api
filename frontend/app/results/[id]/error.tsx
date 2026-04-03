"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function ResultDetailError({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  return (
    <div className="max-w-md space-y-4 rounded-lg border border-destructive/40 bg-destructive/5 p-6">
      <h2 className="font-semibold text-destructive">Failed to load result</h2>
      <p className="text-sm text-muted-foreground">{error.message}</p>
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={reset}>
          Try again
        </Button>
        <Link
          href="/results"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
        >
          ← Back to Results
        </Link>
      </div>
    </div>
  );
}
