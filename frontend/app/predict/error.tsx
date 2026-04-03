"use client";

import { Button } from "@/components/ui/button";

export default function PredictError({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  return (
    <div className="max-w-md space-y-4 rounded-lg border border-destructive/40 bg-destructive/5 p-6">
      <h2 className="font-semibold text-destructive">Something went wrong</h2>
      <p className="text-sm text-muted-foreground">{error.message}</p>
      <Button variant="outline" size="sm" onClick={reset}>
        Try again
      </Button>
    </div>
  );
}
