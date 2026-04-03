"use client";

import { useState } from "react";
import { lershaClient, ApiRequestError } from "@/lib/api";
import { useJobStore } from "@/lib/stores";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { PredictSource } from "@/lib/types";

export default function PredictionForm() {
  const setActiveJobId = useJobStore((s) => s.setActiveJobId);

  const [source, setSource] = useState<PredictSource>("Batch Prediction");
  const [farmerUid, setFarmerUid] = useState("");
  const [rows, setRows] = useState<number>(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const payload =
        source === "Single Value"
          ? { source, farmer_uid: farmerUid }
          : { source, number_of_rows: rows };

      const res = await lershaClient.submitPrediction(payload);
      setActiveJobId(res.job_id);
    } catch (err) {
      setError(
        err instanceof ApiRequestError
          ? `Server error (${err.status}): ${err.message}`
          : err instanceof Error
            ? err.message
            : "Submission failed. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">New Prediction Job</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-1.5">
            <Label htmlFor="source-select">Prediction source</Label>
            <Select
              value={source}
              onValueChange={(v) => setSource(v as PredictSource)}
            >
              <SelectTrigger id="source-select">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Batch Prediction">Batch Prediction</SelectItem>
                <SelectItem value="Single Value">Single Value</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {source === "Single Value" ? (
            <div className="space-y-1.5">
              <Label htmlFor="farmer-uid">Farmer UID</Label>
              <Input
                id="farmer-uid"
                placeholder="e.g. F-001234"
                value={farmerUid}
                onChange={(e) => setFarmerUid(e.target.value)}
                required
              />
            </div>
          ) : (
            <div className="space-y-1.5">
              <Label htmlFor="num-rows">Number of rows (1–100)</Label>
              <Input
                id="num-rows"
                type="number"
                min={1}
                max={100}
                value={rows}
                onChange={(e) => setRows(Number(e.target.value))}
                required
              />
            </div>
          )}

          <Button type="submit" disabled={loading} className="w-full">
            {loading ? "Submitting…" : "Submit Prediction"}
          </Button>

          {error && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          )}
        </form>
      </CardContent>
    </Card>
  );
}
