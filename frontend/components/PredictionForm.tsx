"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { lershaClient, ApiRequestError } from "@/lib/api";
import { useFarmerSearch } from "@/lib/queries";
import { useJobStore } from "@/lib/stores";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { User, Users, Search, Loader2 } from "lucide-react";
import type { FarmerSearchResult } from "@/lib/types";

type Mode = "single" | "batch";

export default function PredictionForm() {
  const setActiveJobId = useJobStore((s) => s.setActiveJobId);

  // Mode
  const [mode, setMode] = useState<Mode>("single");

  // Single mode state
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFarmer, setSelectedFarmer] = useState<FarmerSearchResult | null>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Batch mode state
  const [rows, setRows] = useState(5);
  const [batchGender, setBatchGender] = useState("");
  const [ageMin, setAgeMin] = useState<string>("");
  const [ageMax, setAgeMax] = useState<string>("");

  // Shared state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Farmer search
  const { data: searchResults, isLoading: searching } = useFarmerSearch(searchQuery);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const selectFarmer = useCallback((farmer: FarmerSearchResult) => {
    setSelectedFarmer(farmer);
    const name = [farmer.first_name, farmer.middle_name, farmer.last_name]
      .filter(Boolean)
      .join(" ");
    setSearchQuery(name || farmer.farmer_uid);
    setShowDropdown(false);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedFarmer(null);
    setSearchQuery("");
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const payload =
        mode === "single"
          ? { source: "Single Value" as const, farmer_uid: selectedFarmer!.farmer_uid }
          : {
              source: "Batch Prediction" as const,
              number_of_rows: rows,
              ...(batchGender && { gender: batchGender }),
              ...(ageMin && { age_min: Number(ageMin) }),
              ...(ageMax && { age_max: Number(ageMax) }),
            };

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

  const canSubmit =
    mode === "single"
      ? !!selectedFarmer
      : rows >= 1 && rows <= 100;

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Mode toggle */}
      <div className="grid grid-cols-2 gap-3">
        <button
          type="button"
          onClick={() => setMode("single")}
          className={`flex items-center gap-3 rounded-lg border-2 p-4 text-left transition-all ${
            mode === "single"
              ? "border-primary bg-primary/5 ring-1 ring-primary/20"
              : "border-muted hover:border-muted-foreground/30"
          }`}
        >
          <div className={`rounded-full p-2 ${mode === "single" ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}`}>
            <User className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-medium">Single Farmer</p>
            <p className="text-xs text-muted-foreground">Search and score one farmer</p>
          </div>
        </button>

        <button
          type="button"
          onClick={() => setMode("batch")}
          className={`flex items-center gap-3 rounded-lg border-2 p-4 text-left transition-all ${
            mode === "batch"
              ? "border-primary bg-primary/5 ring-1 ring-primary/20"
              : "border-muted hover:border-muted-foreground/30"
          }`}
        >
          <div className={`rounded-full p-2 ${mode === "batch" ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}`}>
            <Users className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-medium">Batch Prediction</p>
            <p className="text-xs text-muted-foreground">Score multiple farmers at once</p>
          </div>
        </button>
      </div>

      {/* Single farmer mode */}
      {mode === "single" && (
        <Card>
          <CardContent className="pt-6 space-y-4">
            <div className="space-y-1.5">
              <Label>Search Farmer</Label>
              <div className="relative" ref={dropdownRef}>
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Type a name or UID to search..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setSelectedFarmer(null);
                    setShowDropdown(true);
                  }}
                  onFocus={() => searchQuery.length >= 2 && setShowDropdown(true)}
                  className="pl-9 pr-8"
                />
                {searching && (
                  <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
                )}

                {/* Search results dropdown */}
                {showDropdown && searchQuery.length >= 2 && (
                  <div className="absolute z-50 top-full mt-1 w-full rounded-md border bg-popover shadow-lg max-h-60 overflow-y-auto">
                    {searching && (
                      <div className="px-3 py-4 text-sm text-muted-foreground text-center">
                        Searching...
                      </div>
                    )}
                    {!searching && searchResults && searchResults.results.length === 0 && (
                      <div className="px-3 py-4 text-sm text-muted-foreground text-center">
                        No farmers found matching &ldquo;{searchQuery}&rdquo;
                      </div>
                    )}
                    {!searching && searchResults?.results.map((farmer) => {
                      const name = [farmer.first_name, farmer.middle_name, farmer.last_name]
                        .filter(Boolean)
                        .join(" ");
                      return (
                        <button
                          key={farmer.farmer_uid}
                          type="button"
                          onClick={() => selectFarmer(farmer)}
                          className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-muted transition-colors"
                        >
                          <div className="rounded-full bg-muted p-1.5">
                            <User className="h-3.5 w-3.5 text-muted-foreground" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="text-sm font-medium truncate">
                              {name || "Unnamed"}
                            </p>
                            <p className="text-xs text-muted-foreground font-mono">
                              {farmer.farmer_uid}
                            </p>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Selected farmer card */}
            {selectedFarmer && (
              <div className="flex items-center justify-between rounded-md border bg-muted/30 px-3 py-2.5">
                <div className="flex items-center gap-3">
                  <div className="rounded-full bg-primary/10 p-1.5">
                    <User className="h-3.5 w-3.5 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">
                      {[selectedFarmer.first_name, selectedFarmer.middle_name, selectedFarmer.last_name]
                        .filter(Boolean)
                        .join(" ") || "Unnamed"}
                    </p>
                    <p className="text-xs text-muted-foreground font-mono">
                      {selectedFarmer.farmer_uid}
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={clearSelection}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Change
                </button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Batch mode */}
      {mode === "batch" && (
        <Card>
          <CardContent className="pt-6 space-y-5">
            <div className="space-y-1.5">
              <Label htmlFor="num-rows">Number of farmers</Label>
              <Input
                id="num-rows"
                type="number"
                min={1}
                max={100}
                value={rows}
                onChange={(e) => setRows(Number(e.target.value))}
                required
              />
              <p className="text-xs text-muted-foreground">
                Randomly selects farmers from the database (1-100)
              </p>
            </div>

            <div className="border-t pt-4">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-3">
                Optional Filters
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-1.5">
                  <Label>Gender</Label>
                  <Select value={batchGender} onValueChange={(v) => setBatchGender(!v || v === "all" ? "" : v)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Any" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Any</SelectItem>
                      <SelectItem value="Male">Male</SelectItem>
                      <SelectItem value="Female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="age-min">Min Age</Label>
                  <Input
                    id="age-min"
                    type="number"
                    min={0}
                    max={120}
                    placeholder="Any"
                    value={ageMin}
                    onChange={(e) => setAgeMin(e.target.value)}
                  />
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="age-max">Max Age</Label>
                  <Input
                    id="age-max"
                    type="number"
                    min={0}
                    max={120}
                    placeholder="Any"
                    value={ageMax}
                    onChange={(e) => setAgeMax(e.target.value)}
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Submit */}
      <Button type="submit" disabled={loading || !canSubmit} className="w-full">
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Submitting...
          </>
        ) : mode === "single" ? (
          "Score Farmer"
        ) : (
          `Score ${rows} Farmer${rows !== 1 ? "s" : ""}`
        )}
      </Button>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}
    </form>
  );
}
