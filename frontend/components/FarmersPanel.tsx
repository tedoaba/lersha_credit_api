"use client";

import { useState, useCallback, useMemo } from "react";
import { useResultsPaginated } from "@/lib/queries";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import DecisionBadge from "@/components/DecisionBadge";
import FarmerDetailDrawer from "@/components/FarmerDetailDrawer";
import { ChevronLeft, ChevronRight, Search, X, Plus } from "lucide-react";
import { useJobStore } from "@/lib/stores";
import { groupByFarmer } from "@/lib/types";
import type { GroupedFarmer } from "@/lib/types";

const PER_PAGE = 20;

function formatModelName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function FarmersPanel() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [decision, setDecision] = useState<string>("");
  const [gender, setGender] = useState<string>("");
  const [modelName, setModelName] = useState<string>("");

  const [selectedFarmer, setSelectedFarmer] = useState<GroupedFarmer | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const { data, isLoading, error } = useResultsPaginated({
    page,
    per_page: PER_PAGE,
    search: search || undefined,
    decision: decision || undefined,
    gender: gender || undefined,
    model_name: modelName || undefined,
  });

  const grouped = useMemo(() => (data ? groupByFarmer(data.records) : []), [data]);

  // Derive model columns dynamically from data
  const modelColumns = useMemo(() => {
    const set = new Set<string>();
    for (const farmer of grouped) {
      for (const m of farmer.models) set.add(m.model_name);
    }
    return Array.from(set).sort();
  }, [grouped]);

  const totalPages = data ? Math.ceil(data.total / PER_PAGE) : 0;

  const handleSearch = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    setSearch(searchInput);
    setPage(1);
  }, [searchInput]);

  const clearFilters = useCallback(() => {
    setSearch("");
    setSearchInput("");
    setDecision("");
    setGender("");
    setModelName("");
    setPage(1);
  }, []);

  const hasFilters = search || decision || gender || modelName;

  const openDrawer = useCallback((farmer: GroupedFarmer, model: string) => {
    setSelectedFarmer(farmer);
    setSelectedModel(model);
    setDrawerOpen(true);
  }, []);

  const openPredictionModal = useJobStore((s) => s.openPredictionModal);

  return (
    <div className="space-y-6">
      {/* Section Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold tracking-tight">Farmers</h2>
        <Button onClick={openPredictionModal} className="gap-1.5">
          <Plus className="h-4 w-4" />
          New Prediction
        </Button>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <form onSubmit={handleSearch} className="flex gap-2 flex-1 max-w-md">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by name or UID..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="pl-9"
            />
          </div>
          <Button type="submit" variant="secondary" size="default">
            Search
          </Button>
        </form>

        <div className="flex gap-2 flex-wrap">
          <Select value={decision} onValueChange={(v) => { setDecision(!v || v === "all" ? "" : v); setPage(1); }}>
            <SelectTrigger className="w-35">
              <SelectValue placeholder="Decision" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Decisions</SelectItem>
              <SelectItem value="Eligible">Eligible</SelectItem>
              <SelectItem value="Review">Review</SelectItem>
              <SelectItem value="Not Eligible">Not Eligible</SelectItem>
            </SelectContent>
          </Select>

          <Select value={gender} onValueChange={(v) => { setGender(!v || v === "all" ? "" : v); setPage(1); }}>
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Gender" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Genders</SelectItem>
              <SelectItem value="Male">Male</SelectItem>
              <SelectItem value="Female">Female</SelectItem>
            </SelectContent>
          </Select>

          <Select value={modelName} onValueChange={(v) => { setModelName(!v || v === "all" ? "" : v); setPage(1); }}>
            <SelectTrigger className="w-38">
              <SelectValue placeholder="Model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Models</SelectItem>
              <SelectItem value="xgboost">XGBoost</SelectItem>
              <SelectItem value="random_forest">Random Forest</SelectItem>
              <SelectItem value="catboost">CatBoost</SelectItem>
            </SelectContent>
          </Select>

          {hasFilters && (
            <Button variant="ghost" size="default" onClick={clearFilters} className="gap-1">
              <X className="h-4 w-4" /> Clear
            </Button>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="text-sm text-muted-foreground animate-pulse">Loading farmers...</div>
      )}

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          Failed to load results. Check your API key in Settings.
        </div>
      )}

      {data?.total === 0 && (
        <div className="rounded-md border border-dashed p-8 text-center text-muted-foreground text-sm">
          {hasFilters
            ? "No farmers match the current filters."
            : "No results yet. Submit a prediction job to get started."}
        </div>
      )}

      {data && data.total > 0 && (
        <>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10">#</TableHead>
                  <TableHead>Farmer</TableHead>
                  <TableHead className="hidden sm:table-cell">Gender</TableHead>
                  {modelColumns.map((col) => (
                    <TableHead key={col}>{formatModelName(col)}</TableHead>
                  ))}
                  <TableHead className="hidden md:table-cell">Scored at</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {grouped.map((farmer, i) => {
                  const name =
                    [farmer.first_name, farmer.middle_name, farmer.last_name]
                      .filter(Boolean)
                      .join(" ") || farmer.farmer_uid;
                  const rowNum = (data.page - 1) * PER_PAGE + i + 1;
                  const modelMap = new Map(farmer.models.map((m) => [m.model_name, m]));

                  return (
                    <TableRow key={farmer.farmer_uid}>
                      <TableCell className="text-xs text-muted-foreground">{rowNum}</TableCell>
                      <TableCell className="font-medium">{name}</TableCell>
                      <TableCell className="hidden sm:table-cell text-sm text-muted-foreground">
                        {farmer.gender ?? "\u2014"}
                      </TableCell>
                      {modelColumns.map((col) => {
                        const model = modelMap.get(col);
                        return (
                          <TableCell key={col}>
                            {model ? (
                              <button
                                type="button"
                                title={`View ${formatModelName(col)} result`}
                                onClick={() => openDrawer(farmer, col)}
                                className="cursor-pointer hover:opacity-70 transition-opacity"
                              >
                                <DecisionBadge decision={model.predicted_class_name} confidence={model.confidence_score} showIcon={false} />
                              </button>
                            ) : (
                              <span className="text-xs text-muted-foreground">\u2014</span>
                            )}
                          </TableCell>
                        );
                      })}
                      <TableCell className="hidden md:table-cell text-xs text-muted-foreground">
                        {farmer.timestamp
                          ? new Date(farmer.timestamp).toLocaleString()
                          : "\u2014"}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                Page {data.page} of {totalPages} ({data.total} total)
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                  Previous
                </Button>
                <div className="flex gap-1">
                  {generatePageNumbers(page, totalPages).map((p, i) =>
                    p === "..." ? (
                      <span key={`ellipsis-${i}`} className="px-2 text-muted-foreground">
                        ...
                      </span>
                    ) : (
                      <Button
                        key={p}
                        variant={p === page ? "default" : "outline"}
                        size="sm"
                        className="w-9"
                        onClick={() => setPage(p as number)}
                      >
                        {p}
                      </Button>
                    ),
                  )}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                >
                  Next
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </>
      )}

      <FarmerDetailDrawer
        farmer={selectedFarmer}
        modelName={selectedModel}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  );
}

function generatePageNumbers(current: number, total: number): (number | "...")[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);

  const pages: (number | "...")[] = [];
  pages.push(1);

  if (current > 3) pages.push("...");

  const start = Math.max(2, current - 1);
  const end = Math.min(total - 1, current + 1);
  for (let i = start; i <= end; i++) pages.push(i);

  if (current < total - 2) pages.push("...");

  pages.push(total);
  return pages;
}
