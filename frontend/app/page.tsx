"use client";

import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import DashboardPanel from "@/components/DashboardPanel";
import FarmersPanel from "@/components/FarmersPanel";
import PredictPanel from "@/components/PredictPanel";
import { BarChart3, Users, Plus, Settings } from "lucide-react";
import Link from "next/link";

type TabValue = "dashboard" | "farmers" | "predict";

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<TabValue>("dashboard");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Lersha Credit Scoring</h1>
          <p className="text-muted-foreground text-sm mt-1">
            AI-powered credit scoring for smallholder farmers.
          </p>
        </div>
        <Link
          href="/settings"
          className="p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
        >
          <Settings className="h-5 w-5" />
        </Link>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabValue)}>
        <TabsList variant="line" className="w-full justify-start border-b pb-0">
          <TabsTrigger value="dashboard" className="gap-1.5">
            <BarChart3 className="h-4 w-4" />
            Dashboard
          </TabsTrigger>
          <TabsTrigger value="farmers" className="gap-1.5">
            <Users className="h-4 w-4" />
            Farmers
          </TabsTrigger>
          <TabsTrigger value="predict" className="gap-1.5">
            <Plus className="h-4 w-4" />
            New Prediction
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="pt-4">
          <DashboardPanel />
        </TabsContent>

        <TabsContent value="farmers" className="pt-4">
          <FarmersPanel />
        </TabsContent>

        <TabsContent value="predict" className="pt-4">
          <PredictPanel onCompleted={() => setActiveTab("farmers")} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
