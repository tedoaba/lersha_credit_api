"use client";

import DashboardPanel from "@/components/DashboardPanel";
import FarmersPanel from "@/components/FarmersPanel";
import PredictionModal from "@/components/PredictionModal";
import JobProgressBanner from "@/components/JobProgressBanner";
import { Settings } from "lucide-react";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="https://lersha.com/wp-content/uploads/2023/09/lerlogo-1024x313.png"
            alt="Lersha"
            className="h-10 w-auto"
          />
          <div className="border-l pl-4 border-border">
            <h1 className="text-lg font-semibold tracking-tight text-foreground">Credit Scoring</h1>
            <p className="text-muted-foreground text-xs">
              AI-powered evaluation for smallholder farmers
            </p>
          </div>
        </div>
        <Link
          href="/settings"
          className="p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
        >
          <Settings className="h-5 w-5" />
        </Link>
      </div>

      {/* Job Progress Banner (visible when a job is running and modal is closed) */}
      <JobProgressBanner />

      {/* Dashboard */}
      <section className="mt-4">
        <DashboardPanel />
      </section>

      {/* Farmers */}
      <section>
        <FarmersPanel />
      </section>

      {/* Prediction Modal (always rendered, visibility controlled by store) */}
      <PredictionModal />
    </div>
  );
}
