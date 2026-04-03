"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export default function SettingsPage() {
  const [healthStatus, setHealthStatus] = useState<"idle" | "ok" | "error">("idle");
  const [healthMsg, setHealthMsg] = useState("");
  const [checking, setChecking] = useState(false);

  async function handleHealthCheck() {
    setChecking(true);
    setHealthStatus("idle");
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (res.ok && data.status !== "unreachable") {
        setHealthStatus("ok");
        setHealthMsg(data.status ?? "ok");
      } else {
        setHealthStatus("error");
        setHealthMsg(data.status ?? "Backend unreachable");
      }
    } catch {
      setHealthStatus("error");
      setHealthMsg("Network error — backend may be down");
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="max-w-xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground text-sm mt-1">
          System configuration and backend connectivity.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Authentication</CardTitle>
          <CardDescription>
            API authentication is handled server-side via the{" "}
            <code className="text-xs font-mono bg-muted px-1 rounded">API_KEY</code>{" "}
            environment variable. No credentials are stored in the browser.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="text-green-600">✓</span>
            Key managed securely in server environment
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Backend Health</CardTitle>
          <CardDescription>
            Check connectivity between this frontend and the FastAPI backend.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button variant="outline" onClick={handleHealthCheck} disabled={checking}>
            {checking ? "Checking…" : "Test Connection"}
          </Button>

          {healthStatus !== "idle" && (
            <div className="flex items-center gap-2">
              <Badge variant={healthStatus === "ok" ? "default" : "destructive"}>
                {healthStatus === "ok" ? "Connected" : "Error"}
              </Badge>
              <span className="text-sm text-muted-foreground">{healthMsg}</span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
