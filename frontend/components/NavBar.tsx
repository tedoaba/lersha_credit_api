"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Settings, Plus } from "lucide-react";
const NAV_LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/predict", label: "Predict" },
  { href: "/farmers", label: "Farmers" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="border-b bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-14 items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="font-semibold text-foreground text-sm tracking-tight">
              Lersha Credit
            </Link>

            <div className="flex items-center gap-1">
              {NAV_LINKS.map(({ href, label }) => {
                const isActive =
                  href === "/" ? pathname === "/" : pathname.startsWith(href);
                return (
                  <Link
                    key={href}
                    href={href}
                    className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    }`}
                  >
                    {label}
                  </Link>
                );
              })}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/predict"
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-2.5 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              <Plus className="h-4 w-4" />
              <span className="hidden sm:inline">New Prediction</span>
            </Link>
            <Link
              href="/settings"
              className={`p-2 rounded-md transition-colors ${
                pathname.startsWith("/settings")
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
            >
              <Settings className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
