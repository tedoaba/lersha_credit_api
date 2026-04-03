import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[40vh] text-center space-y-4">
      <h1 className="text-4xl font-bold text-muted-foreground">404</h1>
      <p className="text-lg font-medium">Page not found</p>
      <p className="text-sm text-muted-foreground max-w-sm">
        The page you are looking for does not exist or has been moved.
      </p>
      <Link
        href="/"
        className="inline-flex items-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium hover:bg-muted transition-colors"
      >
        ← Back to Dashboard
      </Link>
    </div>
  );
}
