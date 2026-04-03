export default function ResultsLoading() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-7 w-32 bg-muted rounded" />
      <div className="rounded-md border divide-y">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="flex gap-4 px-4 py-3">
            <div className="h-4 w-32 bg-muted rounded" />
            <div className="h-4 w-20 bg-muted rounded" />
            <div className="h-4 w-20 bg-muted rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}
