import { useShallow } from "zustand/react/shallow";
import { useAppStore } from "@/stores/appStore";
import type { JobStep } from "@/types";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const STEPS: { key: JobStep; label: string }[] = [
  { key: "queued", label: "Queued" },
  { key: "removing_background", label: "Removing background" },
  { key: "estimating_depth", label: "Estimating depth" },
  { key: "generating_variants", label: "Generating 4 variants" },
  { key: "done", label: "Complete" },
];

const STEP_ORDER: JobStep[] = STEPS.map((s) => s.key);

function stepIndex(step: JobStep | null) {
  if (!step) return -1;
  return STEP_ORDER.indexOf(step);
}

export function ProcessingStatus() {
  const { jobStatus, jobStep, jobProgress, jobError } = useAppStore(
    useShallow((s) => ({
      jobStatus: s.jobStatus,
      jobStep: s.jobStep,
      jobProgress: s.jobProgress,
      jobError: s.jobError,
    }))
  );

  if (!jobStatus || jobStatus === "complete") return null;

  const currentIdx = stepIndex(jobStep);

  return (
    <div className="bg-forge-surface border border-forge-border rounded-xl p-6 space-y-5">
      <div className="flex items-center gap-3">
        {jobStatus === "failed" ? (
          <XCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
        ) : (
          <Loader2 className="w-5 h-5 text-forge-accent animate-spin flex-shrink-0" />
        )}
        <span className="text-forge-text font-medium text-sm">
          {jobStatus === "failed" ? "Processing failed" : "Processing your image..."}
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-forge-muted rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-700",
            jobStatus === "failed" ? "bg-red-500" : "bg-forge-accent"
          )}
          style={{ width: `${Math.round(jobProgress * 100)}%` }}
        />
      </div>

      {/* Step list */}
      <div className="space-y-2">
        {STEPS.filter((s) => s.key !== "queued").map((step) => {
          const idx = stepIndex(step.key);
          const isDone = idx < currentIdx;
          const isActive = idx === currentIdx;
          return (
            <div key={step.key} className="flex items-center gap-2.5">
              {isDone ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              ) : isActive ? (
                <Loader2 className="w-4 h-4 text-forge-accent animate-spin flex-shrink-0" />
              ) : (
                <div className="w-4 h-4 rounded-full border border-forge-muted flex-shrink-0" />
              )}
              <span
                className={cn(
                  "text-xs",
                  isDone && "text-forge-subtle line-through",
                  isActive && "text-forge-text font-medium",
                  !isDone && !isActive && "text-forge-muted"
                )}
              >
                {step.label}
              </span>
            </div>
          );
        })}
      </div>

      {jobError && (
        <p className="text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          {jobError}
        </p>
      )}
    </div>
  );
}
