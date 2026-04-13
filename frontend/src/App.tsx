import { useShallow } from "zustand/react/shallow";
import { useAppStore } from "@/stores/appStore";
import { useJobPoller } from "@/hooks/useJobPoller";
import { Header } from "@/components/layout/Header";
import { UploadZone } from "@/components/upload/UploadZone";
import { ImagePreview } from "@/components/upload/ImagePreview";
import { ProcessingStatus } from "@/components/processing/ProcessingStatus";
import { ControlPanel } from "@/components/controls/ControlPanel";
import { VariantGrid } from "@/components/variants/VariantGrid";
import { ExportPanel } from "@/components/export/ExportPanel";
import { cn } from "@/lib/utils";

export default function App() {
  useJobPoller();

  const { jobId, jobStatus, removeBackground, setRemoveBackground } = useAppStore(
    useShallow((s) => ({
      jobId: s.jobId,
      jobStatus: s.jobStatus,
      removeBackground: s.removeBackground,
      setRemoveBackground: s.setRemoveBackground,
    }))
  );

  const hasJob = !!jobId;
  const isProcessing = jobStatus === "queued" || jobStatus === "running";
  const isComplete = jobStatus === "complete";
  const isFailed = jobStatus === "failed";

  return (
    <div className="min-h-screen flex flex-col bg-forge-bg">
      <Header />

      <main className="flex-1 max-w-screen-2xl mx-auto w-full px-4 sm:px-6 py-6">
        {/* Hero — shown before any upload */}
        {!hasJob && (
          <div className="max-w-xl mx-auto text-center mb-8">
            <h1 className="text-2xl font-bold text-forge-text mb-2">
              2D → 2.5D Depth Maps for Laser Engraving
            </h1>
            <p className="text-forge-subtle text-sm leading-relaxed">
              Upload any photo, illustration, or logo and get four high-quality
              grayscale depth maps optimized for laser engraving, CNC relief
              carving, and challenge coins.
            </p>
          </div>
        )}

        <div
          className={
            isComplete
              ? "grid grid-cols-1 lg:grid-cols-[280px_1fr_280px] gap-6 items-start"
              : "grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6 items-start max-w-4xl mx-auto"
          }
        >
          {/* ── LEFT COLUMN: Upload + Controls ── */}
          <div className="space-y-4">
            {!hasJob ? (
              <>
                <UploadZone />

                {/* Remove Background — configured BEFORE upload */}
                <div className="bg-forge-surface border border-forge-border rounded-xl px-4 py-3 flex items-center justify-between">
                  <div>
                    <p className="text-forge-text text-xs font-medium">Remove Background</p>
                    <p className="text-forge-subtle text-[10px] mt-0.5">
                      Isolate subject before depth estimation
                    </p>
                  </div>
                  <button
                    onClick={() => setRemoveBackground(!removeBackground)}
                    className={cn(
                      "relative w-9 h-5 rounded-full transition-colors",
                      removeBackground ? "bg-forge-accent" : "bg-forge-muted"
                    )}
                  >
                    <span
                      className={cn(
                        "absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform",
                        removeBackground ? "translate-x-4" : "translate-x-0.5"
                      )}
                    />
                  </button>
                </div>
              </>
            ) : (
              <>
                <ImagePreview />
                <ControlPanel />
              </>
            )}
          </div>

          {/* ── CENTER COLUMN: Processing status or variant grid ── */}
          <div>
            {!hasJob && (
              <div className="hidden lg:flex items-center justify-center h-64 rounded-xl border-2 border-dashed border-forge-border text-forge-muted text-sm">
                Upload an image to get started
              </div>
            )}

            {(isProcessing || isFailed) && <ProcessingStatus />}

            {isComplete && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-forge-text text-sm font-semibold">
                    Style Variants
                  </h2>
                  <span className="text-forge-subtle text-xs">
                    Click a variant to select it
                  </span>
                </div>
                <VariantGrid />
              </div>
            )}
          </div>

          {/* ── RIGHT COLUMN: Export panel (only when results are ready) ── */}
          {isComplete && (
            <div className="space-y-4">
              <h2 className="text-forge-text text-sm font-semibold">Export</h2>
              <ExportPanel />
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-forge-border py-4 text-center text-forge-subtle text-xs">
        DepthForge · Depth Anything V2 Large · Lanczos 4× · rembg
      </footer>
    </div>
  );
}
