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

export default function App() {
  useJobPoller();

  const { jobId, jobStatus } = useAppStore(
    useShallow((s) => ({ jobId: s.jobId, jobStatus: s.jobStatus }))
  );

  const hasJob = !!jobId;
  const isProcessing =
    jobStatus === "queued" || jobStatus === "running";
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
              <UploadZone />
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

        {/* Upload zone below image preview on mobile when job exists */}
        {!hasJob && (
          <div className="lg:hidden mt-4 max-w-xl mx-auto">
            {/* Already rendered above in left column on mobile */}
          </div>
        )}
      </main>

      <footer className="border-t border-forge-border py-4 text-center text-forge-subtle text-xs">
        DepthForge · Depth Anything V2 Large · Real-ESRGAN · rembg
      </footer>
    </div>
  );
}
