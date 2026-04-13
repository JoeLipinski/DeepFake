import { useState } from "react";
import { Download, Maximize2, Loader2, Zap, Info } from "lucide-react";
import { useShallow } from "zustand/react/shallow";
import { useAppStore } from "@/stores/appStore";
import { buildExportUrl, downloadVariant } from "@/api/export";
import { VARIANT_LABELS, VARIANT_DESCRIPTIONS } from "@/types";
import { cn } from "@/lib/utils";

export function ExportPanel() {
  const { jobId, jobResult, jobStatus, selectedVariant, customPreviewUrl } =
    useAppStore(
      useShallow((s) => ({
        jobId: s.jobId,
        jobResult: s.jobResult,
        jobStatus: s.jobStatus,
        selectedVariant: s.selectedVariant,
        customPreviewUrl: s.customPreviewUrl,
      }))
    );

  const [upscale, setUpscale] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  const isComplete = jobStatus === "complete";

  const previewUrl =
    customPreviewUrl ??
    (jobId && isComplete ? buildExportUrl(jobId, selectedVariant) : null);

  const handleDownload = async () => {
    if (!jobId) return;
    setIsDownloading(true);
    try {
      await downloadVariant(jobId, selectedVariant, upscale);
    } finally {
      setIsDownloading(false);
    }
  };

  const handleDownloadAll = () => {
    if (!jobId || !jobResult) return;
    Object.keys(jobResult.variants).forEach((v) => {
      downloadVariant(jobId, v, false);
    });
  };

  return (
    <div className="bg-forge-surface border border-forge-border rounded-xl overflow-hidden space-y-0">
      {/* Large preview */}
      <div className="aspect-square bg-forge-muted relative">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={`${VARIANT_LABELS[selectedVariant]} depth map preview`}
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center p-6">
            <Maximize2 className="w-8 h-8 text-forge-muted" />
            <p className="text-forge-subtle text-xs">
              Depth map preview will appear here after processing
            </p>
          </div>
        )}

        {/* Variant badge */}
        {isComplete && (
          <div className="absolute bottom-3 left-3 px-2 py-1 rounded-md bg-black/60 backdrop-blur-sm">
            <span className="text-white text-xs font-medium">
              {VARIANT_LABELS[selectedVariant]}
            </span>
          </div>
        )}
      </div>

      {/* Metadata */}
      {isComplete && jobResult && (
        <div className="px-4 py-3 border-t border-forge-border flex items-center gap-4 text-[10px] text-forge-subtle font-mono">
          <span>{jobResult.metadata.width} × {jobResult.metadata.height}px</span>
          <span>{jobResult.metadata.processing_time_seconds}s</span>
          <span className="ml-auto">Grayscale PNG</span>
        </div>
      )}

      {/* Export controls */}
      <div className="p-4 space-y-3 border-t border-forge-border">
        {/* Variant description */}
        {isComplete && (
          <div className="flex items-start gap-2 text-[11px] text-forge-subtle bg-forge-muted/50 rounded-lg p-3">
            <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-forge-accent" />
            <span>{VARIANT_DESCRIPTIONS[selectedVariant]}</span>
          </div>
        )}

        {/* HD Upscale toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-forge-accent" />
            <div>
              <p className="text-forge-text text-xs font-medium">HD Upscale (×4)</p>
              <p className="text-forge-subtle text-[10px]">Real-ESRGAN · ~3-5s extra</p>
            </div>
          </div>
          <button
            onClick={() => setUpscale(!upscale)}
            disabled={!isComplete}
            className={cn(
              "relative w-9 h-5 rounded-full transition-colors",
              upscale ? "bg-forge-accent" : "bg-forge-muted",
              !isComplete && "opacity-40 cursor-not-allowed"
            )}
          >
            <span
              className={cn(
                "absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform",
                upscale ? "translate-x-4" : "translate-x-0.5"
              )}
            />
          </button>
        </div>

        {/* Download selected */}
        <button
          onClick={handleDownload}
          disabled={!isComplete || isDownloading}
          className={cn(
            "w-full h-10 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all",
            isComplete && !isDownloading
              ? "bg-forge-accent hover:bg-forge-accent-hover text-white"
              : "bg-forge-muted text-forge-subtle cursor-not-allowed"
          )}
        >
          {isDownloading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              {upscale ? "Upscaling & Downloading..." : "Downloading..."}
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Download {isComplete ? VARIANT_LABELS[selectedVariant] : ""}
              {upscale && " (4×)"}
            </>
          )}
        </button>

        {/* Download all */}
        {isComplete && (
          <button
            onClick={handleDownloadAll}
            className="w-full h-9 rounded-lg text-xs font-medium border border-forge-border
                       text-forge-subtle hover:text-forge-text hover:border-forge-muted
                       flex items-center justify-center gap-2 transition-all"
          >
            <Download className="w-3.5 h-3.5" />
            Download All 4 Variants
          </button>
        )}
      </div>
    </div>
  );
}
