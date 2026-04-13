import { useState, useEffect } from "react";
import { Download, Check, ZoomIn, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { VariantName } from "@/types";
import { VARIANT_LABELS, VARIANT_DESCRIPTIONS } from "@/types";
import { buildExportUrl } from "@/api/export";
import { useAppStore } from "@/stores/appStore";

interface Props {
  name: VariantName;
  isSelected: boolean;
  onSelect: (name: VariantName) => void;
}

export function VariantCard({ name, isSelected, onSelect }: Props) {
  const jobId = useAppStore((s) => s.jobId);
  const [lightboxOpen, setLightboxOpen] = useState(false);

  const imgUrl = jobId ? buildExportUrl(jobId, name) : null;

  // Close lightbox on Escape
  useEffect(() => {
    if (!lightboxOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setLightboxOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightboxOpen]);

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!jobId) return;
    const a = document.createElement("a");
    a.href = buildExportUrl(jobId, name);
    a.download = `depthmap_${name}.png`;
    a.click();
  };

  const handleZoom = (e: React.MouseEvent) => {
    e.stopPropagation();
    setLightboxOpen(true);
  };

  return (
    <>
      <div
        onClick={() => onSelect(name)}
        className={cn(
          "relative rounded-xl overflow-hidden cursor-pointer border-2 transition-all duration-150 group",
          isSelected
            ? "border-forge-accent shadow-[0_0_16px_rgba(249,115,22,0.25)]"
            : "border-forge-border hover:border-forge-muted"
        )}
      >
        {/* Depth map preview */}
        <div className="aspect-square bg-forge-muted relative">
          {imgUrl ? (
            <img
              src={imgUrl}
              alt={`${VARIANT_LABELS[name]} depth map`}
              className="w-full h-full object-cover"
              loading="lazy"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <div className="w-6 h-6 border-2 border-forge-muted border-t-forge-subtle rounded-full animate-spin" />
            </div>
          )}

          {/* Selected badge */}
          {isSelected && (
            <div className="absolute top-2 left-2 w-5 h-5 rounded-full bg-forge-accent flex items-center justify-center">
              <Check className="w-3 h-3 text-white" />
            </div>
          )}

          {/* Hover actions */}
          <div className="absolute top-2 right-2 flex gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
            {imgUrl && (
              <button
                onClick={handleZoom}
                className="w-7 h-7 rounded-lg bg-black/60 backdrop-blur-sm flex items-center justify-center hover:bg-black/80"
                title="Enlarge preview"
              >
                <ZoomIn className="w-3.5 h-3.5 text-white" />
              </button>
            )}
            <button
              onClick={handleDownload}
              className="w-7 h-7 rounded-lg bg-black/60 backdrop-blur-sm flex items-center justify-center hover:bg-black/80"
              title={`Download ${VARIANT_LABELS[name]}`}
            >
              <Download className="w-3.5 h-3.5 text-white" />
            </button>
          </div>
        </div>

        {/* Label */}
        <div className="p-2.5 bg-forge-surface">
          <p className="text-forge-text text-xs font-semibold">{VARIANT_LABELS[name]}</p>
          <p className="text-forge-subtle text-[10px] leading-tight mt-0.5 line-clamp-2">
            {VARIANT_DESCRIPTIONS[name]}
          </p>
        </div>
      </div>

      {/* Lightbox */}
      {lightboxOpen && imgUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-6"
          onClick={() => setLightboxOpen(false)}
        >
          <div
            className="relative flex flex-col items-center gap-3 max-w-[90vw] max-h-[90vh]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header bar */}
            <div className="w-full flex items-center justify-between">
              <span className="text-white text-sm font-medium">
                {VARIANT_LABELS[name]}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download
                </button>
                <button
                  onClick={() => setLightboxOpen(false)}
                  className="w-8 h-8 rounded-lg bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Image */}
            <img
              src={imgUrl}
              alt={`${VARIANT_LABELS[name]} depth map`}
              className="max-w-full max-h-[80vh] object-contain rounded-lg"
            />

            <p className="text-white/40 text-xs">
              Click outside or press Esc to close
            </p>
          </div>
        </div>
      )}
    </>
  );
}
