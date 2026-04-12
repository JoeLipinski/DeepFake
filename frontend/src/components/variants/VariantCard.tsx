import { Download, Check } from "lucide-react";
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
  const { jobId, jobResult } = useAppStore((s) => ({
    jobId: s.jobId,
    jobResult: s.jobResult,
  }));

  const imgUrl = jobId ? buildExportUrl(jobId, name) : null;

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!jobId) return;
    const url = buildExportUrl(jobId, name);
    const a = document.createElement("a");
    a.href = url;
    a.download = `depthmap_${name}.png`;
    a.click();
  };

  return (
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

        {/* Selected overlay */}
        {isSelected && (
          <div className="absolute top-2 left-2 w-5 h-5 rounded-full bg-forge-accent flex items-center justify-center">
            <Check className="w-3 h-3 text-white" />
          </div>
        )}

        {/* Download button */}
        <button
          onClick={handleDownload}
          className="absolute top-2 right-2 w-7 h-7 rounded-lg bg-black/60 backdrop-blur-sm
                     flex items-center justify-center opacity-0 group-hover:opacity-100
                     transition-opacity hover:bg-black/80"
          title={`Download ${VARIANT_LABELS[name]}`}
        >
          <Download className="w-3.5 h-3.5 text-white" />
        </button>
      </div>

      {/* Label */}
      <div className="p-2.5 bg-forge-surface">
        <p className="text-forge-text text-xs font-semibold">{VARIANT_LABELS[name]}</p>
        <p className="text-forge-subtle text-[10px] leading-tight mt-0.5 line-clamp-2">
          {VARIANT_DESCRIPTIONS[name]}
        </p>
      </div>
    </div>
  );
}
