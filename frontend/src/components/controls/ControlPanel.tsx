import { useShallow } from "zustand/react/shallow";
import { useAppStore } from "@/stores/appStore";
import { useReprocess } from "@/hooks/useReprocess";
import { Loader2, RefreshCw, Sliders } from "lucide-react";
import { cn } from "@/lib/utils";

interface SliderRowProps {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  displayValue: string;
  onChange: (v: number) => void;
  disabled: boolean;
}

function SliderRow({
  label,
  description,
  value,
  min,
  max,
  step,
  displayValue,
  onChange,
  disabled,
}: SliderRowProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-forge-text text-xs font-medium">{label}</span>
          <p className="text-forge-subtle text-[10px] mt-0.5">{description}</p>
        </div>
        <span className="text-forge-accent text-xs font-mono tabular-nums min-w-[2.5rem] text-right">
          {displayValue}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className={cn(
          "w-full h-1.5 rounded-full appearance-none cursor-pointer",
          "bg-forge-muted accent-forge-accent",
          disabled && "opacity-40 cursor-not-allowed"
        )}
      />
    </div>
  );
}

export function ControlPanel() {
  const { jobStatus, params, setParam, isReprocessing } =
    useAppStore(
      useShallow((s) => ({
        jobStatus: s.jobStatus,
        params: s.params,
        setParam: s.setParam,
        isReprocessing: s.isReprocessing,
      }))
    );

  const { reprocess } = useReprocess();

  const isJobComplete = jobStatus === "complete";
  const disabled = !isJobComplete || isReprocessing;

  return (
    <div className="bg-forge-surface border border-forge-border rounded-xl overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-forge-border">
        <Sliders className="w-4 h-4 text-forge-subtle" />
        <span className="text-forge-text text-sm font-medium">Controls</span>
        {!isJobComplete && (
          <span className="ml-auto text-forge-subtle text-[10px]">
            Available after processing
          </span>
        )}
      </div>

      <div className="p-4 space-y-5">
        {/* Depth sliders */}
        <SliderRow
          label="Depth Intensity"
          description="Power-law scaling of depth range"
          value={params.depth_intensity}
          min={0.1}
          max={2.0}
          step={0.05}
          displayValue={params.depth_intensity.toFixed(2)}
          onChange={(v) => setParam("depth_intensity", v)}
          disabled={disabled}
        />

        <SliderRow
          label="Smoothing"
          description="Gaussian blur radius for carving paths"
          value={params.blur_radius}
          min={0}
          max={8}
          step={0.1}
          displayValue={`${params.blur_radius.toFixed(1)}px`}
          onChange={(v) => setParam("blur_radius", v)}
          disabled={disabled}
        />

        <SliderRow
          label="Contrast"
          description="Tonal contrast of the depth map"
          value={params.contrast}
          min={0.5}
          max={3.0}
          step={0.05}
          displayValue={`${(params.contrast * 100).toFixed(0)}%`}
          onChange={(v) => setParam("contrast", v)}
          disabled={disabled}
        />

        <SliderRow
          label="Edge Enhancement"
          description="Unsharp mask for ridge definition"
          value={params.edge_enhancement}
          min={0}
          max={1}
          step={0.01}
          displayValue={`${(params.edge_enhancement * 100).toFixed(0)}%`}
          onChange={(v) => setParam("edge_enhancement", v)}
          disabled={disabled}
        />

        {/* Invert toggle */}
        <div className="flex items-center justify-between">
          <div>
            <span className="text-forge-text text-xs font-medium">Invert Depth</span>
            <p className="text-forge-subtle text-[10px] mt-0.5">
              Flip black/white for engraver compatibility
            </p>
          </div>
          <button
            onClick={() => setParam("invert", !params.invert)}
            disabled={disabled}
            className={cn(
              "relative w-11 h-6 rounded-full transition-colors",
              params.invert ? "bg-forge-accent" : "bg-forge-muted",
              disabled && "opacity-40 cursor-not-allowed"
            )}
          >
            <span
              className={cn(
                "absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform",
                params.invert ? "translate-x-5" : "translate-x-0"
              )}
            />
          </button>
        </div>

        {/* Apply button */}
        <button
          onClick={reprocess}
          disabled={disabled}
          className={cn(
            "w-full h-9 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2",
            isJobComplete && !isReprocessing
              ? "bg-forge-accent hover:bg-forge-accent-hover text-white"
              : "bg-forge-muted text-forge-subtle cursor-not-allowed"
          )}
        >
          {isReprocessing ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Applying...
            </>
          ) : (
            <>
              <RefreshCw className="w-4 h-4" />
              Apply Changes
            </>
          )}
        </button>
      </div>
    </div>
  );
}
