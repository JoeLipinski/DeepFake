import { Layers } from "lucide-react";
import { useAppStore } from "@/stores/appStore";

export function Header() {
  const reset = useAppStore((s) => s.reset);

  return (
    <header className="border-b border-forge-border bg-forge-surface/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center justify-between">
        <button
          onClick={reset}
          className="flex items-center gap-2.5 hover:opacity-80 transition-opacity"
        >
          <div className="w-8 h-8 rounded-lg bg-forge-accent flex items-center justify-center">
            <Layers className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-forge-text tracking-tight">
            DepthForge
          </span>
          <span className="hidden sm:block text-forge-subtle text-sm font-normal">
            2.5D Depth Map Generator
          </span>
        </button>

        <div className="flex items-center gap-3">
          <span className="text-xs text-forge-subtle font-mono">
            Powered by Depth Anything V2
          </span>
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
        </div>
      </div>
    </header>
  );
}
