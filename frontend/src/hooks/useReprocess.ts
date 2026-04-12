import { useCallback } from "react";
import { reprocessJob } from "@/api/jobs";
import { useAppStore } from "@/stores/appStore";

export function useReprocess() {
  const {
    jobId,
    jobStatus,
    selectedVariant,
    params,
    setIsReprocessing,
    setCustomPreviewUrl,
  } = useAppStore((s) => ({
    jobId: s.jobId,
    jobStatus: s.jobStatus,
    selectedVariant: s.selectedVariant,
    params: s.params,
    setIsReprocessing: s.setIsReprocessing,
    setCustomPreviewUrl: s.setCustomPreviewUrl,
  }));

  const reprocess = useCallback(async () => {
    if (!jobId || jobStatus !== "complete") return;

    setIsReprocessing(true);
    try {
      const resp = await reprocessJob(jobId, selectedVariant, params);
      // Add cache-busting timestamp so the browser re-fetches the new image
      const base = import.meta.env.VITE_API_BASE_URL ?? "";
      setCustomPreviewUrl(`${base}${resp.preview_url}?t=${Date.now()}`);
    } finally {
      setIsReprocessing(false);
    }
  }, [jobId, jobStatus, selectedVariant, params, setIsReprocessing, setCustomPreviewUrl]);

  return { reprocess };
}
