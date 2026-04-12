import { create } from "zustand";
import type {
  JobStatus,
  JobStep,
  JobResult,
  VariantName,
  ProcessingParams,
} from "@/types";
import { DEFAULT_PARAMS } from "@/types";

interface AppState {
  // Upload state
  uploadedFile: File | null;
  removeBackground: boolean;

  // Job state
  jobId: string | null;
  jobStatus: JobStatus | null;
  jobStep: JobStep | null;
  jobProgress: number;
  jobError: string | null;

  // Results
  jobResult: JobResult | null;
  selectedVariant: VariantName;

  // Processing params (sliders)
  params: ProcessingParams;
  isReprocessing: boolean;
  customPreviewUrl: string | null;

  // Actions
  setUploadedFile: (file: File | null) => void;
  setRemoveBackground: (v: boolean) => void;
  setJob: (jobId: string) => void;
  updateJobStatus: (
    status: JobStatus,
    step: JobStep,
    progress: number,
    error?: string | null
  ) => void;
  setJobResult: (result: JobResult) => void;
  setSelectedVariant: (v: VariantName) => void;
  setParam: <K extends keyof ProcessingParams>(key: K, value: ProcessingParams[K]) => void;
  setIsReprocessing: (v: boolean) => void;
  setCustomPreviewUrl: (url: string | null) => void;
  reset: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  uploadedFile: null,
  removeBackground: false,
  jobId: null,
  jobStatus: null,
  jobStep: null,
  jobProgress: 0,
  jobError: null,
  jobResult: null,
  selectedVariant: "standard",
  params: { ...DEFAULT_PARAMS },
  isReprocessing: false,
  customPreviewUrl: null,

  setUploadedFile: (file) => set({ uploadedFile: file }),
  setRemoveBackground: (v) => set({ removeBackground: v }),
  setJob: (jobId) =>
    set({
      jobId,
      jobStatus: "queued",
      jobStep: "queued",
      jobProgress: 0,
      jobError: null,
      jobResult: null,
      customPreviewUrl: null,
    }),
  updateJobStatus: (status, step, progress, error = null) =>
    set({ jobStatus: status, jobStep: step, jobProgress: progress, jobError: error }),
  setJobResult: (result) => set({ jobResult: result }),
  setSelectedVariant: (v) => set({ selectedVariant: v, customPreviewUrl: null }),
  setParam: (key, value) =>
    set((state) => ({ params: { ...state.params, [key]: value } })),
  setIsReprocessing: (v) => set({ isReprocessing: v }),
  setCustomPreviewUrl: (url) => set({ customPreviewUrl: url }),
  reset: () =>
    set({
      uploadedFile: null,
      jobId: null,
      jobStatus: null,
      jobStep: null,
      jobProgress: 0,
      jobError: null,
      jobResult: null,
      selectedVariant: "standard",
      params: { ...DEFAULT_PARAMS },
      isReprocessing: false,
      customPreviewUrl: null,
    }),
}));
