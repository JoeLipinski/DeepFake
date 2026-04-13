export type JobStatus = "queued" | "running" | "complete" | "failed";

export type ImageType = "photo" | "illustration";

export type JobStep =
  | "queued"
  | "removing_background"
  | "estimating_depth"
  | "refining_depth"
  | "generating_variants"
  | "done";

export type VariantName = "soft" | "standard" | "detailed" | "sharp";

export const VARIANT_NAMES: VariantName[] = ["soft", "standard", "detailed", "sharp"];

export const VARIANT_LABELS: Record<VariantName, string> = {
  soft: "Soft",
  standard: "Standard",
  detailed: "Detailed",
  sharp: "Sharp",
};

export const VARIANT_DESCRIPTIONS: Record<VariantName, string> = {
  soft: "Smooth transitions. Ideal for portraits and fabric.",
  standard: "Balanced default. Works for most subjects.",
  detailed: "High tonal range. Great for coins and architecture.",
  sharp: "Maximum edge definition. Best for logos and text.",
};

export interface JobPollResponse {
  job_id: string;
  status: JobStatus;
  step: JobStep;
  progress: number;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface JobResult {
  job_id: string;
  variants: Record<VariantName, string>;
  original_preview: string;
  metadata: {
    width: number;
    height: number;
    processing_time_seconds: number;
  };
}

export interface ProcessingParams {
  depth_intensity: number;
  blur_radius: number;
  contrast: number;
  edge_enhancement: number;
  invert: boolean;
}

export const DEFAULT_PARAMS: ProcessingParams = {
  depth_intensity: 1.0,
  blur_radius: 1.0,
  contrast: 1.1,
  edge_enhancement: 0.3,
  invert: false,
};

/** Preset params matching each backend style variant config. */
export const VARIANT_PARAMS: Record<VariantName, ProcessingParams> = {
  soft: {
    depth_intensity: 0.75,
    blur_radius: 2.5,
    contrast: 0.9,
    edge_enhancement: 0.0,
    invert: false,
  },
  standard: {
    depth_intensity: 1.0,
    blur_radius: 1.0,
    contrast: 1.1,
    edge_enhancement: 0.3,
    invert: false,
  },
  detailed: {
    depth_intensity: 1.15,
    blur_radius: 0.5,
    contrast: 1.3,
    edge_enhancement: 0.6,
    invert: false,
  },
  sharp: {
    depth_intensity: 1.3,
    blur_radius: 0.0,
    contrast: 1.5,
    edge_enhancement: 1.0,
    invert: false,
  },
};
