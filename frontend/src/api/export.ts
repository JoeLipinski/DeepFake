import { apiClient } from "./client";

export function buildExportUrl(
  jobId: string,
  variant: string,
  upscale = false
): string {
  const base = import.meta.env.VITE_API_BASE_URL ?? "";
  const params = new URLSearchParams();
  if (upscale) params.set("upscale", "true");
  const qs = params.toString();
  return `${base}/api/export/${jobId}/${variant}${qs ? `?${qs}` : ""}`;
}

export async function downloadVariant(
  jobId: string,
  variant: string,
  upscale = false
): Promise<void> {
  const url = buildExportUrl(jobId, variant, upscale);
  const { data } = await apiClient.get(url, { responseType: "blob" });
  const blobUrl = URL.createObjectURL(data);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = `depthmap_${variant}${upscale ? "_4x" : ""}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(blobUrl);
}
