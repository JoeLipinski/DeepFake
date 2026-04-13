import type { ImageType } from "@/types";
import { apiClient } from "./client";

export interface UploadResponse {
  job_id: string;
  status: string;
  estimated_seconds: number;
}

export async function uploadImage(
  file: File,
  removeBackground: boolean,
  imageType: ImageType,
  useMarigold: boolean,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("remove_background", String(removeBackground));
  form.append("image_type", imageType);
  form.append("use_marigold", String(useMarigold));

  const { data } = await apiClient.post<UploadResponse>("/api/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}
