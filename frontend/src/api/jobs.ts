import { apiClient } from "./client";
import type { JobPollResponse, JobResult, ProcessingParams } from "@/types";

export async function pollJobStatus(jobId: string): Promise<JobPollResponse> {
  const { data } = await apiClient.get<JobPollResponse>(`/api/jobs/${jobId}`);
  return data;
}

export async function getJobResult(jobId: string): Promise<JobResult> {
  const { data } = await apiClient.get<JobResult>(`/api/jobs/${jobId}/result`);
  return data;
}

export async function reprocessJob(
  jobId: string,
  variant: string,
  params: ProcessingParams
): Promise<{ preview_url: string }> {
  const { data } = await apiClient.post(`/api/jobs/${jobId}/reprocess`, {
    variant,
    ...params,
  });
  return data;
}
