import { useEffect, useRef } from "react";
import { pollJobStatus, getJobResult } from "@/api/jobs";
import { useAppStore } from "@/stores/appStore";

const POLL_INTERVALS = [500, 1000, 1500, 2000]; // ms; steady-state after index 3

export function useJobPoller() {
  const { jobId, jobStatus, updateJobStatus, setJobResult } = useAppStore(
    (s) => ({
      jobId: s.jobId,
      jobStatus: s.jobStatus,
      updateJobStatus: s.updateJobStatus,
      setJobResult: s.setJobResult,
    })
  );

  const attemptRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!jobId || jobStatus === "complete" || jobStatus === "failed") {
      return;
    }

    attemptRef.current = 0;

    const poll = async () => {
      try {
        const status = await pollJobStatus(jobId);
        updateJobStatus(
          status.status,
          status.step,
          status.progress,
          status.error
        );

        if (status.status === "complete") {
          const result = await getJobResult(jobId);
          setJobResult(result);
          return;
        }

        if (status.status === "failed") {
          return;
        }

        // Schedule next poll with backoff
        const delay =
          POLL_INTERVALS[
            Math.min(attemptRef.current, POLL_INTERVALS.length - 1)
          ];
        attemptRef.current += 1;
        timerRef.current = setTimeout(poll, delay);
      } catch {
        // Network blip — retry with max interval
        timerRef.current = setTimeout(
          poll,
          POLL_INTERVALS[POLL_INTERVALS.length - 1]
        );
      }
    };

    timerRef.current = setTimeout(poll, POLL_INTERVALS[0]);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [jobId, jobStatus, updateJobStatus, setJobResult]);
}
