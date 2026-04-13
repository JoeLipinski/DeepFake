import { useCallback, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { uploadImage } from "@/api/upload";
import { useAppStore } from "@/stores/appStore";

export function useUpload() {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const { setUploadedFile, setJob, removeBackground, imageType, useMarigold } = useAppStore(
    useShallow((s) => ({
      setUploadedFile: s.setUploadedFile,
      setJob: s.setJob,
      removeBackground: s.removeBackground,
      imageType: s.imageType,
      useMarigold: s.useMarigold,
    }))
  );

  const upload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setUploadError(null);
      setUploadedFile(file);

      try {
        const resp = await uploadImage(file, removeBackground, imageType, useMarigold);
        setJob(resp.job_id);
      } catch (err: unknown) {
        const msg =
          err instanceof Error ? err.message : "Upload failed. Please try again.";
        setUploadError(msg);
      } finally {
        setIsUploading(false);
      }
    },
    [removeBackground, imageType, useMarigold, setUploadedFile, setJob]
  );

  return { upload, isUploading, uploadError };
}
