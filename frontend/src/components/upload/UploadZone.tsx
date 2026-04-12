import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, ImageIcon, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUpload } from "@/hooks/useUpload";

const ACCEPTED_TYPES = {
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
  "image/svg+xml": [".svg"],
  "image/webp": [".webp"],
};

export function UploadZone() {
  const { upload, isUploading, uploadError } = useUpload();

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) {
        upload(accepted[0]);
      }
    },
    [upload]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } =
    useDropzone({
      onDrop,
      accept: ACCEPTED_TYPES,
      maxFiles: 1,
      maxSize: 20 * 1024 * 1024,
      disabled: isUploading,
    });

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={cn(
          "relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200",
          "hover:border-forge-accent/60 hover:bg-forge-accent/5",
          isDragActive && !isDragReject && "border-forge-accent bg-forge-accent/10",
          isDragReject && "border-red-500 bg-red-500/10",
          !isDragActive && "border-forge-border bg-forge-surface",
          isUploading && "pointer-events-none opacity-50"
        )}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center gap-3">
          {isUploading ? (
            <>
              <div className="w-10 h-10 border-2 border-forge-accent border-t-transparent rounded-full animate-spin" />
              <p className="text-forge-subtle text-sm">Uploading...</p>
            </>
          ) : isDragReject ? (
            <>
              <AlertCircle className="w-10 h-10 text-red-400" />
              <p className="text-red-400 text-sm font-medium">
                File type not supported
              </p>
            </>
          ) : (
            <>
              <div className="w-10 h-10 rounded-full bg-forge-muted flex items-center justify-center">
                {isDragActive ? (
                  <ImageIcon className="w-5 h-5 text-forge-accent" />
                ) : (
                  <Upload className="w-5 h-5 text-forge-subtle" />
                )}
              </div>
              <div>
                <p className="text-forge-text text-sm font-medium">
                  {isDragActive
                    ? "Drop to upload"
                    : "Drop image here or click to browse"}
                </p>
                <p className="text-forge-subtle text-xs mt-1">
                  JPEG · PNG · SVG · WebP · Max 20 MB
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {uploadError && (
        <div className="flex items-center gap-2 text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {uploadError}
        </div>
      )}
    </div>
  );
}
