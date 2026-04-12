import { useAppStore } from "@/stores/appStore";
import { X } from "lucide-react";

export function ImagePreview() {
  const { uploadedFile, reset } = useAppStore((s) => ({
    uploadedFile: s.uploadedFile,
    reset: s.reset,
  }));

  if (!uploadedFile) return null;

  const objectUrl = URL.createObjectURL(uploadedFile);

  return (
    <div className="relative group rounded-lg overflow-hidden bg-forge-surface border border-forge-border">
      <img
        src={objectUrl}
        alt="Uploaded image"
        className="w-full h-36 object-cover"
        onLoad={() => URL.revokeObjectURL(objectUrl)}
      />
      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors" />
      <button
        onClick={reset}
        className="absolute top-2 right-2 w-6 h-6 rounded-full bg-black/60 flex items-center justify-center
                   opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/80"
        title="Remove image"
      >
        <X className="w-3.5 h-3.5 text-white" />
      </button>
      <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/70 to-transparent">
        <p className="text-white text-xs truncate font-mono">{uploadedFile.name}</p>
      </div>
    </div>
  );
}
