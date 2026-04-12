import { useAppStore } from "@/stores/appStore";
import { VariantCard } from "./VariantCard";
import type { VariantName } from "@/types";
import { VARIANT_NAMES } from "@/types";

export function VariantGrid() {
  const { selectedVariant, setSelectedVariant } = useAppStore((s) => ({
    selectedVariant: s.selectedVariant,
    setSelectedVariant: s.setSelectedVariant,
  }));

  return (
    <div className="space-y-3">
      <p className="text-forge-subtle text-xs">
        Select a variant to use in the export panel. Click a card to select it.
      </p>
      <div className="grid grid-cols-2 gap-3">
        {VARIANT_NAMES.map((name) => (
          <VariantCard
            key={name}
            name={name as VariantName}
            isSelected={selectedVariant === name}
            onSelect={setSelectedVariant}
          />
        ))}
      </div>
    </div>
  );
}
