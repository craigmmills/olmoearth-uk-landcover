import LayerItem from '@/components/LayerItem';
import { Separator } from '@/components/ui/separator';
import type { LayerState } from '@/types';

interface LayerControlsProps {
  layers: LayerState[];
  onToggle: (layerId: string) => void;
  onOpacityChange: (layerId: string, opacity: number) => void;
}

export default function LayerControls({ layers, onToggle, onOpacityChange }: LayerControlsProps) {
  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-sm font-medium text-muted-foreground">Overlay Layers</h2>
      <Separator />
      {layers.map((layer) => (
        <LayerItem
          key={layer.id}
          layer={layer}
          onToggle={() => onToggle(layer.id)}
          onOpacityChange={(val) => onOpacityChange(layer.id, val)}
        />
      ))}
      {layers.length === 0 && (
        <div className="text-sm text-muted-foreground">No layers available</div>
      )}
    </div>
  );
}
