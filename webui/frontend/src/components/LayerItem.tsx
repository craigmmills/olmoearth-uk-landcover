import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import type { LayerState } from '@/types';

interface LayerItemProps {
  layer: LayerState;
  onToggle: () => void;
  onOpacityChange: (opacity: number) => void;
}

export default function LayerItem({ layer, onToggle, onOpacityChange }: LayerItemProps) {
  return (
    <div className="flex flex-col gap-1.5 py-2">
      <div className="flex items-center justify-between">
        <Label htmlFor={`toggle-${layer.id}`} className="text-sm">
          {layer.label}
        </Label>
        <Switch
          id={`toggle-${layer.id}`}
          checked={layer.visible}
          onCheckedChange={onToggle}
        />
      </div>

      {layer.visible && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-14">
            Opacity
          </span>
          <Slider
            min={0}
            max={100}
            step={1}
            value={[Math.round(layer.opacity * 100)]}
            onValueChange={([val]: number[]) => onOpacityChange(val / 100)}
            className="flex-1"
          />
          <span className="text-xs text-muted-foreground w-8 text-right">
            {Math.round(layer.opacity * 100)}%
          </span>
        </div>
      )}
    </div>
  );
}
