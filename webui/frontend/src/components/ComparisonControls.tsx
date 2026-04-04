import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { COMPARISON_MODES } from '@/constants';
import type { ComparisonMode } from '@/types';

interface ComparisonControlsProps {
  enabled: boolean;
  onToggle: () => void;
  mode: ComparisonMode;
  onModeChange: (mode: ComparisonMode) => void;
}

export default function ComparisonControls({
  enabled,
  onToggle,
  mode,
  onModeChange,
}: ComparisonControlsProps) {
  return (
    <div className="flex flex-col gap-3">
      <Separator />
      <div className="flex items-center justify-between">
        <Label htmlFor="comparison-toggle" className="text-sm font-medium">
          Comparison Mode
        </Label>
        <Switch
          id="comparison-toggle"
          checked={enabled}
          onCheckedChange={onToggle}
        />
      </div>

      {enabled && (
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="comparison-mode" className="text-xs text-muted-foreground">
            Compare
          </Label>
          <Select value={mode} onValueChange={(value) => onModeChange(value as ComparisonMode)}>
            <SelectTrigger id="comparison-mode">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {COMPARISON_MODES.map((m) => (
                <SelectItem key={m.value} value={m.value}>
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </div>
  );
}
