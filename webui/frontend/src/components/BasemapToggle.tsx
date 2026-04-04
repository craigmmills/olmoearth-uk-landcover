import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import type { BasemapType } from '@/types';

interface BasemapToggleProps {
  basemap: BasemapType;
  onChange: (basemap: BasemapType) => void;
}

export default function BasemapToggle({ basemap, onChange }: BasemapToggleProps) {
  return (
    <div className="flex items-center justify-between">
      <Label htmlFor="basemap-toggle" className="text-sm">
        {basemap === 'satellite' ? 'Satellite' : 'OpenStreetMap'}
      </Label>
      <Switch
        id="basemap-toggle"
        checked={basemap === 'satellite'}
        onCheckedChange={(checked: boolean) => onChange(checked ? 'satellite' : 'osm')}
      />
    </div>
  );
}
