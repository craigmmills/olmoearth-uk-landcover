import { Popup } from '@vis.gl/react-maplibre';
import { Loader2 } from 'lucide-react';
import type { PixelQueryResult } from '@/types';

interface ClickPopupProps {
  lngLat: { lng: number; lat: number };
  result: PixelQueryResult | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}

export default function ClickPopup({
  lngLat,
  result,
  loading,
  error,
  onClose,
}: ClickPopupProps) {
  return (
    <Popup
      longitude={lngLat.lng}
      latitude={lngLat.lat}
      closeOnClick={false}
      onClose={onClose}
      anchor="bottom"
      className="click-popup"
    >
      <div className="p-2 min-w-[160px]">
        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            Querying...
          </div>
        )}
        {error && (
          <div className="text-sm text-destructive">{error}</div>
        )}
        {result && !loading && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span
                className="inline-block w-4 h-4 rounded-sm border border-border"
                style={{ backgroundColor: result.color }}
              />
              <span className="font-semibold text-sm">{result.className}</span>
            </div>
            <div className="text-xs text-muted-foreground">
              {result.lng.toFixed(5)}, {result.lat.toFixed(5)}
            </div>
          </div>
        )}
      </div>
    </Popup>
  );
}
