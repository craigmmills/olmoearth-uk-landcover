import { LANDCOVER_CLASSES, LANDCOVER_COLORS } from '@/constants';

export default function Legend() {
  return (
    <div className="absolute bottom-8 left-2 bg-white/90 rounded-md shadow-md p-2 text-xs z-10">
      <div className="font-semibold mb-1">Landcover</div>
      {Object.entries(LANDCOVER_CLASSES).map(([key, label]) => (
        <div key={key} className="flex items-center gap-1.5 py-0.5">
          <span
            className="inline-block w-3 h-3 rounded-sm border border-gray-300"
            style={{ backgroundColor: LANDCOVER_COLORS[Number(key)] }}
          />
          <span>{label}</span>
        </div>
      ))}
    </div>
  );
}
