import { useState } from 'react';
import { API_BASE_URL } from '@/constants';

interface ComparisonImagesProps {
  images: Record<string, string | null>;
}

const YEARS = ['2021', '2023'] as const;
const QUADRANTS = ['nw', 'ne', 'sw', 'se'] as const;

export function ComparisonImages({ images }: ComparisonImagesProps) {
  // Track which images failed to load
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());

  const handleImageError = (key: string) => {
    setFailedImages((prev) => new Set(prev).add(key));
  };

  const hasAnyImage = Object.values(images).some((url) => url != null);
  if (!hasAnyImage) return null;

  return (
    <div className="space-y-4">
      <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
        Comparison Images
      </h4>
      {YEARS.map((year) => {
        const fullKey = `${year}_full`;
        const fullUrl = images[fullKey];
        const hasQuadrants = QUADRANTS.some((q) => images[`${year}_${q}`] != null);

        if (!fullUrl && !hasQuadrants) return null;

        return (
          <div key={year}>
            <h5 className="text-sm font-medium mb-2">{year} Comparison</h5>
            {/* Full view */}
            {fullUrl && !failedImages.has(fullKey) && (
              <img
                src={`${API_BASE_URL}${fullUrl}`}
                alt={`${year} full comparison`}
                className="w-full rounded-md border"
                loading="lazy"
                onError={() => handleImageError(fullKey)}
              />
            )}
            {/* Quadrant grid */}
            {hasQuadrants && (
              <div className="grid grid-cols-2 gap-1 mt-2">
                {QUADRANTS.map((quad) => {
                  const key = `${year}_${quad}`;
                  const url = images[key];
                  if (!url || failedImages.has(key)) return null;
                  return (
                    <img
                      key={quad}
                      src={`${API_BASE_URL}${url}`}
                      alt={`${year} ${quad} quadrant`}
                      className="w-full rounded-sm border"
                      loading="lazy"
                      onError={() => handleImageError(key)}
                    />
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
