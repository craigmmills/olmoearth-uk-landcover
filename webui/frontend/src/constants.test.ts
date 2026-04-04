import { describe, it, expect } from 'vitest';
import { LANDCOVER_CLASSES, LANDCOVER_COLORS, AOI_CENTER, DEFAULT_LAYERS, COMPARISON_MODES } from './constants';

describe('LANDCOVER_CLASSES', () => {
  it('has 6 entries (classes 0-5)', () => {
    expect(Object.keys(LANDCOVER_CLASSES)).toHaveLength(6);
  });

  it('has all expected class names', () => {
    expect(LANDCOVER_CLASSES[0]).toBe('Built-up');
    expect(LANDCOVER_CLASSES[4]).toBe('Water');
  });
});

describe('LANDCOVER_COLORS', () => {
  it('has 6 entries matching class keys', () => {
    expect(Object.keys(LANDCOVER_COLORS)).toHaveLength(6);
    for (const key of Object.keys(LANDCOVER_CLASSES)) {
      expect(LANDCOVER_COLORS).toHaveProperty(key);
    }
  });

  it('has valid hex color values', () => {
    for (const color of Object.values(LANDCOVER_COLORS)) {
      expect(color).toMatch(/^#[0-9A-Fa-f]{6}$/);
    }
  });
});

describe('AOI_CENTER', () => {
  it('has valid coordinates for Exeter, UK', () => {
    expect(AOI_CENTER.latitude).toBe(50.72);
    expect(AOI_CENTER.longitude).toBe(-3.50);
  });
});

describe('DEFAULT_LAYERS', () => {
  it('has 3 layers (2021, 2023, change)', () => {
    expect(DEFAULT_LAYERS).toHaveLength(3);
  });

  it('has only 2021 visible by default', () => {
    const visible = DEFAULT_LAYERS.filter((l) => l.visible);
    expect(visible).toHaveLength(1);
    expect(visible[0].id).toBe('landcover-2021');
  });
});

describe('COMPARISON_MODES', () => {
  it('has 3 comparison modes', () => {
    expect(COMPARISON_MODES).toHaveLength(3);
  });

  it('each mode has a value and label', () => {
    for (const mode of COMPARISON_MODES) {
      expect(mode.value).toBeTruthy();
      expect(mode.label).toBeTruthy();
    }
  });
});
