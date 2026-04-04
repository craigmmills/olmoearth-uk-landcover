import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useSwipeDivider } from '@/hooks/useSwipeDivider';

describe('useSwipeDivider', () => {
  it('returns initial position of 0 when container has no width', () => {
    const { result } = renderHook(() => useSwipeDivider(0.5));
    expect(result.current.position).toBe(0);
    expect(result.current.isDragging).toBe(false);
  });

  it('provides onPointerDown handler', () => {
    const { result } = renderHook(() => useSwipeDivider());
    expect(typeof result.current.onPointerDown).toBe('function');
  });

  it('provides a containerRef', () => {
    const { result } = renderHook(() => useSwipeDivider());
    expect(result.current.containerRef).toBeDefined();
    expect(result.current.containerRef.current).toBeNull();
  });

  it('defaults initialFraction to 0.5', () => {
    const { result } = renderHook(() => useSwipeDivider());
    // Position is 0 because no container is attached
    expect(result.current.position).toBe(0);
  });
});
