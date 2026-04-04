import { useState, useCallback, useRef, useEffect } from 'react';

interface SwipeDividerState {
  /** Current X position of the divider in pixels (from left edge of container) */
  position: number;
  /** Whether the user is currently dragging */
  isDragging: boolean;
}

interface UseSwipeDividerReturn {
  /** Current position in pixels */
  position: number;
  /** Whether currently dragging */
  isDragging: boolean;
  /** Attach to the container element that bounds the divider */
  containerRef: React.RefObject<HTMLDivElement | null>;
  /** Attach to the divider handle's onPointerDown */
  onPointerDown: (e: React.PointerEvent) => void;
}

export function useSwipeDivider(initialFraction = 0.5): UseSwipeDividerReturn {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<SwipeDividerState>({
    position: 0,
    isDragging: false,
  });
  const fractionRef = useRef(initialFraction);

  // Initialize position and respond to container resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updatePosition = () => {
      const width = container.getBoundingClientRect().width;
      if (width > 0) {
        setState((s) => ({
          ...s,
          position: width * fractionRef.current,
        }));
      }
    };

    // Set initial position
    updatePosition();

    // Respond to container resize
    const observer = new ResizeObserver(() => {
      if (!state.isDragging) {
        // Recalculate from current fraction, not initial
        const width = container.getBoundingClientRect().width;
        if (width > 0) {
          setState((s) => {
            const currentFraction = width > 0 ? s.position / width : fractionRef.current;
            return { ...s, position: width * currentFraction };
          });
        }
      }
    });
    observer.observe(container);

    return () => observer.disconnect();
  }, [state.isDragging]);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setState((s) => ({ ...s, isDragging: true }));

    const onPointerMove = (moveEvent: PointerEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(moveEvent.clientX - rect.left, rect.width));
      setState((s) => ({ ...s, position: x }));
    };

    const onPointerUp = () => {
      setState((s) => ({ ...s, isDragging: false }));
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
    };

    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', onPointerUp);
  }, []);

  return {
    position: state.position,
    isDragging: state.isDragging,
    containerRef,
    onPointerDown,
  };
}
