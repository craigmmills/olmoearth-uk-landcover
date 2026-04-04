import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import LoopStatus from './LoopStatus';
import type { LoopStatus as LoopStatusType } from '@/types';

describe('LoopStatus', () => {
  it('shows connecting state', () => {
    render(<LoopStatus status={{ state: 'connecting' }} />);
    expect(screen.getByText(/Connecting to event stream/)).toBeTruthy();
  });

  it('shows running state with iteration number', () => {
    const status: LoopStatusType = {
      state: 'running',
      sessionId: 'session_1',
      currentIteration: 3,
    };
    render(<LoopStatus status={status} />);
    expect(screen.getByText(/Iteration 3 running/)).toBeTruthy();
  });

  it('shows "Starting up..." when iteration is 0', () => {
    const status: LoopStatusType = {
      state: 'running',
      sessionId: 'session_1',
      currentIteration: 0,
    };
    render(<LoopStatus status={status} />);
    expect(screen.getByText(/Starting up/)).toBeTruthy();
  });

  it('shows complete state with score', () => {
    const status: LoopStatusType = {
      state: 'complete',
      sessionId: 'session_1',
      nIterations: 5,
      bestScore: 0.87,
      stopReason: 'target_reached',
    };
    render(<LoopStatus status={status} />);
    expect(screen.getByText(/Loop complete/)).toBeTruthy();
    expect(screen.getByText(/0\.87/)).toBeTruthy();
    expect(screen.getByText(/5 iterations/)).toBeTruthy();
  });

  it('shows complete state without score when bestScore is null', () => {
    const status: LoopStatusType = {
      state: 'complete',
      sessionId: 'session_1',
      nIterations: 5,
      bestScore: null,
      stopReason: 'max_iterations',
    };
    render(<LoopStatus status={status} />);
    expect(screen.getByText(/5 iterations/)).toBeTruthy();
    expect(screen.queryByText(/best score/)).toBeNull();
  });

  it('shows idle state', () => {
    render(<LoopStatus status={{ state: 'idle' }} />);
    expect(screen.getByText(/No active loop/)).toBeTruthy();
  });

  it('shows error state with message', () => {
    render(<LoopStatus status={{ state: 'error', message: 'Something broke' }} />);
    expect(screen.getByText(/Something broke/)).toBeTruthy();
  });
});
