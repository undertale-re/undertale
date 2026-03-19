export enum CompletionState {
  Queued = 0,
  Running = 1,
  Complete = 2,
}

export interface Completion {
  id: number;
  input: string;
  output: string | null;
  timestamp: string;
  completed: boolean;
  username: string;
  rating: number | null;
  comments: string | null;
  state: CompletionState;
}

export function stateBadgeClass(state: CompletionState): string {
  switch (state) {
    case CompletionState.Queued:
      return 'badge bg-primary';
    case CompletionState.Running:
      return 'badge bg-warning text-dark';
    case CompletionState.Complete:
      return 'badge bg-secondary';
  }
}

export function stateLabel(state: CompletionState): string {
  switch (state) {
    case CompletionState.Queued:
      return 'Queued';
    case CompletionState.Running:
      return 'Running';
    case CompletionState.Complete:
      return 'Complete';
  }
}
