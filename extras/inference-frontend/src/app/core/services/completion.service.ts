import { Injectable, computed, effect, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Subscription, interval } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { Completion } from '../models/completion.model';

@Injectable({ providedIn: 'root' })
export class CompletionService {
  private readonly http = inject(HttpClient);

  readonly completions = signal<Completion[]>([]);
  readonly selected = signal<Completion | null>(null);
  readonly hasPending = computed(() => this.completions().some((c) => !c.completed));

  private pollSubscription: Subscription | null = null;

  constructor() {
    effect(() => {
      if (this.hasPending()) {
        this.startPolling();
      } else {
        this.stopPolling();
      }
    });
  }

  private startPolling(): void {
    if (this.pollSubscription) return;
    this.pollSubscription = interval(3000)
      .pipe(switchMap(() => this.http.get<Completion[]>('/api/maskedlm/completion/')))
      .subscribe((completions) => {
        this.completions.set(completions);
        const currentSelected = this.selected();
        if (currentSelected) {
          const updated = completions.find((c) => c.id === currentSelected.id);
          if (updated) this.selected.set(updated);
        }
      });
  }

  private stopPolling(): void {
    this.pollSubscription?.unsubscribe();
    this.pollSubscription = null;
  }

  load(): void {
    this.http.get<Completion[]>('/api/maskedlm/completion/').subscribe((completions) => {
      this.completions.set(completions);
    });
  }

  create(input: string): void {
    this.http.post<Completion>('/api/maskedlm/completion/', { input }).subscribe((completion) => {
      this.completions.update((list) => [...list, completion]);
      this.selected.set(completion);
    });
  }

  delete(id: number): void {
    this.http.delete(`/api/maskedlm/completion/${id}/`).subscribe(() => {
      this.completions.update((list) => list.filter((c) => c.id !== id));
      if (this.selected()?.id === id) {
        this.selected.set(null);
      }
    });
  }

  submitFeedback(id: number, rating: number, comments: string): void {
    this.http
      .post(`/api/maskedlm/completion/${id}/feedback/`, { rating, comments })
      .subscribe(() => {
        const patch = (c: Completion) => (c.id === id ? { ...c, rating, comments } : c);
        this.completions.update((list) => list.map(patch));
        const selected = this.selected();
        if (selected?.id === id) this.selected.set({ ...selected, rating, comments });
      });
  }

  reset(): void {
    this.stopPolling();
    this.completions.set([]);
    this.selected.set(null);
  }
}
