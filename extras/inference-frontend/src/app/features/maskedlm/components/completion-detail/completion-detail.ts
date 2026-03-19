import { Component, OnDestroy, computed, effect, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { debounceTime } from 'rxjs/operators';
import { CompletionService } from '../../../../core/services/completion.service';
import { stateBadgeClass, stateLabel } from '../../../../core/models/completion.model';

interface PendingFeedback {
  id: number;
  rating: number;
  comments: string;
}

@Component({
  selector: 'app-completion-detail',
  imports: [FormsModule, DatePipe],
  templateUrl: './completion-detail.html',
  styleUrl: './completion-detail.css',
})
export class CompletionDetail implements OnDestroy {
  protected readonly completionService = inject(CompletionService);

  protected readonly completion = computed(() => this.completionService.selected());
  protected rating = signal<number | null>(null);
  protected comments = signal<string>('');
  protected showComments = computed(() => this.rating() !== null);

  private readonly commentsSubject = new Subject<string>();
  private pending: PendingFeedback | null = null;

  protected stateBadgeClass = stateBadgeClass;
  protected stateLabel = stateLabel;

  protected displayText(text: string): string {
    return text.replace(/\s*\[NEXT\]\s*/g, '\n');
  }

  protected readonly smileys = ['😞', '😕', '😐', '🙂', '😄'];

  constructor() {
    effect(() => {
      const current = this.completion();
      this.flushPending();
      this.rating.set(current?.rating ?? null);
      this.comments.set(current?.comments ?? '');
    });

    this.commentsSubject.pipe(debounceTime(1000)).subscribe((text) => {
      const c = this.completion();
      const r = this.rating();
      if (c && r !== null) {
        this.pending = null;
        this.completionService.submitFeedback(c.id, r, text);
      }
    });
  }

  private flushPending(): void {
    if (this.pending) {
      this.completionService.submitFeedback(
        this.pending.id,
        this.pending.rating,
        this.pending.comments,
      );
      this.pending = null;
    }
  }

  setRating(index: number): void {
    this.rating.set(index);
    const c = this.completion();
    if (c) {
      this.pending = null;
      this.completionService.submitFeedback(c.id, index, this.comments());
    }
  }

  onCommentsChange(text: string): void {
    this.comments.set(text);
    const c = this.completion();
    const r = this.rating();
    if (c && r !== null) {
      this.pending = { id: c.id, rating: r, comments: text };
    }
    this.commentsSubject.next(text);
  }

  ngOnDestroy(): void {
    this.flushPending();
  }
}
