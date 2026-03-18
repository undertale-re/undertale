import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../../../core/services/auth.service';
import { CompletionService } from '../../../../core/services/completion.service';
import { Completion, stateBadgeClass, stateLabel } from '../../../../core/models/completion.model';

@Component({
  selector: 'app-sidebar',
  imports: [FormsModule],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
})
export class Sidebar {
  protected readonly auth = inject(AuthService);
  protected readonly completionService = inject(CompletionService);

  protected search = signal('');

  protected readonly isAdmin = computed(() => {
    const username = this.auth.username();
    return this.completionService.completions().some((c) => c.username !== username);
  });

  protected readonly filtered = computed(() => {
    const query = this.search().toLowerCase().trim();
    const completions = this.completionService.completions();
    if (!query) return completions;
    return completions.filter((c) => c.input.toLowerCase().includes(query));
  });

  protected stateBadgeClass = stateBadgeClass;
  protected stateLabel = stateLabel;

  selectNew(): void {
    this.completionService.selected.set(null);
  }

  select(completion: Completion): void {
    this.completionService.selected.set(completion);
  }

  highlight(text: string): string {
    const display = text.replace(/\s*\[NEXT\]\s*/g, ' ');
    const query = this.search().trim();
    if (!query) return display;
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return display.replace(new RegExp(`(${escaped})`, 'gi'), '<mark>$1</mark>');
  }

  isSelected(completion: Completion): boolean {
    return this.completionService.selected()?.id === completion.id;
  }
}
