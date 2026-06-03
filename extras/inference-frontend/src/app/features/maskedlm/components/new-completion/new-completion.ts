import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CompletionService } from '../../../../core/services/completion.service';
import { AssemblyFormatModal } from './assembly-format-modal';

@Component({
  selector: 'app-new-completion',
  imports: [FormsModule, AssemblyFormatModal],
  templateUrl: './new-completion.html',
})
export class NewCompletion {
  private readonly completionService = inject(CompletionService);

  protected input = signal('');
  protected submitting = signal(false);
  protected showFormatModal = signal(false);

  submit(): void {
    const text = this.input().trim();
    if (!text) return;
    this.submitting.set(true);
    const encoded = text
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .join(' [NEXT] ');
    this.completionService.create(encoded);
    this.input.set('');
    this.submitting.set(false);
  }
}
