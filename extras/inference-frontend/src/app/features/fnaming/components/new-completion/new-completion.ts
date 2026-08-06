import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FnamingCompletionService } from '../../../../core/services/fnaming-completion.service';
import { joinInstructionLines } from '../../../../core/utils/pretoken';
import { AssemblyFormatModal } from '../../../../shared/components/assembly-format-modal/assembly-format-modal';

@Component({
  selector: 'app-new-completion',
  imports: [FormsModule, AssemblyFormatModal],
  templateUrl: './new-completion.html',
})
export class NewCompletion {
  private readonly completionService = inject(FnamingCompletionService);

  protected input = signal('');
  protected submitting = signal(false);
  protected showFormatModal = signal(false);

  submit(): void {
    const text = this.input().trim();
    if (!text) return;
    this.submitting.set(true);
    this.completionService.create(joinInstructionLines(text));
    this.input.set('');
    this.submitting.set(false);
  }
}
