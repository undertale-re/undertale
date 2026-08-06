import { Component, output } from '@angular/core';

@Component({
  selector: 'app-assembly-format-modal',
  templateUrl: './assembly-format-modal.html',
})
export class AssemblyFormatModal {
  readonly dismissed = output<void>();

  dismiss(): void {
    this.dismissed.emit();
  }
}
