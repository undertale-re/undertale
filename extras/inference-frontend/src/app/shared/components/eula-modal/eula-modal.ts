import { Component, output } from '@angular/core';

@Component({
  selector: 'app-eula-modal',
  templateUrl: './eula-modal.html',
})
export class EulaModal {
  readonly dismissed = output<void>();

  dismiss(): void {
    this.dismissed.emit();
  }
}
