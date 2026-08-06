import { Injectable } from '@angular/core';
import { CompletionServiceBase } from './completion.service';

@Injectable({ providedIn: 'root' })
export class FnamingCompletionService extends CompletionServiceBase {
  constructor() {
    super('/api/fnaming/completion/');
  }
}
