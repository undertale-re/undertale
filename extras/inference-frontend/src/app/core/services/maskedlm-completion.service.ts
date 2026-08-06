import { Injectable } from '@angular/core';
import { CompletionServiceBase } from './completion.service';

@Injectable({ providedIn: 'root' })
export class MaskedlmCompletionService extends CompletionServiceBase {
  constructor() {
    super('/api/maskedlm/completion/');
  }
}
