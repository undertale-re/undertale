import { Component, inject, OnInit, signal } from '@angular/core';
import { CompletionService } from '../../core/services/completion.service';
import { Sidebar } from './components/sidebar/sidebar';
import { NewCompletion } from './components/new-completion/new-completion';
import { CompletionDetail } from './components/completion-detail/completion-detail';

@Component({
  selector: 'app-maskedlm',
  imports: [Sidebar, NewCompletion, CompletionDetail],
  templateUrl: './maskedlm.html',
  styleUrl: './maskedlm.css',
})
export class Maskedlm implements OnInit {
  protected readonly completionService = inject(CompletionService);
  protected readonly showSidebar = signal(false);

  ngOnInit(): void {
    this.completionService.load();
  }

  toggleSidebar(): void {
    this.showSidebar.update((v) => !v);
  }

  closeSidebar(): void {
    this.showSidebar.set(false);
  }
}
