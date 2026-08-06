import { Component, inject, OnInit, signal } from '@angular/core';
import { FnamingCompletionService } from '../../core/services/fnaming-completion.service';
import { Sidebar } from './components/sidebar/sidebar';
import { NewCompletion } from './components/new-completion/new-completion';
import { CompletionDetail } from './components/completion-detail/completion-detail';

@Component({
  selector: 'app-fnaming',
  imports: [Sidebar, NewCompletion, CompletionDetail],
  templateUrl: './fnaming.html',
  styleUrl: './fnaming.css',
})
export class Fnaming implements OnInit {
  protected readonly completionService = inject(FnamingCompletionService);
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
