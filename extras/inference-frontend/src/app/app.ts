import { Component, computed, inject, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AuthService } from './core/services/auth.service';
import { CompletionService } from './core/services/completion.service';
import { EulaService } from './core/services/eula.service';
import { Login } from './features/login/login';
import { EulaModal } from './shared/components/eula-modal/eula-modal';
import { ConfirmModal } from './shared/components/confirm-modal/confirm-modal';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, Login, EulaModal, ConfirmModal],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly auth = inject(AuthService);
  private readonly completionService = inject(CompletionService);
  private readonly eulaService = inject(EulaService);

  protected readonly version = '0.1.0';
  protected showEula = signal(false);
  protected showLogoutConfirm = signal(false);

  protected readonly isAuthenticated = computed(() => this.auth.isAuthenticated());

  onLoginSuccess(): void {
    const username = this.auth.username();
    if (username && !this.eulaService.hasSeenEula(username)) {
      this.showEula.set(true);
    }
  }

  onEulaDismissed(): void {
    const username = this.auth.username();
    if (username) {
      this.eulaService.markSeen(username);
    }
    this.showEula.set(false);
  }

  confirmLogout(): void {
    this.showLogoutConfirm.set(true);
  }

  onLogoutConfirmed(): void {
    this.showLogoutConfirm.set(false);
    this.completionService.reset();
    this.auth.logout();
  }

  onLogoutCancelled(): void {
    this.showLogoutConfirm.set(false);
  }

  openEula(): void {
    this.showEula.set(true);
  }
}
