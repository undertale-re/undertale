import { Component, computed, inject, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from './core/services/auth.service';
import { MaskedlmCompletionService } from './core/services/maskedlm-completion.service';
import { FnamingCompletionService } from './core/services/fnaming-completion.service';
import { EulaService } from './core/services/eula.service';
import { Login } from './features/login/login';
import { EulaModal } from './shared/components/eula-modal/eula-modal';
import { ConfirmModal } from './shared/components/confirm-modal/confirm-modal';
import { version } from '../environments/version';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, Login, EulaModal, ConfirmModal],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly auth = inject(AuthService);
  private readonly maskedlmCompletion = inject(MaskedlmCompletionService);
  private readonly fnamingCompletion = inject(FnamingCompletionService);
  private readonly eulaService = inject(EulaService);

  protected readonly version = version;
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
    this.maskedlmCompletion.reset();
    this.fnamingCompletion.reset();
    this.auth.logout();
  }

  onLogoutCancelled(): void {
    this.showLogoutConfirm.set(false);
  }

  openEula(): void {
    this.showEula.set(true);
  }
}
