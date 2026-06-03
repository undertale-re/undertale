import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { Credentials, TokenResponse } from '../models/auth.model';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http = inject(HttpClient);

  readonly token = signal<string | null>(localStorage.getItem('token'));
  readonly username = signal<string | null>(this.decodeUsername(localStorage.getItem('token')));
  readonly isAdmin = signal<boolean>(localStorage.getItem('admin') === 'true');
  readonly isAuthenticated = computed(() => this.token() !== null);

  private decodeUsername(token: string | null): string | null {
    if (!token) return null;
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.username ?? payload.sub ?? null;
    } catch {
      return null;
    }
  }

  login(credentials: Credentials): Observable<TokenResponse> {
    return this.http.post<TokenResponse>('/api/login/', credentials).pipe(
      tap((response) => {
        localStorage.setItem('token', response.token);
        localStorage.setItem('admin', String(response.admin));
        this.token.set(response.token);
        this.username.set(this.decodeUsername(response.token));
        this.isAdmin.set(response.admin);
      }),
    );
  }

  logout(): void {
    localStorage.removeItem('token');
    localStorage.removeItem('admin');
    this.token.set(null);
    this.username.set(null);
    this.isAdmin.set(false);
  }
}
