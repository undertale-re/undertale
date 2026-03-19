import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class EulaService {
  hasSeenEula(username: string): boolean {
    return localStorage.getItem(`eula_seen_${username}`) === 'true';
  }

  markSeen(username: string): void {
    localStorage.setItem(`eula_seen_${username}`, 'true');
  }
}
