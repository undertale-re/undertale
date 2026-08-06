import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', loadComponent: () => import('./features/landing/landing').then((m) => m.Landing) },
  {
    path: 'maskedlm',
    loadComponent: () => import('./features/maskedlm/maskedlm').then((m) => m.Maskedlm),
  },
  {
    path: 'fnaming',
    loadComponent: () => import('./features/fnaming/fnaming').then((m) => m.Fnaming),
  },
  { path: '**', redirectTo: '/' },
];
