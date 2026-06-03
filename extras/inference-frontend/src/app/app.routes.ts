import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: '/maskedlm', pathMatch: 'full' },
  {
    path: 'maskedlm',
    loadComponent: () => import('./features/maskedlm/maskedlm').then((m) => m.Maskedlm),
  },
  { path: '**', redirectTo: '/maskedlm' },
];
