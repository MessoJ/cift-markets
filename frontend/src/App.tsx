/**
 * App Component
 * 
 * Root application component with routing and global keyboard shortcuts.
 */

import { Router, Route, Navigate, useNavigate } from '@solidjs/router';
import { Show, lazy, onMount } from 'solid-js';
import { authStore } from '~/stores/auth.store';
import { MainLayout } from '~/components/layout/MainLayout';
import KeyboardShortcutsProvider, { registerShortcuts } from '~/components/ui/KeyboardShortcuts';

// Lazy load pages for code splitting
const LoginPage = lazy(() => import('~/pages/auth/LoginPage'));
const DashboardPage = lazy(() => import('~/pages/dashboard/DashboardPage'));
const TradingPage = lazy(() => import('~/pages/trading/TradingPage'));
const PortfolioPage = lazy(() => import('~/pages/portfolio/PortfolioPage'));
const AnalyticsPage = lazy(() => import('~/pages/analytics/AnalyticsPage'));
const OrdersPage = lazy(() => import('~/pages/orders/OrdersPage'));
const WatchlistsPage = lazy(() => import('~/pages/watchlists/WatchlistsPage'));
const TransactionsPage = lazy(() => import('~/pages/transactions/TransactionsPage'));
const SettingsPage = lazy(() => import('~/pages/settings/SettingsPage'));
const ProfilePage = lazy(() => import('~/pages/profile/ProfilePage'));

// New Feature Pages
const FundingPage = lazy(() => import('~/pages/funding/FundingPage'));
const FundingTransactionDetail = lazy(() => import('~/pages/funding/FundingTransactionDetail'));
const OnboardingPage = lazy(() => import('~/pages/onboarding/OnboardingPage'));
const SupportPage = lazy(() => import('~/pages/support/SupportPage'));
const TicketDetailPage = lazy(() => import('~/pages/support/TicketDetailPage'));
const ChartsPage = lazy(() => import('~/pages/charts/ChartsPage'));
const NewsPage = lazy(() => import('~/pages/news/NewsPage'));
const ArticleDetailPage = lazy(() => import('~/pages/news/ArticleDetailPage'));
const GlobePage = lazy(() => import('~/pages/globe/GlobePage'));
const StatementsPage = lazy(() => import('~/pages/statements/StatementsPage'));
const ScreenerPage = lazy(() => import('~/pages/screener/ScreenerPage'));
const AlertsPage = lazy(() => import('~/pages/alerts/AlertsPage'));
const VerifyTransactionPage = lazy(() => import('~/pages/VerifyTransactionPage'));
const SymbolDetailPage = lazy(() => import('~/pages/symbol/SymbolDetailPage'));

// Protected Route Component
function ProtectedRoute(props: { children: any }) {
  return (
    <Show
      when={!authStore.isLoading()}
      fallback={
        <div class="h-screen flex items-center justify-center bg-gray-950">
          <div class="flex flex-col items-center gap-4">
            <div class="w-10 h-10 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            <p class="text-gray-400 text-sm font-mono">Loading authentication...</p>
          </div>
        </div>
      }
    >
      <Show
        when={authStore.isAuthenticated()}
        fallback={<Navigate href="/auth/login" />}
      >
        <MainLayout>
          <GlobalShortcuts />
          {props.children}
        </MainLayout>
      </Show>
    </Show>
  );
}

// Public Route Component (redirect if authenticated)
function PublicRoute(props: { children: any }) {
  return (
    <Show
      when={!authStore.isLoading()}
      fallback={
        <div class="h-screen flex items-center justify-center bg-gray-950">
          <div class="flex flex-col items-center gap-4">
            <div class="w-10 h-10 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            <p class="text-gray-400 text-sm font-mono">Loading...</p>
          </div>
        </div>
      }
    >
      <Show
        when={!authStore.isAuthenticated()}
        fallback={<Navigate href="/dashboard" />}
      >
        {props.children}
      </Show>
    </Show>
  );
}

// Global keyboard shortcuts registration component
function GlobalShortcuts() {
  const navigate = useNavigate();
  
  onMount(() => {
    const unregister = registerShortcuts([
      // Navigation shortcuts
      { key: 'd', alt: true, description: 'Go to Dashboard', action: () => navigate('/dashboard'), category: 'Navigation' },
      { key: 't', alt: true, description: 'Go to Trading', action: () => navigate('/trading'), category: 'Navigation' },
      { key: 'c', alt: true, description: 'Go to Charts', action: () => navigate('/charts'), category: 'Navigation' },
      { key: 'p', alt: true, description: 'Go to Portfolio', action: () => navigate('/portfolio'), category: 'Navigation' },
      { key: 'a', alt: true, description: 'Go to Analytics', action: () => navigate('/analytics'), category: 'Navigation' },
      { key: 'w', alt: true, description: 'Go to Watchlists', action: () => navigate('/watchlists'), category: 'Navigation' },
      { key: 's', alt: true, description: 'Go to Screener', action: () => navigate('/screener'), category: 'Navigation' },
      { key: 'l', alt: true, description: 'Go to Alerts', action: () => navigate('/alerts'), category: 'Navigation' },
      { key: 'n', alt: true, description: 'Go to News', action: () => navigate('/news'), category: 'Navigation' },
      
      // Trading shortcuts
      { key: 'b', shift: true, description: 'Quick Buy Order', action: () => navigate('/trading', { state: { action: 'buy' } }), category: 'Trading' },
      { key: 's', shift: true, description: 'Quick Sell Order', action: () => navigate('/trading', { state: { action: 'sell' } }), category: 'Trading' },
      
      // Utility shortcuts
      { key: 'r', ctrl: true, description: 'Refresh Data', action: () => window.location.reload(), category: 'Utility' },
    ]);
    
    return unregister;
  });
  
  return null;
}

export default function App() {
  return (
    <KeyboardShortcutsProvider>
      <Router>
        {/* Public Routes */}
        <Route
          path="/auth/login"
          component={() => (
            <PublicRoute>
              <LoginPage />
            </PublicRoute>
          )}
        />
      
      {/* Public Verification Route (no auth required) */}
      <Route
        path="/verify/:id"
        component={() => <VerifyTransactionPage />}
      />

      {/* Protected Routes */}
      <Route
        path="/dashboard"
        component={() => (
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/trading"
        component={() => (
          <ProtectedRoute>
            <TradingPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/portfolio"
        component={() => (
          <ProtectedRoute>
            <PortfolioPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/analytics"
        component={() => (
          <ProtectedRoute>
            <AnalyticsPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/orders"
        component={() => (
          <ProtectedRoute>
            <OrdersPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/watchlists"
        component={() => (
          <ProtectedRoute>
            <WatchlistsPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/transactions"
        component={() => (
          <ProtectedRoute>
            <TransactionsPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/settings"
        component={() => (
          <ProtectedRoute>
            <SettingsPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/profile"
        component={() => (
          <ProtectedRoute>
            <ProfilePage />
          </ProtectedRoute>
        )}
      />

      {/* Funding Routes */}
      <Route
        path="/funding"
        component={() => (
          <ProtectedRoute>
            <FundingPage />
          </ProtectedRoute>
        )}
      />

      <Route
        path="/funding/transactions/:id"
        component={() => (
          <ProtectedRoute>
            <FundingTransactionDetail />
          </ProtectedRoute>
        )}
      />

      {/* Support Routes */}
      <Route
        path="/support"
        component={() => (
          <ProtectedRoute>
            <SupportPage />
          </ProtectedRoute>
        )}
      />
      <Route
        path="/support/tickets/:id"
        component={() => (
          <ProtectedRoute>
            <TicketDetailPage />
          </ProtectedRoute>
        )}
      />

      {/* Charts Routes */}
      <Route
        path="/charts"
        component={() => (
          <ProtectedRoute>
            <ChartsPage />
          </ProtectedRoute>
        )}
      />

      {/* News Routes */}
      <Route
        path="/news"
        component={() => (
          <ProtectedRoute>
            <NewsPage />
          </ProtectedRoute>
        )}
      />
      <Route
        path="/news/:id"
        component={() => (
          <ProtectedRoute>
            <ArticleDetailPage />
          </ProtectedRoute>
        )}
      />

      {/* Globe Route */}
      <Route
        path="/globe"
        component={() => (
          <ProtectedRoute>
            <GlobePage />
          </ProtectedRoute>
        )}
      />

      {/* Statements Routes */}
      <Route
        path="/statements"
        component={() => (
          <ProtectedRoute>
            <StatementsPage />
          </ProtectedRoute>
        )}
      />

      {/* Screener Routes */}
      <Route
        path="/screener"
        component={() => (
          <ProtectedRoute>
            <ScreenerPage />
          </ProtectedRoute>
        )}
      />

      {/* Alerts Routes */}
      <Route
        path="/alerts"
        component={() => (
          <ProtectedRoute>
            <AlertsPage />
          </ProtectedRoute>
        )}
      />

      {/* Onboarding (Public - before account approval) */}
      <Route
        path="/onboarding"
        component={() => <OnboardingPage />}
      />

      {/* Symbol Detail Route */}
      <Route
        path="/symbol/:symbol"
        component={() => (
          <ProtectedRoute>
            <SymbolDetailPage />
          </ProtectedRoute>
        )}
      />

      {/* Default redirect */}
      <Route path="/" component={() => <Navigate href="/dashboard" />} />
      <Route path="*" component={() => <Navigate href="/dashboard" />} />
      </Router>
    </KeyboardShortcutsProvider>
  );
}
