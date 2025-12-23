# Mobile Responsiveness Strategy & Implementation Plan

## 1. Executive Summary
The goal is to transform the CIFT Markets platform into a "Mobile-First" experience without sacrificing the professional, data-dense nature of the desktop application. We will adopt a **"Progressive Disclosure"** strategy: showing the most critical information immediately, while tucking secondary details behind intuitive interactions (tabs, drawers, modals).

## 2. Global Layout Strategy
The current `MainLayout` uses a sidebar drawer, which is a good start. We will enhance this with:

*   **Bottom Navigation Bar (Mobile Only)**: For primary actions (Dashboard, Trading, Portfolio, Alerts). This is more ergonomic than a top hamburger menu for frequent navigation.
*   **Smart Header**:
    *   **Desktop**: Full search, stats, profile, notifications.
    *   **Mobile**: Logo, Search Icon (expands), Notification Badge, Hamburger Menu.
    *   **Action**: Hide market session/latency details on mobile header, move to a "Status" drawer or bottom of the menu.
*   **Typography**: Adjust base font sizes and line heights for readability on small screens.

## 3. Component-Level Strategy

### A. Data Tables (Positions, Orders)
*   **Problem**: Wide tables with many columns break mobile layouts.
*   **Solution**: "Card View" transformation.
    *   On mobile, hide the `<table>`.
    *   Show a list of `<div class="card">` elements.
    *   Each card displays the Symbol, Price, and P&L prominently.
    *   Secondary details (Time, ID, Type) are shown in a smaller font or expanded on tap.

### B. Charts
*   **Problem**: Charts need width.
*   **Solution**:
    *   Default to a fixed height (e.g., 300px) on mobile.
    *   Add a "Fullscreen" button that rotates the chart to landscape mode or fills the screen.
    *   Disable scroll-to-zoom on mobile by default to prevent page scroll blocking.

### C. Forms (Order Ticket)
*   **Problem**: Complex forms with side-by-side inputs are cramped.
*   **Solution**:
    *   Stack all inputs vertically.
    *   Use native number pads for numeric inputs.
    *   Move "Order Confirmation" to a full-screen bottom sheet instead of a center modal.

## 4. Page-Specific Implementation Plans

### Phase 1: Core Navigation & Dashboard (High Priority)
**Target Files**: `MainLayout.tsx`, `Header.tsx`, `Sidebar.tsx`, `DashboardPage.tsx`

1.  **Layout**:
    *   Implement `BottomNav` component for mobile.
    *   Adjust `MainLayout` to render `BottomNav` only on `< md` screens.
2.  **Dashboard**:
    *   **Market Ticker**: Hide on mobile or make it a single scrolling line.
    *   **Portfolio Card**: Stack "Total Equity" and "Day P&L". Convert the "Mini Stats Grid" (Buying Power, Cash, etc.) into a 2x2 grid.
    *   **Grid Layout**: Ensure `grid-cols-1` is applied correctly.
    *   **Positions Table**: Convert to "Card List" view for mobile.
    *   **Widgets**: Move News/Calendar to a separate tab or below the fold.

### Phase 2: Trading Page (Complex)
**Target Files**: `TradingPage.tsx`, `OrderBook.tsx`, `TimeSales.tsx`

1.  **Layout**:
    *   **Desktop**: 3-Column (L2 | Chart | Ticket).
    *   **Mobile**: Tabbed Interface.
        *   Tab 1: **Chart & Ticket** (Primary Action).
        *   Tab 2: **Order Book (L2)**.
        *   Tab 3: **Recent Trades (Time & Sales)**.
        *   Tab 4: **My Orders/Positions**.
2.  **Order Ticket**:
    *   Make it sticky at the bottom or easily accessible via a "Trade" FAB (Floating Action Button).

### Phase 3: Portfolio & Analytics
**Target Files**: `PortfolioPage.tsx`, `AnalyticsPage.tsx`

1.  **Portfolio**:
    *   Similar "Card View" treatment for the main holdings table.
    *   Donut charts: Ensure legends wrap correctly.
2.  **Analytics**:
    *   Stack performance metrics.
    *   Ensure date pickers are touch-friendly.

## 5. Implementation Steps (Immediate)

1.  **Step 1**: Create `MobileNav.tsx` and integrate into `MainLayout`.
2.  **Step 2**: Refactor `Header.tsx` to be cleaner on mobile.
3.  **Step 3**: Update `DashboardPage.tsx` with the "Card View" logic for positions.
4.  **Step 4**: Apply responsive grid fixes to `DashboardPage` widgets.

## 6. UX Guidelines
*   **Touch Targets**: Minimum 44x44px for all clickable elements.
*   **Spacing**: Increase padding between list items.
*   **Feedback**: Ensure active states (taps) have immediate visual feedback.
