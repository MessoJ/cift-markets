/**
 * MonthlyReturnsHeatmap Component
 * 
 * Displays a table of monthly returns for the portfolio.
 * Standard industry visualization for fund performance.
 */

import { For, Show, createMemo } from 'solid-js';
import { formatPercent } from '~/lib/utils/format';

interface MonthlyReturn {
  year: number;
  month: number; // 1-12
  value: number; // percentage (e.g., 5.2 for 5.2%)
}

interface MonthlyReturnsHeatmapProps {
  data: MonthlyReturn[];
  className?: string;
}

export function MonthlyReturnsHeatmap(props: MonthlyReturnsHeatmapProps) {
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ];

  // Group data by year
  const yearData = createMemo(() => {
    const years: Record<number, Record<number, number>> = {};
    const yearTotals: Record<number, number> = {};

    props.data.forEach(item => {
      if (!years[item.year]) {
        years[item.year] = {};
      }
      years[item.year][item.month] = item.value;
    });

    // Calculate YTD for each year
    Object.keys(years).forEach(yearStr => {
      const year = parseInt(yearStr);
      let ytd = 1;
      for (let m = 1; m <= 12; m++) {
        const val = years[year][m];
        if (val !== undefined) {
          ytd *= (1 + val / 100);
        }
      }
      yearTotals[year] = (ytd - 1) * 100;
    });

    // Sort years descending
    return Object.keys(years)
      .map(y => parseInt(y))
      .sort((a, b) => b - a)
      .map(year => ({
        year,
        months: years[year],
        total: yearTotals[year]
      }));
  });

  const getColor = (value: number | undefined) => {
    if (value === undefined) return 'bg-transparent';
    if (value === 0) return 'bg-gray-800 text-gray-400';
    
    // Green for positive, Red for negative
    // Opacity based on magnitude (capped at 10%)
    const intensity = Math.min(Math.abs(value) / 10, 1);
    
    if (value > 0) {
      // Green-500 is roughly #22c55e
      return `bg-green-500/${Math.round(intensity * 100)} text-green-100`;
    } else {
      // Red-500 is roughly #ef4444
      return `bg-red-500/${Math.round(intensity * 100)} text-red-100`;
    }
  };

  return (
    <div class={`overflow-x-auto ${props.className || ''}`}>
      <table class="w-full text-xs border-collapse">
        <thead>
          <tr>
            <th class="p-2 text-left text-gray-400 font-medium border-b border-gray-800">Year</th>
            <For each={months}>
              {(month) => (
                <th class="p-2 text-center text-gray-400 font-medium border-b border-gray-800">{month}</th>
              )}
            </For>
            <th class="p-2 text-right text-gray-400 font-bold border-b border-gray-800">YTD</th>
          </tr>
        </thead>
        <tbody>
          <For each={yearData()}>
            {(yearRow) => (
              <tr class="hover:bg-gray-800/30 transition-colors">
                <td class="p-2 text-left font-medium text-gray-300 border-b border-gray-800/50">
                  {yearRow.year}
                </td>
                <For each={Array.from({ length: 12 }, (_, i) => i + 1)}>
                  {(monthIndex) => {
                    const val = yearRow.months[monthIndex];
                    return (
                      <td class="p-1 border-b border-gray-800/50">
                        <div 
                          class={`w-full h-full py-1.5 rounded text-center font-mono ${getColor(val)}`}
                        >
                          {val !== undefined ? val.toFixed(1) + '%' : '-'}
                        </div>
                      </td>
                    );
                  }}
                </For>
                <td class={`p-2 text-right font-bold border-b border-gray-800/50 font-mono ${yearRow.total >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercent(yearRow.total)}
                </td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}
