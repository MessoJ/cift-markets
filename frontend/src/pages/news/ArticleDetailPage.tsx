/**
 * NEWS ARTICLE DETAIL PAGE
 * Displays full article content
 */

import { createSignal, onMount, Show } from 'solid-js';
import { useParams, useNavigate } from '@solidjs/router';
import { ArrowLeft, ExternalLink, Calendar, Tag } from 'lucide-solid';
import { apiClient, NewsArticle } from '../../lib/api/client';

export default function ArticleDetailPage() {
  const params = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = createSignal(true);
  const [article, setArticle] = createSignal<NewsArticle | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  onMount(async () => {
    try {
      const data = await apiClient.getNewsArticle(params.id);
      setArticle(data);
    } catch (err: any) {
      console.error('Failed to load article:', err);
      setError(err?.response?.data?.detail || 'Failed to load article');
    } finally {
      setLoading(false);
    }
  });

  const getSentimentColor = (sentiment: string | undefined | null) => {
    switch (sentiment) {
      case 'positive': return 'text-success-500 bg-success-500/10';
      case 'negative': return 'text-danger-500 bg-danger-500/10';
      default: return 'text-gray-400 bg-gray-800/50';
    }
  };

  return (
    <div class="h-full overflow-auto p-3 sm:p-6">
      <div class="max-w-4xl mx-auto">
        {/* Back Button */}
        <button
          onClick={() => navigate('/news')}
          class="flex items-center gap-2 text-gray-400 hover:text-white mb-4 sm:mb-6 transition-colors"
        >
          <ArrowLeft size={20} />
          <span>Back to News</span>
        </button>

        <Show when={loading()}>
          <div class="flex items-center justify-center py-12">
            <div class="text-center">
              <div class="spinner w-12 h-12 mx-auto mb-4" />
              <p class="text-gray-400">Loading article...</p>
            </div>
          </div>
        </Show>

        <Show when={error()}>
          <div class="bg-danger-500/10 border border-danger-500 rounded-lg p-6 text-center">
            <p class="text-danger-500 font-semibold mb-2">Error Loading Article</p>
            <p class="text-gray-400">{error()}</p>
          </div>
        </Show>

        <Show when={!loading() && !error() && article()}>
          <article class="bg-terminal-900 border border-terminal-750 rounded-lg overflow-hidden">
            {/* Article Image */}
            <Show when={article()!.image_url}>
              <div class="w-full h-48 sm:h-64 md:h-96 bg-terminal-850">
                <img
                  src={article()!.image_url!}
                  alt={article()!.title}
                  class="w-full h-full object-cover"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                  }}
                />
              </div>
            </Show>

            <div class="p-4 sm:p-6 md:p-8">
              {/* Tags */}
              <div class="flex items-center gap-2 mb-4">
                <span class={`px-3 py-1 rounded text-sm font-semibold capitalize ${getSentimentColor(article()!.sentiment)}`}>
                  {article()!.sentiment || 'neutral'}
                </span>
                <span class="px-3 py-1 rounded text-sm font-semibold capitalize bg-terminal-850 text-gray-400">
                  {article()!.category}
                </span>
                <Show when={article()!.symbols && Array.isArray(article()!.symbols) && article()!.symbols.length > 0}>
                  <div class="flex items-center gap-1 ml-2">
                    <Tag size={14} class="text-gray-500" />
                    {article()!.symbols.slice(0, 5).map((symbol) => (
                      <span class="px-2 py-1 bg-terminal-850 rounded text-xs font-mono text-accent-500">
                        {symbol}
                      </span>
                    ))}
                  </div>
                </Show>
              </div>

              {/* Title */}
              <h1 class="text-xl sm:text-2xl md:text-3xl font-bold text-white mb-4">
                {article()!.title}
              </h1>

              {/* Meta Info */}
              <div class="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-4 text-sm text-gray-400 pb-4 sm:pb-6 border-b border-terminal-750 mb-4 sm:mb-6">
                <span class="font-semibold">{article()!.source}</span>
                <span>•</span>
                <div class="flex items-center gap-2">
                  <Calendar size={14} />
                  <span>{new Date(article()!.published_at).toLocaleString()}</span>
                </div>
              </div>

              {/* Summary */}
              <div class="text-base sm:text-lg text-gray-300 mb-4 sm:mb-6 leading-relaxed">
                {article()!.summary}
              </div>

              {/* Content */}
              <Show when={(article() as any)?.content}>
                <div class="text-gray-400 leading-relaxed mb-6 sm:mb-8 whitespace-pre-wrap">
                  {(article() as any).content}
                </div>
              </Show>

              {/* External Link */}
              <a
                href={article()!.url}
                target="_blank"
                rel="noopener noreferrer"
                class="inline-flex items-center gap-2 px-6 py-3 bg-accent-500 hover:bg-accent-600 text-white font-semibold rounded-lg transition-colors"
              >
                <span>Read Full Article</span>
                <ExternalLink size={16} />
              </a>
            </div>
          </article>
        </Show>
      </div>
    </div>
  );
}
