/**
 * SUPPORT CENTER PAGE
 * Comprehensive help system with FAQ, tickets, and knowledge base
 */

import { createSignal, createEffect, For, Show } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import {
  HelpCircle,
  MessageSquare,
  FileText,
  Search,
  Plus,
  ChevronRight,
  Mail,
  Phone,
  MessageCircle,
  CheckCircle2,
  XCircle,
} from 'lucide-solid';
import { apiClient } from '../../lib/api/client';
import type { SupportTicket, FAQItem } from '../../lib/api/client';

type TabType = 'faq' | 'tickets' | 'contact';

export default function SupportPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = createSignal<TabType>('faq');
  const [searchQuery, setSearchQuery] = createSignal('');
  const [faqs, setFaqs] = createSignal<FAQItem[]>([]);
  const [tickets, setTickets] = createSignal<SupportTicket[]>([]);
  const [loading, setLoading] = createSignal(false);
  const [selectedCategory, setSelectedCategory] = createSignal<string>('all');
  
  // Notification state
  const [notification, setNotification] = createSignal<{type: 'success' | 'error', message: string} | null>(null);
  
  // Ticket creation state
  const [showTicketModal, setShowTicketModal] = createSignal(false);
  const [ticketSubject, setTicketSubject] = createSignal('');
  const [ticketMessage, setTicketMessage] = createSignal('');

  const categories = [
    { id: 'all', label: 'All Topics', icon: HelpCircle },
    { id: 'account', label: 'Account & Profile', icon: FileText },
    { id: 'trading', label: 'Trading & Orders', icon: MessageSquare },
    { id: 'funding', label: 'Deposits & Withdrawals', icon: FileText },
    { id: 'technical', label: 'Technical Issues', icon: HelpCircle },
    { id: 'billing', label: 'Fees & Billing', icon: FileText },
  ];

  createEffect(() => {
    loadData();
  });

  const loadData = async () => {
    console.log('🎫 Loading support data for category:', selectedCategory());
    setLoading(true);
    try {
      console.log('🌐 Fetching FAQs and tickets...');
      const [faqData, ticketData] = await Promise.all([
        apiClient.getFAQs(selectedCategory() === 'all' ? undefined : selectedCategory()),
        apiClient.getSupportTickets(),
      ]);
      console.log('✅ FAQs loaded:', faqData?.length || 0);
      console.log('✅ Tickets loaded:', ticketData?.tickets?.length || 0);
      setFaqs(faqData || []);
      setTickets(ticketData?.tickets || []);
    } catch (err: any) {
      console.error('❌ Failed to load support data:', err);
      console.error('❌ Error details:', err.message, err.response?.data);
      setFaqs([]);
      setTickets([]);
    } finally {
      setLoading(false);
      console.log('✅ Support data loading complete');
    }
  };

  const handleSearch = async () => {
    if (!searchQuery()) return;
    console.log('🔍 Searching FAQs for:', searchQuery());
    setLoading(true);
    try {
      const results = await apiClient.searchFAQs(searchQuery());
      console.log('✅ Search results:', results?.length || 0);
      setFaqs(results || []);
    } catch (err: any) {
      console.error('❌ Search failed:', err);
      console.error('❌ Error details:', err.message, err.response?.data);
      setFaqs([]);
    } finally {
      setLoading(false);
    }
  };
  
  // Auto-hide notification after 5 seconds
  createEffect(() => {
    if (notification()) {
      setTimeout(() => setNotification(null), 5000);
    }
  });
  
  const handleCreateTicket = async () => {
    const subject = ticketSubject();
    const message = ticketMessage();
    
    console.log('📝 Creating ticket:', subject);
    
    // Validation
    if (subject.length < 5) {
      setNotification({type: 'error', message: 'Subject must be at least 5 characters'});
      return;
    }
    
    if (message.length < 10) {
      setNotification({type: 'error', message: 'Message must be at least 10 characters'});
      return;
    }
    
    setLoading(true);
    try {
      const ticket = await apiClient.createSupportTicket({
        subject,
        message,
        category: 'other',
        priority: 'medium',
      });
      console.log('✅ Ticket created:', ticket);
      setNotification({type: 'success', message: `Support ticket created successfully! Ticket ID: ${ticket.id}`});
      
      // Reset form and close modal
      setTicketSubject('');
      setTicketMessage('');
      setShowTicketModal(false);
      
      await loadData(); // Reload tickets
    } catch (err: any) {
      console.error('❌ Failed to create ticket:', err);
      console.error('❌ Error details:', err.message, err.response?.data);
      
      // Parse error message
      let errorMsg = 'Failed to create ticket';
      if (err.response?.data?.detail) {
        if (Array.isArray(err.response.data.detail)) {
          errorMsg = err.response.data.detail.map((e: any) => e.msg || e.message).join(', ');
        } else if (typeof err.response.data.detail === 'string') {
          errorMsg = err.response.data.detail;
        }
      } else if (err.message) {
        errorMsg = err.message;
      }
      
      setNotification({type: 'error', message: errorMsg});
    } finally {
      setLoading(false);
    }
  };
  const filteredFAQs = () => {
    return faqs();
  };

  return (
    <div class="h-full flex flex-col gap-2 sm:gap-3 p-2 sm:p-3">
      {/* Inline Notification */}
      <Show when={notification()}>
        <div class={`p-4 rounded-lg border ${
          notification()?.type === 'success' 
            ? 'bg-success-500/10 border-success-500/30 text-success-500' 
            : 'bg-danger-500/10 border-danger-500/30 text-danger-500'
        } flex items-center justify-between animate-in fade-in slide-in-from-top-2`}>
          <div class="flex items-center gap-3">
            {notification()?.type === 'success' ? (
              <CheckCircle2 size={20} />
            ) : (
              <XCircle size={20} />
            )}
            <span class="text-sm font-semibold">{notification()?.message}</span>
          </div>
          <button 
            onClick={() => setNotification(null)}
            class="p-1 hover:bg-white/10 rounded transition-colors"
          >
            <XCircle size={16} />
          </button>
        </div>
      </Show>
      
      {/* Header */}
      <div class="bg-terminal-900 border border-terminal-750 p-4 sm:p-6">
        <div class="max-w-3xl mx-auto text-center">
          <div class="w-16 h-16 bg-accent-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
            <HelpCircle size={32} class="text-accent-500" />
          </div>
          <h1 class="text-xl sm:text-2xl md:text-3xl font-bold text-white mb-2">How can we help you?</h1>
          <p class="text-gray-400 mb-6">
            Search our knowledge base or contact our support team
          </p>

          {/* Search */}
          <div class="relative max-w-2xl mx-auto">
            <Search size={20} class="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
              type="text"
              value={searchQuery()}
              onInput={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search for help articles..."
              class="w-full bg-terminal-850 border border-terminal-750 text-white pl-12 pr-4 py-4 rounded-lg focus:outline-none focus:border-accent-500"
            />
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-3">
        <div class="bg-terminal-900 border border-terminal-750 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-primary-500/10 rounded flex items-center justify-center">
              <FileText size={20} class="text-primary-500" />
            </div>
            <div>
              <div class="text-2xl font-bold text-white tabular-nums">{faqs()?.length || 0}</div>
              <div class="text-xs text-gray-400">Help Articles</div>
            </div>
          </div>
        </div>

        <div class="bg-terminal-900 border border-terminal-750 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-success-500/10 rounded flex items-center justify-center">
              <MessageSquare size={20} class="text-success-500" />
            </div>
            <div>
              <div class="text-2xl font-bold text-white tabular-nums">
                {tickets()?.filter((t) => t.status === 'open').length || 0}
              </div>
              <div class="text-xs text-gray-400">Open Tickets</div>
            </div>
          </div>
        </div>

        <div class="bg-terminal-900 border border-terminal-750 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-warning-500/10 rounded flex items-center justify-center">
              <MessageCircle size={20} class="text-warning-500" />
            </div>
            <div>
              <div class="text-2xl font-bold text-white">24/7</div>
              <div class="text-xs text-gray-400">Support Available</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div class="flex items-center gap-1 bg-terminal-900 border border-terminal-750 p-1">
        <button
          onClick={() => setActiveTab('faq')}
          class={`flex-1 px-4 py-2 text-sm font-semibold rounded transition-colors ${
            activeTab() === 'faq'
              ? 'bg-primary-500/10 text-primary-500'
              : 'text-gray-400 hover:text-white hover:bg-terminal-800'
          }`}
        >
          <div class="flex items-center justify-center gap-2">
            <HelpCircle size={16} />
            <span class="hidden sm:inline">FAQ & Knowledge Base</span>
            <span class="sm:hidden">FAQ</span>
          </div>
        </button>
        <button
          onClick={() => setActiveTab('tickets')}
          class={`flex-1 px-4 py-2 text-sm font-semibold rounded transition-colors ${
            activeTab() === 'tickets'
              ? 'bg-success-500/10 text-success-500'
              : 'text-gray-400 hover:text-white hover:bg-terminal-800'
          }`}
        >
          <div class="flex items-center justify-center gap-2">
            <MessageSquare size={16} />
            <span class="hidden sm:inline">My Support Tickets</span>
            <span class="sm:hidden">Tickets</span>
          </div>
        </button>
        <button
          onClick={() => setActiveTab('contact')}
          class={`flex-1 px-4 py-2 text-sm font-semibold rounded transition-colors ${
            activeTab() === 'contact'
              ? 'bg-accent-500/10 text-accent-500'
              : 'text-gray-400 hover:text-white hover:bg-terminal-800'
          }`}
        >
          <div class="flex items-center justify-center gap-2">
            <Phone size={16} />
            <span class="hidden sm:inline">Contact Us</span>
            <span class="sm:hidden">Contact</span>
          </div>
        </button>
      </div>

      {/* Content */}
      <div class="flex-1 overflow-auto">
        {/* FAQ TAB */}
        <Show when={activeTab() === 'faq'}>
          <div class="grid grid-cols-1 lg:grid-cols-[250px_1fr] gap-2 sm:gap-3 h-full">
            {/* Categories Sidebar */}
            <div class="bg-terminal-900 border border-terminal-750 p-4">
              <h3 class="text-sm font-bold text-white mb-3">Categories</h3>
              <div class="space-y-1">
                <For each={categories}>
                  {(category) => (
                    <button
                      onClick={() => setSelectedCategory(category.id)}
                      class={`w-full flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors ${
                        selectedCategory() === category.id
                          ? 'bg-accent-500/10 text-accent-500'
                          : 'text-gray-400 hover:text-white hover:bg-terminal-850'
                      }`}
                    >
                      <category.icon size={16} />
                      <span>{category.label}</span>
                    </button>
                  )}
                </For>
              </div>
            </div>

            {/* FAQ List */}
            <div class="bg-terminal-900 border border-terminal-750">
              <div class="p-4 border-b border-terminal-750">
                <h3 class="text-sm font-bold text-white">
                  {selectedCategory() === 'all' ? 'All Questions' : categories.find((c) => c.id === selectedCategory())?.label}
                </h3>
              </div>
              <div class="divide-y divide-terminal-750">
                <Show when={filteredFAQs()?.length === 0}>
                  <div class="p-8 text-center text-gray-500">
                    No articles found. Try a different search or category.
                  </div>
                </Show>
                <For each={filteredFAQs() || []}>
                  {(faq) => (
                    <div class="p-4 hover:bg-terminal-850 transition-colors border-b border-terminal-750 last:border-0">
                      <h4 class="text-sm font-semibold text-white mb-2">
                        {faq.question}
                      </h4>
                      <div class="text-xs text-gray-400 leading-relaxed">
                        {faq.answer}
                      </div>
                      <div class="flex items-center gap-2 mt-3 text-xs">
                        <span class="px-2 py-1 bg-primary-500/10 text-primary-500 rounded">
                          {faq.category}
                        </span>
                      </div>
                    </div>
                  )}
                </For>
              </div>
            </div>
          </div>
        </Show>

        {/* TICKETS TAB */}
        <Show when={activeTab() === 'tickets'}>
          <div class="bg-terminal-900 border border-terminal-750 h-full flex flex-col">
            <div class="p-4 border-b border-terminal-750 flex items-center justify-between">
              <h3 class="text-sm font-bold text-white">Support Tickets</h3>
              <button
                onClick={() => setShowTicketModal(true)}
                class="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white text-sm font-semibold rounded transition-colors"
              >
                <Plus size={16} />
                <span>New Ticket</span>
              </button>
            </div>

            <div class="flex-1 overflow-auto">
              <Show when={tickets()?.length === 0}>
                <div class="p-8 text-center">
                  <MessageSquare size={48} class="text-gray-600 mx-auto mb-4" />
                  <div class="text-gray-500 mb-2">No support tickets</div>
                  <div class="text-xs text-gray-600 mb-4">
                    Create a ticket if you need help from our support team
                  </div>
                  <button
                    onClick={() => navigate('/support/tickets/new')}
                    class="px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white text-sm font-semibold rounded transition-colors"
                  >
                    Create Ticket
                  </button>
                </div>
              </Show>

              <div class="divide-y divide-terminal-750">
                <For each={tickets() || []}>
                  {(ticket) => (
                    <button
                      onClick={() => navigate(`/support/tickets/${ticket.id}`)}
                      class="w-full p-4 text-left hover:bg-terminal-850 transition-colors group border-b border-terminal-750 last:border-0"
                    >
                      <div class="flex items-start gap-3">
                        <div class="flex-1">
                          <div class="flex items-center gap-2 mb-1">
                            <h4 class="text-sm font-semibold text-white group-hover:text-accent-500 transition-colors">
                              {ticket.subject}
                            </h4>
                            <span class={`px-2 py-0.5 rounded text-xs font-semibold ${
                              ticket.status === 'open' ? 'bg-success-500/10 text-success-500' :
                              ticket.status === 'pending' ? 'bg-warning-500/10 text-warning-500' :
                              ticket.status === 'resolved' ? 'bg-primary-500/10 text-primary-500' :
                              'bg-gray-800 text-gray-400'
                            }`}>
                              {ticket.status}
                            </span>
                          </div>
                          <div class="flex items-center gap-3 text-xs text-gray-400">
                            <span class="capitalize">{ticket.category}</span>
                            <span>•</span>
                            <span>{new Date(ticket.created_at).toLocaleDateString()}</span>
                            <span>•</span>
                            <span>{ticket.messages_count} messages</span>
                          </div>
                        </div>
                        <ChevronRight size={16} class="text-gray-600 group-hover:text-accent-500 transition-colors flex-shrink-0 mt-1" />
                      </div>
                    </button>
                  )}
                </For>
              </div>
            </div>
          </div>
        </Show>

        {/* CONTACT TAB */}
        <Show when={activeTab() === 'contact'}>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-2 sm:gap-3 h-full">
            <div class="bg-terminal-900 border border-terminal-750 p-6">
              <h3 class="text-lg font-bold text-white mb-4">Contact Methods</h3>
              <div class="space-y-4">
                <div class="p-4 bg-terminal-850 border border-terminal-750 rounded">
                  <div class="flex items-center gap-3 mb-2">
                    <div class="w-10 h-10 bg-primary-500/10 rounded flex items-center justify-center">
                      <Mail size={20} class="text-primary-500" />
                    </div>
                    <div>
                      <div class="text-sm font-semibold text-white">Email Support</div>
                      <div class="text-xs text-gray-400">Response within 24 hours</div>
                    </div>
                  </div>
                  <a
                    href="mailto:support@ciftmarkets.com"
                    class="text-sm text-accent-500 hover:underline"
                  >
                    support@ciftmarkets.com
                  </a>
                </div>

                <div class="p-4 bg-terminal-850 border border-terminal-750 rounded">
                  <div class="flex items-center gap-3 mb-2">
                    <div class="w-10 h-10 bg-success-500/10 rounded flex items-center justify-center">
                      <Phone size={20} class="text-success-500" />
                    </div>
                    <div>
                      <div class="text-sm font-semibold text-white">Phone Support</div>
                      <div class="text-xs text-gray-400">24/7 Available</div>
                    </div>
                  </div>
                  <div class="text-sm text-accent-500">+1 (646) 978-2187</div>
                </div>

                <div class="p-4 bg-terminal-850 border border-terminal-750 rounded">
                  <div class="flex items-center gap-3 mb-2">
                    <div class="w-10 h-10 bg-warning-500/10 rounded flex items-center justify-center">
                      <MessageCircle size={20} class="text-warning-500" />
                    </div>
                    <div>
                      <div class="text-sm font-semibold text-white">Live Chat</div>
                      <div class="text-xs text-gray-400">Instant assistance</div>
                    </div>
                  </div>
                  <button class="px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white text-sm font-semibold rounded transition-colors">
                    Start Chat
                  </button>
                </div>
              </div>
            </div>

            <div class="bg-terminal-900 border border-terminal-750 p-6">
              <h3 class="text-lg font-bold text-white mb-4">Business Hours</h3>
              <div class="space-y-3 text-sm">
                <div class="flex justify-between p-3 bg-terminal-850 rounded">
                  <span class="text-gray-400">Monday - Friday</span>
                  <span class="text-white font-semibold">9:00 AM - 6:00 PM ET</span>
                </div>
                <div class="flex justify-between p-3 bg-terminal-850 rounded">
                  <span class="text-gray-400">Saturday</span>
                  <span class="text-white font-semibold">10:00 AM - 4:00 PM ET</span>
                </div>
                <div class="flex justify-between p-3 bg-terminal-850 rounded">
                  <span class="text-gray-400">Sunday</span>
                  <span class="text-white font-semibold">Closed</span>
                </div>
              </div>

              <div class="mt-6 p-4 bg-primary-500/5 border border-primary-500/20 rounded">
                <div class="text-sm font-semibold text-primary-500 mb-2">Emergency Trading Issues</div>
                <div class="text-xs text-gray-400 mb-3">
                  For urgent trading-related issues during market hours, call our trading desk directly
                </div>
                <div class="text-sm text-accent-500 font-semibold">+1 (646) 978-2187</div>
              </div>
            </div>
          </div>
        </Show>
      </div>
      
      {/* CREATE TICKET MODAL */}
      <Show when={showTicketModal()}>
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-auto">
            <div class="p-6 border-b border-terminal-750 flex items-center justify-between">
              <h2 class="text-lg font-bold text-white">Create Support Ticket</h2>
              <button
                onClick={() => setShowTicketModal(false)}
                class="p-2 hover:bg-terminal-800 rounded transition-colors"
              >
                <XCircle size={20} class="text-gray-400" />
              </button>
            </div>
            
            <div class="p-6 space-y-4">
              <div>
                <label class="block text-sm font-semibold text-white mb-2">
                  Subject <span class="text-danger-500">*</span>
                </label>
                <input
                  type="text"
                  value={ticketSubject()}
                  onInput={(e) => setTicketSubject(e.target.value)}
                  placeholder="Brief description of your issue"
                  class="w-full bg-terminal-850 border border-terminal-750 text-white px-4 py-3 rounded-lg focus:outline-none focus:border-accent-500"
                />
                <div class="text-xs text-gray-500 mt-1">Minimum 5 characters</div>
              </div>
              
              <div>
                <label class="block text-sm font-semibold text-white mb-2">
                  Message <span class="text-danger-500">*</span>
                </label>
                <textarea
                  value={ticketMessage()}
                  onInput={(e) => setTicketMessage(e.target.value)}
                  placeholder="Describe your issue in detail..."
                  rows={8}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white px-4 py-3 rounded-lg focus:outline-none focus:border-accent-500 resize-none"
                />
                <div class="text-xs text-gray-500 mt-1">Minimum 10 characters</div>
              </div>
            </div>
            
            <div class="p-6 border-t border-terminal-750 flex items-center justify-end gap-3">
              <button
                onClick={() => setShowTicketModal(false)}
                class="px-4 py-2 bg-terminal-800 hover:bg-terminal-750 text-white text-sm font-semibold rounded transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateTicket}
                disabled={loading()}
                class="px-4 py-2 bg-accent-500 hover:bg-accent-600 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold rounded transition-colors"
              >
                {loading() ? 'Creating...' : 'Create Ticket'}
              </button>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
