"""
Test script for Alpaca real-time streaming.

Connects to Alpaca WebSocket and subscribes to quotes/trades
for test symbols to verify the connection works.
"""

import asyncio
import os
import sys

# Windows requires SelectorEventLoop for aiodns
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cift.data.alpaca_stream import (
    AlpacaStreamClient,
    AlpacaFeed,
    AlpacaQuote,
    AlpacaTrade,
    AlpacaBar,
)


async def test_alpaca_stream():
    """Test Alpaca streaming connection."""
    print("=" * 60)
    print("ALPACA REAL-TIME STREAM TEST")
    print("=" * 60)
    
    # Create client (IEX feed is free, SIP requires paid subscription)
    client = AlpacaStreamClient(feed=AlpacaFeed.IEX)
    
    # Check if configured
    if not client.is_configured:
        print("ERROR: Alpaca API keys not configured!")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return False
    
    print(f"API Key: {client.api_key[:8]}...")
    print(f"Feed: {client.feed.value}")
    
    # Counters
    quote_count = 0
    trade_count = 0
    
    def on_quote(quote: AlpacaQuote):
        nonlocal quote_count
        quote_count += 1
        print(f"[QUOTE] {quote.symbol}: "
              f"Bid ${quote.bid_price:.2f} x {quote.bid_size} | "
              f"Ask ${quote.ask_price:.2f} x {quote.ask_size} | "
              f"Spread: {quote.spread_bps:.1f} bps | "
              f"Imbalance: {quote.imbalance:+.2f}")
    
    def on_trade(trade: AlpacaTrade):
        nonlocal trade_count
        trade_count += 1
        print(f"[TRADE] {trade.symbol}: "
              f"${trade.price:.2f} x {trade.size} = ${trade.notional:,.0f} | "
              f"Exchange: {trade.exchange}")
    
    def on_bar(bar: AlpacaBar):
        print(f"[BAR] {bar.symbol}: "
              f"O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} "
              f"V:{bar.volume:,} VWAP:{bar.vwap:.2f}")
    
    # Register callbacks
    client.on_quote(on_quote)
    client.on_trade(on_trade)
    client.on_bar(on_bar)
    
    # Connect
    print("\nConnecting to Alpaca stream...")
    if not await client.connect():
        print("ERROR: Failed to connect!")
        return False
    
    print("Connected and authenticated!")
    
    # Test symbols
    symbols = ["AAPL", "MSFT", "SPY"]
    print(f"\nSubscribing to: {symbols}")
    
    await client.subscribe_quotes(symbols)
    await client.subscribe_trades(symbols)
    
    # Run for 30 seconds
    print("\nListening for data (30 seconds)...")
    print("-" * 60)
    
    try:
        # Create a task for the run loop
        run_task = asyncio.create_task(client.run())
        
        # Wait for 30 seconds
        await asyncio.sleep(30)
        
        # Stop
        await client.disconnect()
        run_task.cancel()
        
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()
    
    print("-" * 60)
    print(f"\nResults:")
    print(f"  Quotes received: {quote_count}")
    print(f"  Trades received: {trade_count}")
    
    stats = client.get_stats()
    print(f"  Total messages: {stats['message_count']}")
    print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
    
    success = quote_count > 0 or trade_count > 0
    if success:
        print("\n✅ Alpaca streaming is working!")
    else:
        print("\n⚠️  No data received (market may be closed)")
    
    return success


if __name__ == "__main__":
    try:
        result = asyncio.run(test_alpaca_stream())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(0)
