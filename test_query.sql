SELECT * FROM (
    SELECT
        symbol,
        first(price) as open_price,
        last(price) as current_price,
        sum(volume) as total_volume,
        count(*) as tick_count,
        ((last(price) - first(price)) / first(price) * 100) as change_percent
    FROM ticks
    WHERE timestamp >= '2025-12-22 00:00:00'
    GROUP BY symbol
)
WHERE tick_count > 10
LIMIT 5;
