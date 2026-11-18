-- Answers to the fuelhh SQL practice: one query per chart in the fuelhh EDA. DuckDB dialect.
-- Run from the repo root. Check values are in the comment above each query.

-- =============================================================================================
-- Setup: two views the answers reuse
-- =============================================================================================

-- The sample, one row per fuel type per half-hour
CREATE OR REPLACE VIEW fuelhh AS
SELECT * FROM read_csv_auto('eda/practice/fuelhh_sample.csv');

-- One row per half-hour, one column per category: "long to wide" with conditional sums.
-- Interconnectors (INT*) are flows, so they are kept out of domestic generation.
CREATE OR REPLACE VIEW halfhour AS
SELECT
    settlement_date,
    settlement_period,
    SUM(generation_mw) FILTER (WHERE fuel_type = 'WIND')                        AS wind,
    SUM(generation_mw) FILTER (WHERE fuel_type IN ('CCGT', 'OCGT'))             AS gas,
    SUM(generation_mw) FILTER (WHERE fuel_type = 'NUCLEAR')                     AS nuclear,
    SUM(generation_mw) FILTER (WHERE fuel_type = 'BIOMASS')                     AS biomass,
    SUM(generation_mw) FILTER (WHERE fuel_type LIKE 'INT%')                     AS net_imports,
    SUM(generation_mw) FILTER (WHERE fuel_type = 'NPSHYD')                      AS hydro,
    SUM(generation_mw) FILTER (WHERE fuel_type IN ('COAL', 'OIL'))              AS coal_oil,
    SUM(generation_mw) FILTER (WHERE fuel_type IN ('OTHER', 'PS'))              AS other_ps,
    SUM(generation_mw) FILTER (WHERE fuel_type NOT LIKE 'INT%')                 AS domestic,
    SUM(generation_mw) FILTER (WHERE fuel_type IN ('WIND', 'NPSHYD', 'BIOMASS')) AS renewable
FROM fuelhh
GROUP BY settlement_date, settlement_period;

-- Daily renewable share, complete days only. A UK day has 46, 48 or 50 half-hours: the length of the day
-- in Europe/London divided by 30 minutes. (The sample has no clock-change days, so every day is 48.)
-- The x 0.5 that turns MW into MWh is on top and bottom, so it cancels.
CREATE OR REPLACE VIEW daily_share AS
SELECT
    settlement_date,
    100 * SUM(renewable) / SUM(domestic) AS share
FROM halfhour
GROUP BY settlement_date
HAVING COUNT(*) = (epoch(timezone('Europe/London', (settlement_date + 1)::TIMESTAMP))
                 - epoch(timezone('Europe/London', settlement_date::TIMESTAMP))) / 1800;


-- =============================================================================================
-- 1. Generation mix, weekly mean (stacked area)
-- Check: 21 weeks; gas in the week of 2024-01-01 = 11.01 GW
-- =============================================================================================
WITH daily AS (                                   -- mean of each day first, then of each week
    SELECT settlement_date,
           AVG(wind) AS wind, AVG(gas) AS gas, AVG(nuclear) AS nuclear, AVG(biomass) AS biomass,
           AVG(net_imports) AS net_imports, AVG(hydro) AS hydro, AVG(coal_oil) AS coal_oil,
           AVG(other_ps) AS other_ps
    FROM halfhour
    GROUP BY settlement_date
)
SELECT
    date_trunc('week', settlement_date) AS week,  -- weeks start on Monday
    ROUND(AVG(wind) / 1000, 2)        AS wind_gw,
    ROUND(AVG(gas) / 1000, 2)         AS gas_gw,
    ROUND(AVG(nuclear) / 1000, 2)     AS nuclear_gw,
    ROUND(AVG(biomass) / 1000, 2)     AS biomass_gw,
    ROUND(AVG(net_imports) / 1000, 2) AS net_imports_gw,
    ROUND(AVG(hydro) / 1000, 2)       AS hydro_gw,
    ROUND(AVG(coal_oil) / 1000, 2)    AS coal_oil_gw,
    ROUND(AVG(other_ps) / 1000, 2)    AS other_ps_gw
FROM daily
GROUP BY week
ORDER BY week;


-- =============================================================================================
-- 2. Daily renewable share, with a 28-day rolling mean (line)
-- Check: 84 rows; 2024-01-01 = 59.84
-- =============================================================================================
SELECT
    settlement_date,
    ROUND(share, 2) AS share,
    -- RANGE, not ROWS: the window is 28 calendar days, so gaps in the sample are respected
    ROUND(AVG(share) OVER (ORDER BY settlement_date
                           RANGE BETWEEN INTERVAL 27 DAYS PRECEDING AND CURRENT ROW), 2) AS share_28d
FROM daily_share
ORDER BY settlement_date;


-- =============================================================================================
-- 3. Distribution of the daily share (histogram, 5-point bins)
-- Check: min 13.66, max 70.98
-- =============================================================================================
SELECT
    FLOOR(share / 5) * 5     AS bin_from,
    FLOOR(share / 5) * 5 + 5 AS bin_to,
    COUNT(*)                 AS days
FROM daily_share
GROUP BY bin_from
ORDER BY bin_from;

SELECT ROUND(MIN(share), 2) AS min_share, ROUND(MAX(share), 2) AS max_share FROM daily_share;


-- =============================================================================================
-- 4. Share by month of year (box plot)
-- Check: January median = 37.22
-- =============================================================================================
SELECT
    month(settlement_date)                      AS month,
    ROUND(quantile_cont(share, 0.01), 2)        AS p01,
    ROUND(quantile_cont(share, 0.25), 2)        AS q1,
    ROUND(quantile_cont(share, 0.50), 2)        AS median,
    ROUND(quantile_cont(share, 0.75), 2)        AS q3,
    ROUND(quantile_cont(share, 0.99), 2)        AS p99
FROM daily_share
GROUP BY month
ORDER BY month;


-- =============================================================================================
-- 5. Share by day of week (box plot; the mean is the check)
-- Check: Sunday mean = 44.63
-- =============================================================================================
SELECT
    isodow(settlement_date)               AS day_number,   -- 1 = Monday ... 7 = Sunday
    dayname(settlement_date)              AS day,
    ROUND(AVG(share), 2)                  AS mean_share,
    ROUND(quantile_cont(share, 0.25), 2)  AS q1,
    ROUND(quantile_cont(share, 0.50), 2)  AS median,
    ROUND(quantile_cont(share, 0.75), 2)  AS q3
FROM daily_share
GROUP BY day_number, day
ORDER BY day_number;


-- =============================================================================================
-- 6. Autocorrelation of the share (bar, lags 1-30)
-- Check: lag 1 r = 0.592 over 72 pairs
-- Join each day to the day k days earlier. Only pairs that exist count, which handles the gaps
-- between months. (The EDA used the textbook ACF on the full series, which differs slightly.)
-- =============================================================================================
SELECT
    k                                  AS lag_days,
    ROUND(corr(today.share, earlier.share), 3) AS r,
    COUNT(*)                           AS pairs
FROM range(1, 31) AS lags(k)
JOIN daily_share AS today   ON TRUE
JOIN daily_share AS earlier ON earlier.settlement_date = today.settlement_date - k::INTEGER
GROUP BY k
ORDER BY k;


-- =============================================================================================
-- 7. Mean output by fuel type (horizontal bar)
-- Check: CCGT = 8,114.2
-- =============================================================================================
SELECT
    fuel_type,
    ROUND(AVG(generation_mw), 1) AS mean_mw
FROM fuelhh
GROUP BY fuel_type
ORDER BY mean_mw DESC;


-- =============================================================================================
-- 8. Spread of half-hourly output by fuel type (box plot)
-- Check: WIND q1 4,030.8, median 6,621.5, q3 10,282.5
-- =============================================================================================
SELECT
    fuel_type,
    ROUND(quantile_cont(generation_mw, 0.01), 1) AS p01,
    ROUND(quantile_cont(generation_mw, 0.25), 1) AS q1,
    ROUND(quantile_cont(generation_mw, 0.50), 1) AS median,
    ROUND(quantile_cont(generation_mw, 0.75), 1) AS q3,
    ROUND(quantile_cont(generation_mw, 0.99), 1) AS p99
FROM fuelhh
GROUP BY fuel_type
ORDER BY fuel_type;


-- =============================================================================================
-- 9. Average day by source (line, one point per half-hour)
-- Check: gas at period 37 (18:00) = 11.495 GW
-- =============================================================================================
SELECT
    settlement_period,
    printf('%02d:%02d', (settlement_period - 1) // 2, ((settlement_period - 1) % 2) * 30) AS uk_time,
    ROUND(AVG(wind) / 1000, 3)        AS wind_gw,
    ROUND(AVG(gas) / 1000, 3)         AS gas_gw,
    ROUND(AVG(nuclear) / 1000, 3)     AS nuclear_gw,
    ROUND(AVG(biomass) / 1000, 3)     AS biomass_gw,
    ROUND(AVG(net_imports) / 1000, 3) AS net_imports_gw
FROM halfhour
WHERE settlement_period <= 48                    -- leave out the extra periods of a 50-period day
GROUP BY settlement_period
ORDER BY settlement_period;


-- =============================================================================================
-- 10. Gas (CCGT) by month and time of day (heatmap)
-- Check: January, period 35 = 16,596.7
-- =============================================================================================
SELECT
    month(settlement_date)       AS month,
    settlement_period,
    ROUND(AVG(generation_mw), 1) AS mean_mw
FROM fuelhh
WHERE fuel_type = 'CCGT' AND settlement_period <= 48
GROUP BY month, settlement_period
ORDER BY month, settlement_period;

-- The same grid shaped like the heatmap: one row per month, one column per half-hour
PIVOT (SELECT month(settlement_date) AS month, settlement_period, generation_mw
       FROM fuelhh WHERE fuel_type = 'CCGT' AND settlement_period <= 48)
ON settlement_period
USING ROUND(AVG(generation_mw), 0)
ORDER BY month;


-- =============================================================================================
-- 11. Wind by month and time of day (heatmap)
-- Check: July, period 25 = 7,318.6
-- =============================================================================================
SELECT
    month(settlement_date)       AS month,
    settlement_period,
    ROUND(AVG(generation_mw), 1) AS mean_mw
FROM fuelhh
WHERE fuel_type = 'WIND' AND settlement_period <= 48
GROUP BY month, settlement_period
ORDER BY month, settlement_period;


-- =============================================================================================
-- 12. Wind duration curve (area)
-- Check: P10 1,876.2; P50 6,621.5; P90 13,419.8
-- A duration curve sorts every half-hour from highest to lowest. "At 90% of the time, wind was at
-- least X" is the 10th percentile, so the curve's x is 100 minus the percentile.
-- =============================================================================================
SELECT
    ROUND(100.0 * ROW_NUMBER() OVER (ORDER BY generation_mw DESC) / COUNT(*) OVER (), 2)
                  AS pct_of_time_at_least,
    generation_mw AS wind_mw
FROM fuelhh
WHERE fuel_type = 'WIND'
ORDER BY pct_of_time_at_least;

SELECT
    ROUND(quantile_cont(generation_mw, 0.10), 1) AS p10,
    ROUND(quantile_cont(generation_mw, 0.50), 1) AS p50,
    ROUND(quantile_cont(generation_mw, 0.90), 1) AS p90
FROM fuelhh
WHERE fuel_type = 'WIND';


-- =============================================================================================
-- 13. Correlation between sources (heatmap)
-- Check: WIND vs CCGT = -0.483
-- A self-join puts two fuels side by side in the same half-hour; a < b keeps each pair once.
-- =============================================================================================
SELECT
    a.fuel_type                                    AS fuel_a,
    b.fuel_type                                    AS fuel_b,
    ROUND(corr(a.generation_mw, b.generation_mw), 3) AS r
FROM fuelhh AS a
JOIN fuelhh AS b
  ON  a.settlement_date   = b.settlement_date
  AND a.settlement_period = b.settlement_period
  AND a.fuel_type < b.fuel_type
GROUP BY fuel_a, fuel_b
ORDER BY fuel_a, fuel_b;

-- The category version used in the EDA, from the wide view
SELECT
    ROUND(corr(wind, gas), 3)         AS wind_gas,
    ROUND(corr(wind, net_imports), 3) AS wind_imports,
    ROUND(corr(wind, nuclear), 3)     AS wind_nuclear,
    ROUND(corr(gas, net_imports), 3)  AS gas_imports
FROM halfhour;


-- =============================================================================================
-- 14. How often each interconnector imports (horizontal bar)
-- Check: INTFR = 93.6%; INTGRNL has only 1,420 half-hours (it opened in March 2024)
-- =============================================================================================
SELECT
    fuel_type                                                              AS link,
    COUNT(generation_mw)                                                   AS half_hours,
    ROUND(100.0 * COUNT(*) FILTER (WHERE generation_mw > 0) / COUNT(generation_mw), 1) AS pct_importing,
    ROUND(100.0 * COUNT(*) FILTER (WHERE generation_mw < 0) / COUNT(generation_mw), 1) AS pct_exporting
FROM fuelhh
WHERE fuel_type LIKE 'INT%'
GROUP BY fuel_type
ORDER BY pct_importing DESC;


-- =============================================================================================
-- 15. Coal by month (bar)
-- Check: January 34.2 GWh, September 28.7, October 0.0
-- (The sample has 7 days a month, so these are 7-day totals.)
-- =============================================================================================
SELECT
    date_trunc('month', settlement_date)        AS month,
    ROUND(SUM(generation_mw) * 0.5 / 1000, 1)   AS coal_gwh,
    COUNT(*) FILTER (WHERE generation_mw > 0)   AS half_hours_running
FROM fuelhh
WHERE fuel_type = 'COAL'
GROUP BY month
ORDER BY month;


-- =============================================================================================
-- 16. Delay between a period starting and its publication (histogram)
-- Check: median 30 minutes, max 233
-- =============================================================================================
SELECT
    date_diff('minute', start_time, publish_time) AS delay_minutes,
    COUNT(*)                                      AS rows
FROM fuelhh
GROUP BY delay_minutes
ORDER BY delay_minutes;

SELECT
    quantile_cont(date_diff('minute', start_time, publish_time), 0.5) AS median_minutes,
    MAX(date_diff('minute', start_time, publish_time))                AS max_minutes
FROM fuelhh;
