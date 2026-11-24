"""What the numbers say, per table. Written by hand from eda/out/digest.txt; update it when the data changes."""

OVERVIEW = {
    'facts': [
        ('The target is well behaved',
         'The daily renewable share averages 44.9% (std dev 13.8 points), from 11.9% on 2 Dec 2023 to 72.4% on 26 Oct 2025. '
         'It is nearly symmetric (skew −0.07), stationary (ADF p < 0.001) and forgets quickly: correlation with the day '
         'before is 0.57, with a week before 0.10.'),
        ('Wind decides it',
         'The share moves with wind output (r = 0.88) and wind speed at 100 m (r = 0.86). Demand (−0.18), temperature '
         '(−0.01) and solar (−0.10) barely register. Solar is absent by construction: FUELHH has no solar fuel type, so '
         'the target counts wind, hydro and biomass only.'),
        ('Day-ahead forecasts carry most of the signal',
         'The Elexon day-ahead wind and solar forecast alone correlates 0.89 with the target. One linear term (forecast '
         'wind + solar over forecast demand) reaches R² 0.75 and a mean error of 5.4 points, against 11.7 for always '
         'guessing the mean.'),
        ('Persistence is a trap',
         'Predicting the share from two days earlier does worse than predicting the mean (R² −0.41). The lag features '
         'are weak (r = 0.30 and 0.10); the model has to lean on forecasts.'),
        ('Demand is the most predictable series',
         'It follows temperature (r = −0.76), the week (weekends −12%, bank holidays −14%) and the clock (autocorrelation '
         '0.90 at 24 hours). The national demand forecast is within 2.5% a day ahead.'),
        ('Prices are heavy-tailed',
         'The system price averages £86/MWh but has skew 15 and reached £2,900 for eight straight half-hours on '
         '8 Jan 2025. It is negative in 5.9% of half-hours. Use medians or a robust transform.'),
        ('Gas sets carbon intensity',
         'Carbon intensity tracks the gas share of generation almost exactly (r = 0.96) and mirrors the target (r = −0.88).'),
        ('Solar is growing fast',
         'The PV_Live peak rose from 11.5 GW (2024) to 16.4 GW (April 2026) and embedded solar capacity from 16.6 to '
         '24.0 GW. Midday demand minimums now come on sunny May afternoons, not at night.'),
    ],
    'issues': [
        ('bad', 'wind_solar_actual', 'AGWS wind under-reports from April to November 2024',
         'In healthy months AGWS wind is about 1.2 times FUELHH wind. For eight months it falls to 0.41–0.89, mostly '
         'offshore. Every row passes every silver rule. Flag or exclude those months wherever AGWS is used as truth.'),
        ('warn', 'remit_outages', 'A third of REMIT capacity "on outage" is closed power stations',
         '33 active notices run longer than a year and hold 5.0 GW of the 13.7 GW out on an average day: coal units at '
         'Fiddlers Ferry, Ferrybridge and Ratcliffe with end dates in 2050. They make the outage features nearly '
         'constant; capacity on outage has r = 0.02 with the target.'),
        ('warn', 'wind_forecast', 'The wind forecasts cover more wind than FUELHH measures',
         'WINDFOR sits about 1.3 GW above FUELHH wind at every lead time, even for hours already under way, so it is a '
         'difference of coverage, not of skill. The day-ahead DGWS forecast is higher still. Use them as features, '
         'not as estimates of FUELHH wind.'),
        ('warn', 'weather_day_ahead', 'Day-ahead weather starts four months late',
         'It begins on 1 March 2024 and lacks wind and radiation at three points until 6 March. That leaves 977 of '
         '1,100 target days with every key feature.'),
        ('warn', 'wind_solar_actual', 'AGWS has gaps',
         '885 settlement periods are missing across 78 days, 37 of them in the second quarter of 2026.'),
        ('warn', 'wind_solar_forecast', 'Ten target days lack a full day-ahead wind and solar forecast',
         'Four have none (21 Feb 2024, 9 Feb 2025, 9 Jan 2026, 5 Mar 2026) and six are partial.'),
        ('warn', 'neso_demand', 'NESO demand ends 21 days early',
         'It stops on 16 Oct 2026; every other source runs to 6 Nov 2026. Embedded wind and solar are missing for '
         'those days.'),
        ('warn', 'market_index', 'Zero is not a price',
         'N2EX is zero in 99.8% of rows (gold already ignores it). APX reports a zero price in 25 half-hours without '
         'trades; those should be NULL before averaging.'),
        ('info', 'remit_outages', 'More than half of REMIT notices arrive after the outage starts',
         '56% of revisions are published after the event began (median lead −0.1 hours), and 66 rows report negative '
         'unavailable capacity, which the hard rule rejects.'),
        ('good', 'pv_live', 'PV_Live timestamps mark the end of the half-hour',
         'Read that way, PV_Live matches NESO embedded solar with r = 0.99998; read as the start, 0.988. The README '
         'lists this as an assumption to check: it holds.'),
        ('good', 'fuelhh', 'FUELHH is clean',
         '1,100 of 1,102 days are complete (the last day is partial, and one period is missing on 29 Dec 2023). 17,840 '
         'keys arrived more than once and none carried a different value, so the history table holds no revisions yet.'),
    ],
}

TABLES = {
    'fuelhh': [
        'Gas and wind each supply about 30% of GB electricity, nuclear 14–16%, net imports 11–14% and biomass 7–9%. Wind '
        'and gas are the two that move; they are the strongest opposites in the table (r = −0.35 half-hourly, −0.73 '
        'between gas and the renewable share daily).',
        'The daily renewable share averages 44.9%. It is highest in December (50.1%) and September (48.9%) and lowest in '
        'July (39.2%) and May (40.8%). Weekends run 3–4 points above midweek because demand falls and wind does not.',
        'Wind output is below 2.5 GW in one half-hour in ten and above 14.4 GW in one in ten; the median is 7.5 GW and '
        'the record 18.4 GW.',
        'Sustained coal generation ends on 30 September 2024. After that only eight isolated half-hours report any coal.',
        'France imports to GB 91% of the time. The three Irish links mostly export (they import in only 7–12% of '
        'half-hours). Imports fall when it is windy (r = −0.37).',
        'Every period is published 30 minutes after it starts, and no value was ever revised.',
    ],
    'demand_outturn': [
        'Demand averages 26.1 GW and swings with the season: 32.6 GW in January against 21.9 GW in August.',
        'Weekdays average 27.0 GW, Saturdays 24.2 GW and Sundays 23.8 GW. The morning ramp is a working-day feature; '
        'the evening peak is there every day.',
        'The record peak is 47.3 GW at 17:00 on 5 January 2026. The record low is 12.6 GW at 13:00 on 24 May 2026: a '
        'sunny spring afternoon, when rooftop solar hides demand from the transmission system.',
        'Autocorrelation is 0.90 at 24 hours and 0.88 at a week. Same-time-yesterday and same-time-last-week are strong '
        'baselines.',
        'Transmission demand (ITSDO) is always above national demand (INDO), by 2.4 GW on average.',
        'Complete apart from the partial last day and one period on 29 December 2023.',
    ],
    'wind_forecast': [
        'Eight forecast runs a day, each covering up to 66 hours ahead. 17% of rows describe hours that had already '
        'begun when published; a cutoff filter removes them.',
        'The forecast is about 1.3 GW above FUELHH wind at every lead time, including hours already under way. That is a '
        'difference in what the two feeds cover, not forecast error.',
        'With that offset removed, the mean error grows from 1.24 GW within six hours to 1.72 GW at 60–72 hours, and '
        'the correlation with actual falls from 0.96 to 0.89.',
        'In the day-ahead range (18–42 hours) the forecast correlates 0.94 with actual wind.',
        'No gaps: every hour between the first and last start time has at least one forecast.',
    ],
    'demand_forecast': [
        'Two runs a day, at about 17:15 and 17:45 UTC, each covering roughly 69 settlement periods (0 to 35 hours ahead).',
        'A day ahead it is within 2.5% of outturn on average: mean error 633 MW, bias +18 MW, r = 0.990.',
        'The bias has a daily shape: about 170 MW low at 02:00 and 150 MW high late in the evening.',
        'Error is largest 12–24 hours ahead (802 MW). Those are the next day\'s daylight hours, where embedded solar '
        'makes demand hardest to call; it is a time-of-day effect, not a lead-time one.',
        'All but two target days have a full forecast published before the 18:00 cutoff.',
    ],
    'wind_solar_forecast': [
        'Three forecasts a day: day-ahead at 16:45 UTC, an intraday update at 07:00 and an intraday total at 11:00. '
        'Each day-ahead document holds 144 rows (48 periods for three technologies).',
        'Scored outside the months where the actuals fail: solar error is 12% of mean output (r = 0.988), onshore wind '
        '20% (r = 0.946), offshore wind 26% (r = 0.902).',
        'The wind forecasts run high: +1.25 GW offshore and +0.64 GW onshore against AGWS. Forecasts describe what the '
        'wind could produce; metered output is lower when farms are curtailed.',
        'The intraday forecasts are barely better than day-ahead, so the day-ahead forecast loses little by being early.',
        'Ten target days lack a complete day-ahead forecast; four have none at all.',
    ],
    'wind_solar_actual': [
        'From April to November 2024 AGWS reports far too little wind: 41–89% of FUELHH wind, where it normally reports '
        '120%. Offshore is hit hardest (692 MW mean in the third quarter of 2024 against a forecast of 5,140 MW). '
        'No row rule can catch it because every value is plausible on its own.',
        'In healthy months AGWS wind is 1.2 times FUELHH wind with r = 0.96: it sees wind farms FUELHH does not.',
        'Onshore and offshore wind move together (r = 0.74). Solar is nearly independent of both (r = −0.08 and −0.17).',
        'Solar peaks grew from 10.3 GW (2024) to 13.6 GW (2025) and 15.2 GW (2026).',
        '885 periods are missing across 78 days, 37 of them in the second quarter of 2026. 4,323 records were superseded '
        'by a later publish.',
        'A period is typically published 1.5 to 2.5 hours after it starts.',
    ],
    'system_prices': [
        'The price averages £86/MWh. By year: £71 (2024), £81 (2025), £110 (2026 to date).',
        'When the system is short the price averages £119; when it is long, £56. It is short 47% of the time. That '
        'split is why the distribution has two humps.',
        'The tails are extreme: skew 15, excess kurtosis 640. The price hit £2,900 for eight consecutive half-hours on '
        '8 January 2025 and went as low as −£98.',
        'Prices are negative in 5.9% of half-hours, most often in April and June (over 10%).',
        'Cheapest around 03:30 (£69 on average), dearest around 18:30 (£116).',
        'The price persists for a few hours (autocorrelation 0.84 at 30 minutes, 0.50 at 3 hours) and keeps a daily '
        'echo (0.28 at 24 hours).',
        'Sell and buy prices are equal in every row. The replacement price is NULL in 75% of rows; the reserve scarcity '
        'price in 5%.',
    ],
    'market_index': [
        'Only APX carries information. N2EX is zero in 99.8% of rows.',
        'The APX price averages £85/MWh and tracks the system price closely (daily r = 0.94), with a smaller spread: '
        'std dev £43 against £63.',
        'The system price minus the market price is centred on zero (mean +£0.37, median £0.00) with std dev £37: being '
        'out of balance costs nothing on average but is a wide gamble.',
        'Traded volume has grown each year, from 1,891 to 2,773 MWh per half-hour, and is highest in mid-afternoon.',
        '25 APX half-hours have a price of exactly zero because nothing traded; 1,452 have a negative price, which is real.',
    ],
    'remit_outages': [
        '152,672 notices are revisions of 48,099 messages: 3.2 revisions each on average, and up to 676.',
        'Gas units file half of all messages. Biomass files many small, short, mostly unplanned ones (Lynemouth and Drax '
        'units lead the asset table). Nuclear files few, but they are large (585 MW) and long (445 hours at the median).',
        'Most outages are short: the median lasts 4.5 hours and three quarters end within 17 hours. One in ten runs '
        'longer than four days.',
        '56% of notices are published after the event began. Only what was published before the cutoff can be a feature.',
        'About 13.7 GW is reported unavailable on an average day, of which 5.0 GW is closed coal stations whose notices '
        'never end.',
        '17,890 rows have no fuel type and 1,784 no unavailable capacity; 66 report a negative one.',
        'Notices follow the working day and week.',
    ],
    'bank_holidays': [
        '280 rows from 2019 to 2028: Northern Ireland 103, Scotland 94, England and Wales 83. The table includes future '
        'years, so tomorrow\'s calendar is always known.',
        'No holiday falls on a weekend; 43 are substitute days.',
        'National demand on an England and Wales bank holiday is 14% below working days of the same month. A weekend is '
        '12% below.',
        'The effect varies: Christmas Day is 21–27% lower, the early May holiday about 5%.',
        '82 holiday rows fall inside the data window.',
    ],
    'carbon_intensity': [
        'Intensity averages 127 gCO2/kWh, between 0 and 309. By year: 125 (2024), 129 (2025), 123 (2026 to date).',
        'It is the gas share in other units: r = 0.96 with the gas share of generation, −0.87 with the renewable share.',
        'The forecast in the table is close to a nowcast: mean error 9.7 g (7.7%), no bias, r = 0.974.',
        'Low and moderate bands cover 79% of half-hours; very high is rare (1.7%).',
        '31 half-hours are missing, starting with a run on 11–12 June 2024. The 13 missing actuals are all on the last '
        'day and will fill in on a later fetch.',
    ],
    'pv_live': [
        'Solar output averages 2.0 GW and is zero in 48% of half-hours. A June day yields 83 GWh; a December day 9.5 GWh.',
        'The record peak rose every year: 11.5 GW (2024), 13.9 GW (2025), 16.4 GW (23 April 2026).',
        'The timestamp marks the end of each half-hour. Read that way it matches NESO embedded solar with r = 0.99998.',
        'Elexon AGWS solar tracks PV_Live closely (r = 0.994) at about 91% of its level.',
        'No gaps: every half-hour between the first and last is present.',
    ],
    'neso_demand': [
        'NESO national demand and Elexon INDO are the same series: they differ by 37 MW on average (r = 0.9999). Either '
        'can stand in for the other.',
        'Embedded solar capacity grew from 16.6 GW to 24.0 GW over the window; embedded wind stayed flat near 6.5 GW.',
        'Embedded wind and solar meet 13.6% of underlying demand on an average day and up to 33% on the best day. None '
        'of it is in FUELHH or in the target.',
        'IFA imports 92% of the time; Moyle exports 81%; Greenlink sits idle 44%.',
        'The data stops on 16 October 2026, 21 days before the other sources.',
        'The forecast/actual indicator is NULL before 2026 and "A" in every row after.',
    ],
    'weather_day_ahead': [
        'The leakage-safe weather input. Against the short-range forecast it is within 4.6 km/h on wind at 100 m '
        '(r = 0.92) and 0.5 °C on temperature (r = 0.99).',
        'Wind is forecast best at Dogger Bank (r = 0.94) and worst in Southern England (r = 0.89).',
        'It starts on 1 March 2024, four months after the other sources.',
        '414 rows lack wind at 100 m or radiation: three locations, 1–6 March 2024.',
        'Its daily mean wind speed correlates 0.84 with the target, nearly as much as the measured wind does.',
    ],
    'weather_historical': [
        'The three offshore points average 34–36 km/h at 100 m; the Scottish Highlands 28 and Southern England 25.',
        'Wind at the five points is only loosely related (r = 0.23 to 0.69), so the set captures real geographic spread. '
        'The Highlands and East Anglia are the least alike.',
        'Wind at 100 m is 1.2 times wind at 10 m over the sea, and 1.7–1.9 times over land.',
        'Temperature ranges from −13.7 °C (Highlands) to 36.0 °C (Southern England). Sea points stay between 0 and 23 °C.',
        'Complete: every hour is present at every location, with no missing values.',
    ],
    'across': [
        'The correlation matrix splits into two blocks. One is weather-driven supply: wind speed, wind output, renewable '
        'share, and their opposites gas output and carbon intensity. The other is the season: temperature, radiation, '
        'solar and demand.',
        'Daily wind output rises roughly linearly with mean wind speed across the range that occurs, then flattens.',
        'Demand rises as temperature falls (r = −0.76) and flattens above about 15 °C, with only a slight rise on the '
        'hottest days: GB has little cooling load.',
        'Half-hourly price is only loosely tied to residual demand (r = 0.40). Prices need more than a supply and demand '
        'picture.',
        'Demand and price both fail the stationarity test on daily data (they trend with the season and the gas market); '
        'the target, wind and carbon intensity pass.',
    ],
}


# What the inferential statistics say, written by hand from eda/out/inference_digest.txt.
INFERENCE_OVERVIEW = [
    ('1,100 days are worth about 300',
     'Autocorrelation shrinks the evidence. The mean renewable share is 44.9% with a robust 95% interval of 43.4 to 46.4, twice as '
     'wide as the naive one: the information of about 320 independent days. For demand it is about 125.'),
    ('The model holds up out of sample',
     'Fitted before 2026 and scored on 307 days of 2026, eight day-ahead features give R² 0.81 and a mean error of 4.4 points '
     '(95% interval 3.9 to 4.9), against 11.4 for the training mean. In sample the R² is 0.85.'),
    ('Three features do the work, and they overlap',
     'Forecast wind and solar, forecast demand and day-ahead wind speed carry the regression, with variance inflation factors of 6 '
     'to 8. Their individual coefficients are unstable; their joint effect is not. The 7-day lag and the weekend flag add nothing '
     'once the forecasts are in (p = 0.36 and 0.25).'),
    ('Every wind forecast is biased high and none is efficient',
     'WINDFOR is 1.2 to 1.3 GW above FUELHH wind at every lead time, and the day-ahead offshore forecast 1.3 GW above AGWS. '
     'Mincer–Zarnowitz slopes are near 0.8: a linear recalibration would improve each one. The solar and demand forecasts are unbiased.'),
    ('The AGWS break is a break',
     'The AGWS to FUELHH wind ratio drops by 0.60 (95% interval 0.51 to 0.69) from April to November 2024, and settles 0.07 higher '
     'afterwards than before: the feed covers more wind since December 2024.'),
    ('No trend in the target that survives scrutiny',
     'The share drifts up 1.8 points a year (p = 0.023), but 2025 against 2024 on the same calendar days shows no change '
     '(+1.2 points, interval −2.5 to 4.8). Demand, carbon intensity and wind speed show no trend. Solar grows 20% a year '
     '(14 to 26%) and prices rose £18/MWh a year.'),
    ('A bank holiday is worth more than a Sunday',
     'Given the weather and the season, a bank holiday lowers demand by 4.7 GW (18%), a Sunday by 3.3 GW. Christmas, Boxing Day and '
     'New Year take off 7 to 8 GW; the early May holiday 1.9 GW.'),
    ('The two prices are tied together',
     'The system and market prices are cointegrated, and their mean difference is not distinguishable from zero (£0.35, '
     'interval −0.50 to 1.19). The market price predicts the next day\'s system price much more than the reverse.'),
]

INFERENCE = {
    'fuelhh': [
        'The mean share is 44.9%, with a robust 95% interval of 43.4 to 46.4. Day-to-day dependence makes 1,100 days worth about 320.',
        'The two stationarity tests nearly agree: a unit root is firmly rejected, and level-stationarity sits on the 5% line '
        '(p = 0.050) because of a slow upward drift of 1.8 points a year (0.3 to 3.4). 2025 against 2024, day for day, shows no change.',
        'Weekends are 3.6 points higher than weekdays (1.8 to 5.3), a small effect. Months differ but explain only 5% of the variance: '
        'the target is weather, not calendar.',
        'The share is not normal (flatter than a bell curve, excess kurtosis −0.9), and its variance clusters: calm spells and '
        'volatile spells come in runs.',
        'Wind and gas (r = −0.39, interval −0.48 to −0.30) and wind and imports (−0.42) are reliably opposed.',
    ],
    'demand_outturn': [
        'Demand is so persistent (lag-1 autocorrelation 0.90) that 1,101 days carry the information of about 125. The mean is '
        '26.1 GW, 25.4 to 26.9.',
        'A regression on heating degrees, day type, bank holidays and season explains 88% of daily demand. Each degree below 15.5 °C '
        'adds 592 MW (473 to 712); a Saturday removes 3.0 GW, a Sunday 3.3 GW, a bank holiday 4.7 GW.',
        'There is no trend: −35 MW a year, interval −332 to 262. Month of the year explains 73% of the variance.',
        'A unit root is not rejected on the raw series, as expected of something this seasonal; difference it or model the season '
        'before using it in a time-series model.',
    ],
    'wind_forecast': [
        'The bias is the same at every lead time: 1.2 to 1.3 GW above FUELHH wind, with intervals that all overlap. It is a level '
        'difference between the feeds, not a forecasting fault.',
        'The forecast is not efficient: actual wind rises 0.80 MW for each forecast MW (0.79 to 0.82). Rescaling it would help.',
        'Accuracy degrades measurably with lead time: 152 MW of mean error between 6 and 30 hours ahead, 222 MW more by 54 hours.',
        'It beats the obvious alternative by a wide margin: 2.7 GW less error than using the wind of 48 hours earlier.',
    ],
    'demand_forecast': [
        'No bias: the mean error is 17 MW, interval −20 to 55.',
        'It is far better than persistence: 2.0 GW less error than demand two days earlier and 1.5 GW less than a week earlier.',
        'It is slightly inefficient (slope 0.991, interval 0.987 to 0.995) and its errors are autocorrelated (lag-1 r = 0.21): '
        'yesterday\'s miss carries some information about today\'s.',
        'The bias is distinguishable from zero in 12 of 24 hours and differs by day of the week, so a small hour-and-weekday '
        'correction is available.',
    ],
    'wind_solar_forecast': [
        'Solar is unbiased a day ahead (27 MW, interval −3 to 58). Offshore wind is 1.26 GW high (1.11 to 1.41) and onshore 0.64 GW '
        '(0.56 to 0.72).',
        'The wind forecasts are inefficient, with slopes of 0.79 and 0.81. The solar forecast is efficient within noise '
        '(slope 0.993; its test does not survive the multiple-testing correction).',
        'The intraday forecasts are better than day-ahead, and the difference is real but small: 45 MW for solar, 111 MW offshore, '
        '126 MW onshore. Intraday wins on 58 to 67% of days.',
    ],
    'wind_solar_actual': [
        'The gap is a structural break: the ratio to FUELHH wind falls by 0.60 (0.51 to 0.69) and the two regimes barely overlap '
        '(Kolmogorov–Smirnov D = 0.88).',
        'After the gap the ratio is 0.07 higher than before it (0.03 to 0.10): AGWS covers more wind from December 2024. Any series '
        'that spans both periods needs a level adjustment.',
        'Offshore and onshore wind are correlated at 0.76 (0.73 to 0.80). Solar and wind are opposed day to day (r = −0.42), more '
        'than the half-hourly figure suggested: the season pulls them apart.',
        'Solar grew 37% from 2024 to 2025 on the same calendar days (530 MW, interval 374 to 686).',
    ],
    'system_prices': [
        'The daily price is far from normal (skew 4.3, excess kurtosis 60) and logs do not fix it. Prefer medians, quantiles and '
        'rank methods.',
        'It is not stationary: both tests say so, and the trend is £18/MWh a year (13 to 23). Model changes or a rolling level, not '
        'the raw price.',
        'On the same day, short half-hours cost £60/MWh more than long ones (57 to 63). The system is short 47.0% of the time '
        '(45.8 to 48.3), reliably less than half.',
        'Weekends are £14 cheaper (10 to 18), and windy days are cheaper (r = −0.31, interval −0.43 to −0.22).',
        'The volatility-clustering test finds nothing, but one day (8 January 2025) dominates the variance and blinds it.',
    ],
    'market_index': [
        'The system and market prices are cointegrated: they share one long-run path.',
        'Their mean difference is £0.35 and not distinguishable from zero (−0.50 to 1.19). Being out of balance costs nothing on '
        'average; the cost is the risk.',
        'The market price predicts the next day\'s system price strongly (F = 50); the reverse is much weaker (F = 7).',
        'Like the system price, it trends up by £18 a year and fails both stationarity tests.',
    ],
    'remit_outages': [
        'Whether an outage is planned depends strongly on fuel type (Cramér V = 0.40): 11% of biomass messages are planned, 49% of '
        'gas, 70% of pumped storage.',
        'Planned outages last longer: a median of 8.0 hours against 2.4.',
        'Notices arrive in bursts. The day-to-day variance is 29 times what random arrivals would give, and weekdays see 156 a day '
        'against 96 at weekends.',
        'Weekend notices are more often unplanned (63% against 54%), but the association is weak (V = 0.07). With this many rows '
        'almost everything is significant; the effect sizes are what to read.',
        '56% of notices, or 47% counted per message, are published after the event began.',
    ],
    'bank_holidays': [
        'Given the weather and the season, a bank holiday lowers national demand by 4.7 GW (3.5 to 6.0), or 18%. That is more than '
        'a Sunday.',
        'The holidays differ widely: Christmas Day, Boxing Day and New Year remove 7.4 to 8.1 GW; Good Friday 2.6 GW; the early May '
        'holiday 1.9 GW. One holiday flag is too coarse. Each estimate rests on three days.',
        'Mondays hold 57% of holidays. The three nations place theirs on the same weekdays but have different numbers a year '
        '(8.3, 9.4 and 10.3).',
    ],
    'carbon_intensity': [
        'Each point of gas share adds 3.40 gCO2/kWh (3.28 to 3.53) and explains 95% of daily carbon intensity.',
        'Stationary by both tests, with no trend (−0.9 g a year, interval −7.2 to 5.4) and no change from 2024 to 2025.',
        'Weekends are 22 g lower (16 to 27).',
        'The forecast is unbiased (−0.30 g, interval −0.78 to 0.19) but slightly inefficient (slope 0.963).',
        'The mix of intensity bands "changed" between years with p < 0.001 and Cramér V = 0.09: a large sample finding a trivial '
        'difference.',
    ],
    'pv_live': [
        'Solar output grows 19.7% a year once the season is removed (14.1 to 25.5%).',
        '2025 beat 2024 on 68% of calendar days, by 12.3 GWh a day on average (8.2 to 16.4).',
        'No weekend effect (0.3 GWh, p = 0.79): a useful sanity check, since the sun keeps no calendar.',
        'Month of the year explains 68% of the variance, and the mean is known only to within 43 to 54 GWh a day.',
        'Daily energy tracks radiation at the five weather points with r = 0.93 (0.91 to 0.95).',
    ],
    'neso_demand': [
        'NESO national demand is 37 MW above Elexon INDO (27 to 47): statistically different, practically identical. An equivalence '
        'test puts them within ±100 MW, 0.4% of demand.',
        'NESO embedded solar and PV_Live do not differ (1.6 MW, interval −1.1 to 4.3).',
        'Embedded solar capacity grows 2.85 GW a year (2.79 to 2.91). Embedded wind capacity is shrinking slightly, by 56 MW a year.',
        'Embedded wind follows FUELHH wind with r = 0.92, so the target\'s blindness to embedded wind loses little signal.',
    ],
    'weather_day_ahead': [
        'The day-ahead forecast is slightly biased in every variable: wind +0.6 km/h, radiation +7.8 W/m², cloud cover −5.4 points. '
        'Small against the errors, but consistent.',
        'It predicts the renewable share almost as well as the wind that happened: r = 0.841 against 0.853, with overlapping '
        'intervals. Little is lost by using the leakage-safe input.',
        'It beats persistence by 8.1 km/h of mean error.',
        'Accuracy differs by location: Southern England and Dogger Bank are easiest, the Irish Sea hardest.',
    ],
    'weather_historical': [
        'The five locations differ in mean wind, but location explains only 11% of the variance. The three sea points are '
        'nearly interchangeable on average (Irish Sea and Dogger Bank differ by 0.5 km/h, not distinguishable).',
        'Hourly wind fits a Weibull distribution at the three sea points, with shape 2.2 to 2.3. It is rejected on land, where the '
        'Highlands (shape 1.9) are gustier and Southern England (2.4) steadier.',
        'Stationary by both tests, with no trend and no change from 2024 to 2025: the wind resource was stable over the window.',
        'No weekend effect, as there should not be.',
    ],
    'across': [
        'Eight day-ahead features explain 85% of the target in sample and 81% out of sample, on 2026 data the fit never saw.',
        'Forecast wind and solar (+0.52 standard deviations), forecast demand (−0.43) and day-ahead wind speed (+0.32) carry it. '
        'Their variance inflation factors are 6 to 8: they share information, so judge them together.',
        'The lags and the weekend flag add little: the 2-day lag is just significant (p = 0.019), the 7-day lag and the weekend are not.',
        'The eight-feature model beats the two-forecast model by 0.64 points of mean error (0.21 to 1.07).',
        'Residuals are autocorrelated (lag-1 r = 0.41) and heteroscedastic. Yesterday\'s error is usable information the features '
        'do not hold; a model with an error-correction term or a previous-day residual should gain.',
        'Naive intervals on correlations are too narrow by a third to a half; the block-bootstrap ones are the honest ones.',
    ],
}


# What the feature engineering shows, written by hand from eda/out/features_digest.txt.
FEATURES = [
    '131 features, one row per target day, from all 15 tables and the calendar. Fitted on the days before 2026 and scored on 309 days of '
    '2026, a gradient-boosted model on all of them has a mean error of 3.55 points of renewable share, against 4.39 for the eight '
    'features gold has today: 19% lower, and the difference is statistically clear (Diebold–Mariano p < 0.001).',
    'The gain comes from the features, not the model. Gradient boosting on gold\'s eight features is no better than the linear model '
    '(4.41 against 4.39). A ridge regression on the new features already reaches 3.75.',
    'Two engineered ratios carry most of it. Forecast wind as a share of forecast demand is the single strongest feature (r = 0.90 with '
    'the target; shuffling it adds 3.1 points of error). The physical estimate of the target, built from the calibrated wind forecast, '
    'yesterday\'s biomass and hydro, forecast demand and yesterday\'s imports, is second, and on its own with no fitting beyond a '
    'straight line scores 5.00.',
    'Solar belongs in demand, not in the numerator. Wind over demand beats wind + solar over demand, because the target has no solar in it.',
    'Two tables do the work: the day-ahead wind and solar forecast (shuffling its features adds 7.7 points of error) and the day-ahead '
    'weather (1.6). The demand forecast and REMIT outages add a little. Dropping any one of the other eleven tables and refitting '
    'does not hurt, and often helps slightly: their information is already in the forecasts.',
    'Yesterday\'s actuals add almost nothing once the forecasts are in. The share so far yesterday has r = 0.50 with the target but no '
    'importance in the model: tomorrow\'s forecast already knows what yesterday\'s weather was heading towards.',
    'A turbine power curve applied to hourly wind speed beats mean wind speed (r = 0.87 against 0.84), and per-location power matters: '
    'Southern England and East Anglia add information the national mean hides.',
    'Thirty features are enough. The selected set scores 3.56, the same as all 131. Many of the rest are near-duplicates: WINDFOR and '
    'the day-ahead forecast agree at r = 0.998, so one of the two wind forecasts is redundant.',
    'The leakage audit rebuilt 12 days from tables with every row published after the cutoff deleted, and no feature changed. It earned '
    'its place: the first run caught NESO embedded generation being used a day before NESO publishes it, and an AGWS ratio that used '
    'republished values. Both are fixed.',
    'Three limits. The availability times for PV_Live, carbon intensity, the market index, system prices, NESO and the weather are '
    'assumptions, because those tables carry no publish time. System price features use the latest settlement run, which can differ '
    'from the first price. And one year of test data is one draw of the weather: the interval on the best model\'s error is 3.3 to 4.0.',
]


# What the modelling shows, written by hand from the output of eda/model.py.
MODELS = [
    'The baseline is what the pipeline could ship today: a linear regression on the eight features in gold. Predicting a day ahead '
    'over 674 days from January 2025, it has a mean error of 4.47 points of renewable share (R² 0.825). That is already far better '
    'than guessing the average (11.7) or repeating the share of two days earlier (12.8).',
    'The best model is an equal blend of a ridge regression and a gradient-boosted model, both on the 131 engineered features. Its '
    'mean error is 3.31 points (R² 0.907): 26% lower than the baseline. The gain is 1.16 points with a 95% interval of 0.89 to 1.43, '
    'and it wins in 22 of 23 months.',
    'Blending matters. Ridge and gradient boosting each score 3.56; their average scores 3.31. They make different mistakes: one is a '
    'straight-line model of the forecasts, the other catches thresholds and interactions.',
    'The largest gains are at the extremes, where the baseline is weakest. On days under 25% the error falls from 6.1 to 2.6 points, '
    'and on days of 55–65% from 5.8 to 3.5. In the middle (35–45%) the two are level.',
    'Very windy days are still under-predicted: above 65% the best model is 3.6 points low on average. Wind farms run flat out there '
    'and the forecast ratios stop distinguishing a good day from a record one.',
    'The baseline leaves more behind in its errors (autocorrelation 0.45 at one day) than the best model does (0.34). Some signal '
    'remains: yesterday\'s miss still says something about today\'s.',
    'The 80% prediction interval needed calibrating. Straight from the quantile models it held only 66% of days. Widened by how far '
    'earlier out-of-sample days had fallen outside it, it holds 80.0% and is 12.5 points wide on average.',
    'Cross-validation chose a simple gradient-boosted model: 200 shallow trees of depth 2. With about 400 training days at the start, '
    'anything more flexible overfits.',
    'The worst miss, 23 November 2025, was 16.7 points low for the best model and 20.7 for the baseline: the forecasts themselves '
    'were wrong that day, and no model built on them can recover that.',
    'Limits: the features rest on assumed publish times for six tables, the test covers under two years of weather, and the models '
    'are saved locally in eda/out/models, not deployed.',
]
