# Demo data

`hotel_bookings.csv.gz`: 119,390 bookings at a resort hotel and a city hotel in Portugal (2015–2017),
32 columns, target `is_canceled` (37% cancelled).

Source: Antonio, Almeida & Nunes (2019), "Hotel booking demand datasets", *Data in Brief* 22, 41–49,
via the TidyTuesday project (2020-02-11).

Note: `reservation_status` and `reservation_status_date` are recorded *after* the outcome
(they literally say "Canceled"). Glassbox's leakage detector flags them, which makes a good demo moment.
