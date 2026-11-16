# Competitor tracker

A cheap daily snapshot of what is gaining views among code-made simulation, quiz,
chess and map Shorts channels, so daily research needs no manual browsing.
Python standard library only. About 85 requests and 75 seconds per run.

## Commands

Run from the repository root (or anywhere; paths are relative to this folder).

```bash
python3 shorts-factory/research/track.py                    # snapshot for today (UTC) and summary
python3 shorts-factory/research/track.py --date 2026-11-17  # use another date label
python3 shorts-factory/research/track.py --add HANDLE       # append a channel to the watchlist, then exit
python3 shorts-factory/research/track.py --add HANDLE --category quiz --counter-example
python3 shorts-factory/research/track.py --summary-only     # rebuild the summary from saved data, no network
python3 shorts-factory/research/track.py --source page      # skip RSS, use the uploads page (rss = RSS only)
```

Run it once a day at about the same hour. A second run on the same date replaces
that date's snapshot (a channel that fails on the rerun keeps its earlier data).

## Files

| File | Content |
|---|---|
| `track.py` | the tracker |
| `watchlist.json` | 41 channels: 33 active formats, 8 failed counter-examples (`counter_example: true`); channel ids are cached here |
| `data/YYYY-MM-DD.json` | one snapshot per day: channel stats and latest uploads, one channel per line |
| `latest_summary.md` | at most 70 lines: top 20 videos, channels ranked by gain, counter-examples, common title words, problems |

Watchlist categories: sim, merge, marble-flag, loop3d, math, counting, puzzle, chess, quiz, map.

## What is measured

- Channel stats come from the About page: total views (exact), video count (exact),
  subscribers (rounded by YouTube to 3 digits).
- Channel gain = change in total views since the most recent earlier snapshot.
- Video lists come from the RSS feed (15 latest uploads, exact views and publish times).
  If the feed fails 3 times in a row, the run switches to the public uploads playlist page
  (30 latest uploads kept).
- Video gain = change in views of a listed video since the earlier snapshot. A video listed
  above every video known yesterday counts as new, and all its views count as gain.
- With no earlier video list, videos are ranked by views per day since publication.

## Limits and heuristics (read before trusting a number)

- `~` marks a value YouTube rounded. The uploads page shows "134K" or "1.2M"; the true value
  is assumed to lie between the shown value and the next step, and the middle is used. A large
  video can therefore show a gain of 0 one day and a full step the next.
- `≈` marks an age guessed from the upload order and the channel's lifetime upload rate.
  The uploads page shows no dates for Shorts-only channels. This is a rough guess.
  From the second snapshot on, new videos are dated by when they first appeared (within a day).
- Views per day uses a minimum age of 6 hours (exact times) or 1 day (estimated times).
- "uploads 7d" is shown only when every listed video has a real date, otherwise `?`.
  "N+" means the whole list is newer than 7 days.
- YouTube updates the About totals in steps, not continuously (observed on 2026-11-16: one
  channel's total changed twice by about 586K within hours). One day's channel gain is noisy;
  trends over 2 or more days are more reliable.
- At most 5 rows per channel appear in the top-20 table; the summary says how many
  higher-ranked rows were left out. The full data is in the snapshot file.
- The top-20 table only covers listed uploads (15 or 30 latest), not a channel's older videos.
- Title words are counted over the top 40 ranked videos, at most 6 per channel.

## Status on 2026-11-16

- The RSS feed returned HTTP 404 for every channel tested, so the baseline used the uploads
  page (rounded counts). The RSS code path is tested only against a sample feed, not live.
- `data/2026-11-15.json` was not produced by a run. It was seeded from the earlier research
  result (channel totals only, recorded on 2026-11-15 at about 08:30 UTC, exact time per
  channel unknown), so the first summary can show channel gains over roughly 19 hours.
- Pacing is 0.5 s between requests, 15 s timeout, one retry on transient errors, and a
  215 s overall budget; a channel that fails is recorded with its error and the run continues.
