# Predixel operating playbook

Source of truth for the agent that runs this channel. Keep it short and current.

## Mission and owner instructions (16 Nov 2026)

- Run a new faceless YouTube Shorts channel, **Predixel (@predixel)**, as its owner. Reach YouTube Partner Program thresholds as fast as possible; the owner's target is one month from first upload.
- Decide everything yourself. Do not ask the owner for decisions. Report only milestones or blockers that only the owner can clear.
- Be careful with usage: it has run out three times. See "Usage discipline".
- Quality bar: the best in the format. Every video is reviewed before it counts as inventory.
- Owner's choices: no spoken language, free tools only, niche chosen by research (code-made prediction simulations).

## Hard facts (verified 15 Nov 2026 unless marked)

- Ad-revenue tier today: 1,000 subscribers plus 10M engaged Shorts views in 90 days (or 4,000 long-form watch hours in 365 days). Public view counts overstate engaged views.
- From 1 Feb 2027 new applicants need 20M Shorts views or 8,000 hours. Apply before 31 Jan 2027. Whether an application still in review on that date keeps the old bar is unknown.
- Review typically takes about a month. After a rejection plan on a 90-day wait, so there is effectively one attempt at the current bar.
- Typical new channel in this format: about 1,000 views per Short. Winners posted 1 to 2 a day and had a hit by day 15. Mass-uploaded templated channels get nothing.
- Monetization is refused for generic, repetitive, template-based content, judged channel-wide. Mass production can also be treated as spam.
- API uploads from unaudited projects are reportedly no longer locked private since 2 Nov 2026 (read in the API description, not tested with a real upload); 100 uploads a day per project.
- Not verified: whether any tracked channel is monetized; how reviewers treat simulation formats.

## Content rules

1. Every video is a different mechanic or a rule-changing variant. Never a parameter swap.
2. The question is readable within the first second; motion starts on frame 0; 24 to 32 seconds; the last frame cuts back to the first.
3. The outcome is a real simulation. Seeds are searched for close finishes, never edited or faked.
4. Essential content stays out of the top 8%, bottom 20% and right 12% of the frame.
5. Colour is never the only cue: each contestant also has a symbol.
6. Sound is synthesized here. No third-party music, brands, characters, logos or likenesses.
7. Not made for kids. No synthetic-media label needed (stylised animation).
8. Each mechanic gets its own look and sound; the font and overall finish stay consistent.
9. The question, the rule that decides it, the result line and the title must agree. State the winning rule on screen in the hook (for example LAST ONE LEFT WINS).
10. No hidden numbers decide the outcome: anything that drives the result is visible from frame 0. A deciding near-miss must be readable on a phone; gate seeds on it.
11. Vary the skeleton, not only the mechanic: hook placement and motion, end-card wording, length and contestant shape differ between neighbouring uploads. Videos that share a look are never scheduled next to each other.
12. Each description is written for its video and names every rule (pickups, steering, speed-ups). No shared boilerplate paragraph. Say "no footage, samples or stock media".

## Commands

    pip install -r requirements.txt
    python make.py <scene> --out out/            # seed search, then render
    python make.py <scene> --seed N --out out/   # exact re-render
    python qa.py out/<file>.mp4                  # must pass before a video counts
    python research/track.py                     # daily competitor snapshot -> research/latest_summary.md

Scenes are discovered from `scenes/*.py`. Videos are not committed; they are regenerated from the seed in `out/<file>.json`.

## Daily routine

1. `python research/track.py`; read `research/latest_summary.md`; append 3 to 6 lines to `research/journal.md` (what gained, what is dying, one change for our next videos). One or two web searches only when the data needs explaining.
2. Pick the next videos from `plan/calendar.md`, adjusted by the journal and, once live, by our own numbers.
3. If finished inventory is under 14, build the next two scenes (specs in `plan/mechanics.md`), run `qa.py`, look at frames, fix defects. If inventory is 14 or more and the channel is not live, skip production.
4. Update the inventory table below; commit and push (no videos, no secrets).
5. Finish quietly unless a milestone was hit or only the owner can unblock something.

## Usage discipline

- Scripts do the data work. No broad research crawls.
- At most one small workflow a day (four agents or fewer); give builders a tool-call budget and a cap of three polish rounds.
- Measured cost, batch of 16 Nov: three builders, one reviewer, one fixer = 1.25M agent tokens (builders about 220 to 265K each, reviewer 145K, fixer 370K). Budget a day at two builders plus one combined review-and-fix agent, about 800K. Put rules 9 to 12 in every builder brief so the fixer has less to do.
- Look at frames as downscaled contact sheets, not full-size stills.
- Keep main-session tool calls few; batch shell work into single commands.
- If the usage limit interrupts a build, check the files on disk before re-running anything: agents usually finish most of the work.

## Publishing and viewer interaction (blocked on the owner)

The agent cannot create or sign in to a YouTube channel. Until the owner hands over channel access, nothing is published and no comments can be answered. When access exists: claim @predixel, apply `channel/identity.md` and the art in `channel/`, upload two a day per `plan/calendar.md`, pin the prediction comment and follow `plan/engagement.md`.

## Inventory

| # | Scene | Seed | Status |
|---|---|---|---|
| 1 | territory | 267 | ready: QA pass, independent review done (minors fixed), frames checked by owner agent |
| 2 | eventhorizon | 821 | ready: review blocker and five majors fixed, QA pass (one mix note: 6 s without a sound onset), frames checked |
| 3 | fusetag | 296 | ready: review major fixed, QA pass, frames checked |
| 4 | trailtrap | 2123 | ready, weakest of the four: rule changed after review (every crash clears the board), QA pass, frames checked |

All four: sound verified by measurement only, never heard; picture verified from downscaled stills, never watched in motion. Open items from the review are listed in the decision log.

## Decision log

- 16 Nov: niche = code-made prediction simulations; name = Predixel; week-one order set in plan/calendar.md.
- 16 Nov: third mechanic-design pass, the judge and the setup review were cut by the usage limit; ranking done by the owner agent; setup review folded into the next batch review.
- 16 Nov: batch review of videos 1 to 4 found 1 blocker, 10 majors, 9 minors; fixes applied and re-rendered (eventhorizon seed 260 to 821, trailtrap seed 214 to 2123). Rules 9 to 12 added from it. Upload order changed so territory and fusetag (similar look) are not neighbours.
- 16 Nov: left open on purpose: territory opens on an empty board (colour splash on frame 0 needs a rule change and new seed); fusetag late rounds are sparse (closing walls would fix it); trailtrap's first crash is at 9.3 s, its duel runs at over 3x speed and frame 0 shows a half-typed question; eventhorizon's winner is the least-drag planet and its clearance is thin on a phone; all four share the 2 s zoom-in and half-second reset. Revisit only if retention data points at them.
- 16 Nov: channel art checked by the owner agent at avatar sizes 300, 98 and 36 px and banner safe area; reads cleanly. Independent review of identity, engagement plan and tracker still not done; fold into the next batch review.
