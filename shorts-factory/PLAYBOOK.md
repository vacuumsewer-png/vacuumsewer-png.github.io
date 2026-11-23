# Predixel operating playbook

Source of truth for the agent that runs this channel. Keep it short and current.

## Mission and owner instructions (16 Nov 2026)

- Run a new faceless YouTube Shorts channel, **Predixel (@predixel)**, as its owner. Reach YouTube Partner Program thresholds as fast as possible; the owner's target is one month from first upload.
- Decide everything yourself. Do not ask the owner for decisions. Report only milestones or blockers that only the owner can clear.
- Be careful with usage: it has run out three times. See "Usage discipline".
- Quality bar: the best in the format. Every video is reviewed before it counts as inventory.
- Owner's choices: no spoken language, free tools only, niche chosen by research (code-made prediction simulations).
- Audience: United States (owner, 16 Nov). American English in everything viewers see; upload slots in US Eastern time (plan/calendar.md). The channel's country-of-residence setting stays the owner's real country: it is used for Partner Program eligibility and payments.

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
13. All on-screen text, titles, descriptions, tags and comments use American spelling (color, favorite, center).
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

## Where the work runs (owner, 16 Nov)

All work now runs in a session on the owner's own computer: research, production and everything on YouTube. The owner signs in to YouTube in that machine's browser; never ask for a password or code in chat. The earlier cloud session and its daily task are stopped. Rendered videos live only in `shorts-factory/out/` on that computer and are never committed. A daily scheduled task `predixel-daily` (Claude desktop app, 09:00 IST) runs the daily routine.

## Publishing and viewer interaction (needs the owner signed in on this computer)

The agent never signs in itself. Once the owner is signed in to YouTube in this computer's browser: claim @predixel, apply `channel/identity.md` and the art in `channel/`, upload two a day per `plan/calendar.md`, pin the prediction comment and follow `plan/engagement.md`.

## Inventory

| # | Scene | Seed | Status |
|---|---|---|---|
| 1 | territory | 267 | ready: QA pass, independent review done (minors fixed), frames checked by owner agent |
| 2 | eventhorizon | 821 | ready: review blocker and five majors fixed, QA pass (one mix note: 6 s without a sound onset), frames checked |
| 3 | fusetag | 296 | ready: review major fixed, QA pass, frames checked |
| 4 | trailtrap | 2123 | ready, weakest of the four: rule changed after review (every crash clears the board), QA pass, frames checked |
| 5 | pressurepie | 88 | ready: QA pass, frames checked by owner agent (minor: muddy 0.6 s rewind crossfade) |
| 6 | fuserace | 34 | ready: QA pass, finale reworked after owner-agent review, frames checked |
| 7 | mazeflood | 80 | scheduled: QA pass, frames checked (minor: small wall-follower icons, 3 s still on the result) |
| 8 | tippingpoint | 2750 | scheduled: QA pass, frames checked (minor: dense late piles, mild near-miss) |
| 9 | magnetchaos | 126 | scheduled 23 Nov 12:00 ET (A9QKEfoTTiA, moved from 22 Nov 19:00 for Domino Giant 2): QA pass except loop cut 1.63 (in-video max 16), frames checked; needs MAXLEV 12 (zoom 4^12) |
| 10 | dominogiant2 | 90 | REJECTED by owner 22 Nov (copy of Domino Giant): set private, never publish (Tdx9NKZ61b0): QA pass after AAC limiter fix (true peak -1.8), frames checked; GROW 1.65 (1.7 stalls at domino 2), 1 km tower, answer 21, lock-in countdown + streak end card |
| 11 | starforge | 155 | scheduled 23 Nov 19:00 ET (dLsv56j5Ekc): QA pass (agent build), ChatGPT title "3 Cores, 600 Grains… Who Becomes the Biggest? 🤯" + thumb p7_starforge.png; Related video -> Event Horizon; files on H: |

All four: regenerated on the owner's Windows computer 17 Nov and QA clean. Reviewed as 2 fps contact sheets over the full length plus spectrograms and noise-flatness measurements; no agent can listen, so audio is still judged by measurement only. Upload copies (2-pass, under 10 MB for the Chrome upload tool) are in `out/upload/` and pass QA.

## Decision log

- 16 Nov: niche = code-made prediction simulations; name = Predixel; week-one order set in plan/calendar.md.
- 16 Nov: third mechanic-design pass, the judge and the setup review were cut by the usage limit; ranking done by the owner agent; setup review folded into the next batch review.
- 16 Nov: batch review of videos 1 to 4 found 1 blocker, 10 majors, 9 minors; fixes applied and re-rendered (eventhorizon seed 260 to 821, trailtrap seed 214 to 2123). Rules 9 to 12 added from it. Upload order changed so territory and fusetag (similar look) are not neighbours.
- 16 Nov: left open on purpose: territory opens on an empty board (colour splash on frame 0 needs a rule change and new seed); fusetag late rounds are sparse (closing walls would fix it); trailtrap's first crash is at 9.3 s, its duel runs at over 3x speed and frame 0 shows a half-typed question; eventhorizon's winner is the least-drag planet and its clearance is thin on a phone; all four share the 2 s zoom-in and half-second reset. Revisit only if retention data points at them.
- 16 Nov: channel art checked by the owner agent at avatar sizes 300, 98 and 36 px and banner safe area; reads cleanly. Independent review of identity, engagement plan and tracker still not done; fold into the next batch review.
- 16 Nov: owner set the audience to the United States. Video 1 re-rendered with COLOR (same seed, same result); plans and channel kit switched to American spelling; upload slots set to 12:00 and 19:00 US Eastern as an unverified starting point.
- 17 Nov: channel created: Predixel, @predixel (id UCEDAAp2gSSw5J38SRza4dUQ), a brand channel under the owner's Google account next to their other channels. Description, keywords, not-made-for-kids, avatar, banner and watermark applied. Country of residence left for the owner.
- 17 Nov: Windows fixes: render pool falls back from fork to spawn or serial; BT.709 tags via setparams (QA failed without). Event Horizon dive whoosh followed summed proximity (always about 1) and hissed under the whole video; now follows the closest planet cubed.
- 17 Nov: uploads go through Claude in Chrome (`file_upload`, 10 MB cap) because the built-in browser pane cannot drive a file dialog. Use `out/upload/*.mp4`.
- 17 Nov: RESOLVED (was BLOCKED). First upload refused: "Daily upload limit reached", needs the owner's one-time phone verification (youtube.com/verify) or a 24 h wait. Git push also waits on the owner's GitHub sign-in on this computer.
- 17 Nov: owner verified Predixel. Territory (youtube.com/shorts/tvaF_XUmFuU) uploaded Private, played as a vertical Short with audio, then scheduled 17 Nov 12:00 ET. Event Horizon (youtube.com/shorts/0uuz0yfP0Gk) scheduled 17 Nov 19:00 ET. Both: not made for kids, altered content No, tags from out/*.json.
- 17 Nov: all four day-1/day-2 Shorts scheduled (Studio shows IST; all at New York time):
  - Territory tvaF_XUmFuU: 17 Nov 12:00 ET. Event Horizon 0uuz0yfP0Gk: 17 Nov 19:00 ET.
  - Trail Trap Bhe6fXd0Pt4: 18 Nov 12:00 ET (the frozen draft was finished, not duplicated). Fuse Tag prJ7rLytJ34: 18 Nov 19:00 ET.
  - Studio schedule picker: Time zone "(GMT-05:00) New York", pick the date first, then choose the time from the list (typed times are ignored).
- 17 Nov: packaging pass on all four scheduled Shorts (see plan/calendar.md "Packaging research"): new hook titles, `#simulation #satisfying` + topic hashtag on the description's last line, category Science & Technology, video language Not applicable, title/description language English (US). Thumbnails come from ChatGPT (owner), never our own code or raw frames.
- 17 Nov: built Pressure Pie (seed 88, 29.8 s) and Fuse Race (seed 34, 25.0 s; finale reworked after review: dead 4 s ending removed, staggered bursts, readable near-miss). Both QA clean. Descriptions had the result in them (spoiler); removed. Scheduled day 3: Pressure Pie l-z9BllHKOs 19 Nov 12:00 ET, Fuse Race z2uwgKn44wE 19 Nov 19:00 ET, both with category/languages set.
- 17 Nov: all six Shorts have ChatGPT-made thumbnails (out/thumbs/, not committed). First drafts for Territory, Pressure Pie and Event Horizon showed the winner leading or the winner dying; redone. Rule: check every thumbnail against drama.winner before upload.
- 17 Nov: the PC ran out of memory (about 1 GB free; owner's other render jobs plus Chrome) and Studio froze repeatedly. Keep one Chrome tab, close it when done.
- 17 Nov: after a reboot Studio's active channel was Just AI Minute; switched back to Predixel. Always check the active channel before posting comments.
- 17 Nov: Territory (tvaF_XUmFuU) public; prediction comment posted as Predixel and pinned. No viewer comments yet (replied to 0). Chrome's active identity was the owner's personal channel on another Google account; switched via avatar menu > Switch account > Predixel.
- 18 Nov: Event Horizon (0uuz0yfP0Gk) public; prediction comment posted as Predixel and pinned. No viewer comments on it or on Territory (replied to 0). Chrome's active identity was Strange Little Things; switched to Predixel at youtube.com/channel_switcher.
- 18 Nov: the 09:00 predixel-daily run stopped at its first step and Run now did not start a new one, so the main session did day 4. Built Maze Flood (seed 80, 26.2 s; water timing is drawn per seed so close finishes exist, only about 1% of seeds pass) and Tipping Point (seed 2750, 26.1 s; the upload copy's AAC must be encoded from the mastered WAV, because AAC to AAC overshot true peak). Titles drafted with ChatGPT; ChatGPT thumbnails in out/thumbs/p4_*.png (the first Maze Flood draft showed the winner, yellow, highest, so it was redone). Scheduled: Maze Flood ZfbaUr-fgic 20 Nov 12:00 ET, Tipping Point wfqispvAOZw 20 Nov 19:00 ET. Studio had switched to Strange Little Things; switched back via avatar > Switch account. Pressing Escape in the upload dialog closes it and loses the upload; never send Escape there.
- 18 Nov: Trail Trap (Bhe6fXd0Pt4) public; prediction comment posted as Predixel and pinned. No unresponded viewer comments on Trail Trap, Event Horizon or Territory (replied to 0). Active identity was already Predixel.
- 19 Nov: Fuse Tag (prJ7rLytJ34) public; prediction comment posted as Predixel and pinned. No viewer comments on Fuse Tag, Trail Trap, Event Horizon or Territory (replied to 0). Pressure Pie still Private (scheduled) at 00:30 ET. Active identity was IQ Technologies; switched at youtube.com/channel_switcher. Another job's ffmpeg renders left under 1 GB free and froze Chrome (Studio never loaded); waiting for each render to end and using watch pages worked.
- 20 Nov: Pressure Pie (l-z9BllHKOs) and Fuse Race (z2uwgKn44wE) public; prediction comments posted as Predixel and pinned. Studio unresponded-comments inbox empty (replied to 0). Maze Flood still Private (scheduled 12:00 ET). Active identity was Just AI Minute; switched at youtube.com/channel_switcher.
- 19-20 Nov: both predixel-daily runs died on network errors (ENOTFOUND). 19 Nov run built Storm Race (seed 736), Domino Giant (39), River Drift (1782); Magnet Chaos not finished. 20 Nov run left Storm Race as a raw-titled draft. Main session finished packaging (ChatGPT titles where better; ChatGPT thumbnails out/thumbs/p5_*.png, spoiler-checked) and scheduled: Storm Race P7iFtKhmtiE 21 Nov 12:00 ET, Domino Giant lUFVZSGBhFg 21 Nov 19:00 ET, River Drift kVq51DfsMIg 22 Nov 12:00 ET. Next free slot: 22 Nov 19:00 ET.
- 20 Nov: Maze Flood (ZfbaUr-fgic) public; prediction comment posted as Predixel and pinned (replied to 0, no viewer comments). Tipping Point not yet public (19:00 ET), left for the next run. Active identity was Strange Little Things; switched at youtube.com/channel_switcher. Free RAM about 0.5 GB with three other ffmpeg renders running; Chrome was slow but worked. YouTube autoplay can jump to another video on a watch page; use /shorts/<id>.
- 21 Nov: Tipping Point (wfqispvAOZw) public; prediction comment posted as Predixel and pinned (replied to 0, no viewer comments). Storm Race (P7iFtKhmtiE) still Private (scheduled 12:00 ET), left for the next run. Active identity was already Predixel.
- 21 Nov (owner rule): every Short has a Studio "Related video" link to a similar public Predixel Short. Set: Territory->Pressure Pie, Pressure Pie->Territory, Event Horizon->Tipping Point, Tipping Point->Event Horizon, Trail Trap->Maze Flood, Maze Flood->Trail Trap, Fuse Tag->Fuse Race, Fuse Race->Fuse Tag, Storm Race->Fuse Race, Domino Giant->Tipping Point, River Drift->Maze Flood. The picker lists only public videos; Studio save can be slow (check the "Changes saved" toast).
- TODO (each video once it is public): post and pin the prediction comment from the channel, then reply per `plan/engagement.md`:
  - Storm Race: "Blue, pink, gold or mint: which one touches the ground first? Pick before the first fork. Same 30 steps per second for all four, and dry air stalls any tip inside it."
  - Domino Giant: "How many dominoes did you guess? Each one is 1.5 times bigger than the last, starting at 5 cm, and the tower is 300 m. Post your number before you watch the end."
  - River Drift: "Arrow, Leaf, Kite or Twin: which float wins the river? Pick before they leave the start line. Nobody steers, the current and ten drifting whirlpools decide."
- 16 Nov: owner moved all work to a session on their own computer and ended cloud work. Inventory at handover: four finished videos (regenerate from the seeds in out/*.json); next builds are Pressure Pie and Fuse Race.
- 21 Nov (run 2): tracker + journal done. Magnet Chaos render failed: seed search picked seed 126 which raises "still mixed at zoom 4^8" in scenes/magnetchaos.py _maps; needs a seed gate or higher MAXLEV. Studio work not done; scheduled buffer is only about 3 slots (target 14).
- 22 Nov: Storm Race (P7iFtKhmtiE) and Domino Giant (lUFVZSGBhFg) public; prediction comments posted as Predixel and pinned (0 viewer comments, replied to 0). River Drift still scheduled. Active identity was IQ Technologies; switched via avatar > Switch account. Magnet Chaos parked: seeds 126 and 828 both fail the zoom resolve at 4^8 (kest probes only 8 border points per level; full map mixes); scene needs a full-map seed gate. Scheduled buffer is nearly empty (only River Drift); Star Forge handed to a builder agent. Owner note: PC off means no new uploads, buffer cannot be built without finished videos.
- 22 Nov: first weekly scan (research/weekly/2026-11-22.md). 5 subs, 10K views/28 d, stayed-to-watch 33%. Best: Domino Giant 64%, Tipping Point 55% (question-first formats); weakest Event Horizon 24%. Retention leaks 3-9 s and after the reveal. New build rules (event by 2.5 s, PICK NOW countdown, 22-26 s, late reveal, streak end card, half question-first formats, Domino Giant 2 next) added to predixel-daily; pin task now replies/hearts and asks viewers to pick the next rematch. Weekly task predixel-weekly-analytics runs Sundays ~06:30 IST (off-peak, moved 22 Nov at owner request).
- 22 Nov (pin run, later): River Drift (kVq51DfsMIg) still Private (scheduled 12:00 ET), left for next run. Studio unresponded-comments inbox empty (replied to 0). Active identity was already Predixel. No rematch votes yet.
- 22 Nov: Star Forge built by agent (seed 155, 27.2 s, winner GOLD 252 vs RED 225 vs BLUE 210; QA pass incl. upload copy 9.15 MB; make.py now reads optional CEILING_DB per scene). Open: quiet 12-22 s middle. Not yet packaged or uploaded.
- 22 Nov (main session): Magnet Chaos fixed by raising MAXLEV 8 -> 12 (seed 126 resolves to all GOLD at 4^12; only seed 126 of 1-800 passes the scene's validity gate). Rendered, QA'd, ChatGPT title "3 Magnets, 4 Pixels Apart… Which One Catches It? 🧲" and thumbnail out/thumbs/p6_magnetchaos.png (spoiler-checked), scheduled A9QKEfoTTiA 22 Nov 19:00 ET, Related video -> Domino Giant. Winner GOLD (second drop AQUA). Pinned comment text: "Gold, aqua or rose: which magnet did you call? Pick before the release, then watch what 4 pixels do to the answer."
- 22 Nov evening: Domino Giant went viral (32.2K views in 11.5 h, 12.1K in the last hour, 76.1% stayed, +28 subs; channel 29 subs). Built DOMINO GIANT 2 (scenes/dominogiant2.py, seed 90): each domino 1.65x bigger, the largest ratio this model topples (1.7x stalls at domino 2), 1 km tower, answer 21, first build with the weekly-scan rules (LOCK IN YOUR NUMBER 3-2-1, streak end card). ChatGPT title "Part 2: How Many Dominoes to Topple a 1 KM Tower? 🤔" and thumbnail out/thumbs/p6_dominogiant2.png (no count shown). Scheduled Tdx9NKZ61b0 22 Nov 19:00 ET; Magnet Chaos moved to 23 Nov 12:00 ET. Related video DG2 -> Domino Giant. TODO once DG2 is public: set Domino Giant's Related video to DG2 (Part 1 -> Part 2). Pinned comment text for DG2: "Part 2! Each domino is 1.65 times bigger (at 1.7 times the chain stalls), starting at 5 cm, and the tower is 1 km. How many dominoes did you guess? Got Part 1 right too?"
- 22 Nov late: owner rejected Domino Giant 2 as a copy of Domino Giant ("let the domino wave settle, then analyze and follow the pattern/structure, not the same video"); Tdx9NKZ61b0 set Private, never publish. Owner storage rule: uploaded videos move to H:\My Drive\Predixel videos (done for all 13 uploaded Shorts + review stills; only Star Forge stays on D: until uploaded). Owner reports Domino Giant at ~35K views, channel 35 subs.
- 23 Nov: queue now Magnet Chaos A9QKEfoTTiA 23 Nov 12:00 ET, Star Forge dLsv56j5Ekc 23 Nov 19:00 ET (Related video -> Event Horizon). Domino Giant 2 confirmed Private. Channel 37 subs. Buffer after 23 Nov is empty: daily task must build new mechanics (no copies). Note: Studio's active channel keeps drifting (Science in 10s, Monk & Honk); switch via Studio avatar > Switch account > Predixel. ChatGPT reused an old domino chat once and returned the wrong image: always start a fresh chat and check the image matches the video. Pinned comment for Star Forge: "Blue, red or gold: which core did you call? Pick before the dust starts to merge."
- 23 Nov (daily run): tracker+journal done. River Drift (kVq51DfsMIg) public; prediction comment posted as Predixel and pinned (0 viewer comments). Magnet Chaos (23 Nov 12:00 ET) and Star Forge (23 Nov 19:00 ET) still scheduled (Studio shows Nov 23 / Nov 24 IST). Built STRIKE by agent (scenes/strike.py, seed 89, 25.0 s, winner SKY, runner-up GOLD by 1 cell, QA pass incl. upload copy 9.15 MB; leader exponent 0.7 not 2). Title idea "5 Towers, 1 Lightning Leader… Which Gets Struck? ⚡"; still needs ChatGPT title/thumbnail + Studio schedule. Unshatter builder running. Studio active channel was already Predixel.
- 23 Nov (daily run, later): built and scheduled two new mechanics (both question-first, PICK NOW countdown, late reveal): STRIKE PWmwZEXqULs 24 Nov 12:00 ET (seed 89, winner SKY, runner-up GOLD; ChatGPT thumb p7_strike.png; Related video -> Storm Race) and UNSHATTER 4I524l9IJu4 24 Nov 19:00 ET (seed 442, answer BULB; title "What Was It Before It Broke? Pick One"; thumb p7_unshatter.png shows only abstract shards; Related video -> Domino Giant). Files moved to H:. Buffer is now 2 slots after Star Forge; next free slot 25 Nov 12:00 ET. Remaining mechanics in plan/mechanics.md: Bridge Load Test, Breaking Point. Pinned comments to post once public: Strike "Coral, lime, sky, violet or gold: which tower gets struck? Pick before the leader reaches the ground. The leader grows toward the strongest field, and the first tower it touches is hit." Unshatter "Key, bulb or vase: what was it before it broke? Pick before the shards lock together. The rebuild is the shatter recording played backward." Magnet Chaos and Star Forge pins still pending (public 23 Nov). Note: file_upload may report a timeout/offline error even though the upload went through; check the dialog before retrying.
- 23 Nov: owner showed the channel grid (desktop and mobile) cutting the top line of every thumbnail's text; on phones the bottom ~20% is also covered by the view count. Fixed by ChatGPT edits of each original thumbnail (text moved to 25-50% height, nothing else changed) and replaced in Studio: Magnet Chaos, Star Forge, Strike, Tipping Point, Fuse Tag, Maze Flood, Trail Trap, Territory, Storm Race, Fuse Race, Pressure Pie, Event Horizon, River Drift (files out/thumbs/*_v2/_v3.png). Domino Giant left unchanged while it is viral; Unshatter already fine. Tipping Point's old thumbnail tilted the beam down on the winning RIGHT side (a spoiler); the new one is level. Rule added to predixel-daily: text in the middle band, check the 20-75% crop before upload. Domino Giant at 51,117 views.
