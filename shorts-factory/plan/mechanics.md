# Mechanics backlog

Ranked 2026-11-16 by the channel owner agent from two independent design passes (prediction lens, spectacle lens).
Ranking criteria: hook readable in under a second, sustained tension, distinct from saturated formats, honest payoff, feasible in 2D pycairo, policy-safe.
Video 1 is TERRITORY BATTLE (built, scenes/territory.py). BRIDGE LOAD TEST and BREAKING POINT are one idea (pick one); STRIKE is a variant of STORM RACE.

## 2. FUSE TAG  (complexity: low)

- **Hook:** Who is holding it at zero?
- **Rules:** Six balls bounce in a box. One carries a lit charge with a visible countdown ring of four to six seconds. Touching another ball passes the charge; the passer is immune for half a second. The carrier moves 30% faster. At zero the carrier bursts and the charge jumps to the nearest ball. Last ball wins.
- **Tension:** The countdown ring is always visible, so every second has a concrete question: will the carrier reach someone before zero. Because the carrier is faster it actually chases, so passes cluster in the last half second of each round. Five bursts in about 25 seconds give a beat every 4 to 6 seconds, and each burst empties the box, so passes get rarer and holds get longer. The final round is a two-ball duel with one charge. Seed-search targets (design heuristics, not measured): at least 3 passes in the final round, last pass within 0.7 s of zero, the first carrier is not the first ball out.
- **Payoff:** A last-instant pass in the final duel, then the loser bursts into particles and the winner is left alone with a slow glow and a pass-count readout. The viewer's pick was in danger at least once in every round.
- **Variants (rule-changing):**
  - Two charges with different timers are live at once; one ball can hold both, and both burst separately.
  - Crown mode (reversed stakes): the token is safe to hold; at zero the ball farthest from the holder bursts, shown by a distance line.
  - Rotating divider: a slowly spinning wall with one gap splits the box into two chambers, so the charge can be trapped on one side and viewers must track the gap.
- **Audio:** Metronome clicks whose rate and pitch rise as the ring empties; each pass is a short two-tone sine zip panned by x position; burst is a noise burst with a low-pass sweep plus a sub-bass thump; one voice leaves the mix after each burst so the texture thins; winner gets a held additive-sine chord.
- **Risk:** NOTE: the list is ordered by suggested build priority (my heuristic: lens fit against build cost). Cross-cutting search finding from 8 YouTube result pages on 2026-11-16 (small, relevance-ranked samples, heuristic): almost every plain pick-a-colour premise already exists at roughly 1K views per Short; the few outliers seen stated a counter-intuitive or escalating rule in the title (for example one elimination sim at about 550K views, a track race at about 30K, a bridge test at about 33K). So the rule twist and close finish matter more than the premise. Specific to this mechanic: saturation of hot-potato sims is unknown (not searched); adjacent musical-chairs ball sims were seen at about 1K to 6K views. A cartoon bomb would look toy-like and child-directed, so draw an abstract spark or charge. A hidden timer would feel rigged, so the ring must always be visible. Balls in a box is the most generic look in the niche; it needs distinct styling.

## 3. TRAIL TRAP  (complexity: low)

- **Hook:** Which line outlasts the others?
- **Rules:** Four heads move one cell per tick on a 36 by 64 grid, leaving permanent trails. Hitting any trail or wall eliminates that head. Each head uses one rule: go straight until blocked, hug the wall, turn toward the larger open region, or turn randomly. Speed rises 5% every three seconds. Last head moving wins.
- **Tension:** Each colour shows a counter of free cells it can still reach. That number swings violently: one rival crossing in front can cut a head's region from 600 cells to 40 in a single tick. Once a head is sealed in a pocket viewers can count its remaining room, and the question becomes whether it fills the pocket efficiently or wastes it. The usual ending is two heads in separate pockets of similar size. Seed-search targets (heuristic): at least 2 large swings in the reachable-area lead, last two eliminations within 15 ticks of each other, first elimination after 8 s.
- **Payoff:** The final two spiral inside their pockets with the cell counters ticking down side by side; one runs out a few ticks before the other. The winner's whole trail pulses from start to head, and the unused cells of the loser's pocket are highlighted.
- **Variants (rule-changing):**
  - Area scoring: after all heads stop, every enclosed region is filled with the colour of the trail that surrounds it, and the largest area wins, so stopping first can still win.
  - Fading trails with a closing border: trails vanish after 40 ticks while the outer wall moves in one ring every 3 seconds, so pressure comes from the arena, not the trails.
  - Wrap-around pairs: edges wrap to the opposite side and heads play two against two, with teammates' trails passable.
- **Audio:** Each head is a square-wave step tick at its own pentatonic pitch, with tempo rising as speed rises; a turn bends the pitch briefly; a crash is a short bit-crushed noise burst after which that voice stops, so the pattern thins from four voices to a duet to a solo.
- **Risk:** The look and vocabulary of light cycles belong to a film franchise: no neon bikes, and do not use the franchise name or light-cycle in titles, tags or hashtags. One similar pick-a-colour light-trail Short was seen at about 1.2K views, alongside many film clips. The larger-region rule may win too often; check the winner distribution across a few hundred seeds and rebalance (for example limit its look-ahead). It resembles the generic snake game, which is acceptable but should not be styled as a retro game for children.

## 4. EVENT HORIZON  (complexity: medium)

- **Hook:** Which planet falls in last?
- **Rules:** Five planets orbit a central black hole on different ellipses under inverse-square gravity, with weak drag that slowly shrinks every orbit. Planets do not steer. A planet crossing the horizon is swallowed, and the horizon radius grows 15% with each meal. Last planet outside the horizon wins.
- **Tension:** Every close pass is a dive: the planet accelerates, whips around the horizon and either clears it or does not, and with five planets a dive happens roughly every one to two seconds. A readout shows each planet's closest gap so far, which acts as the standings and reorders as orbits shrink at different rates. Each meal makes the horizon jump, which instantly puts the next-closest planet in danger. Seed-search targets (heuristic): the eventual winner survives at least one pass with a gap under 5 percent of the horizon radius, the last two planets are swallowed or decided within 3 s, first meal after 7 s.
- **Payoff:** The last two planets dive on alternating passes; one clears by a few pixels and the other is swallowed on the next pass. The survivor's trail is left drawn as a rosette around the enlarged horizon with its smallest gap displayed.
- **Variants (rule-changing):**
  - Binary hole: two black holes orbit each other and either can swallow, giving chaotic figure-eight paths and sudden hand-offs.
  - Merging planets: planets attract each other and merge on contact, the heavier colour surviving; the last body left wins.
  - Escape race (reversed goal): no drag, a passing star kicks the orbits, and the first planet flung beyond the frame wins.
- **Audio:** Each planet is a sine tone whose pitch follows its orbital speed, so every dive is a rising then falling whoosh; the hole is a sub-bass drone that drops a semitone with each meal; a swallow is a downward glissando followed by a short silence; the survivor ends on a sustained fifth.
- **Risk:** Two near-identical which-planet-survives-the-black-hole Shorts were seen at about 211 and 1.3K views, so the dive-gap readout and growing horizon must be visibly central. Drag and horizon growth are game rules, not astrophysics, and the video must not claim realism. Do not use real planet names or photographic textures (third-party imagery and implied factual claims). Outcome can be partly guessed from the opening orbits; that is fair, but seeds where one planet is obviously safest should be rejected. Not child-directed if kept abstract.

## 5. PRESSURE PIE  (complexity: medium)

- **Hook:** Which colour takes the whole circle?
- **Rules:** Four spokes split a disc into four coloured wedges; one ball bounces inside each. A ball hitting a spoke rotates it away, by an amount proportional to that ball's wedge angle, growing 3% per second. A wedge under 14 degrees pops and its neighbours close the gap. Last colour holding the full disc wins.
- **Tension:** Territory is a live number: each wedge shows its percentage on the rim. A small wedge makes its ball hit the spokes more often, a large wedge makes each hit stronger, so the two effects roughly cancel and the borders random-walk instead of snowballing (expected behaviour, to be confirmed in simulation). That gives frequent changes of the largest colour and visible comebacks: a wedge squeezed to 20 degrees rattles frantically and can re-expand. Push strength grows every second, so swings get bigger and a pop becomes inevitable. Seed-search targets (heuristic): at least 6 changes of the leading colour, first pop after 10 s, last two colours within 40 to 60 percent split at 20 s, final pop between 24 and 29 s.
- **Payoff:** The last spoke pair sweeps shut, the final small wedge pops, and the whole disc floods to one colour while the winner's percentage counts up to 100. A short trace of each colour's share over time is drawn around the rim.
- **Variants (rule-changing):**
  - Defectors: when a wedge pops, its ball changes colour and joins the wedge that delivered the final push, so that colour now has two balls.
  - Stacked bars: the same push rule in a tall rectangle with horizontal dividers and gravity, so lower colours are hit by faster falling balls and upper colours by slower ones.
  - Leaky spokes: each spoke has a small gate; a ball that slips through keeps its colour and pushes from inside enemy territory until it bounces back out.
- **Audio:** Each spoke hit is a plucked sine-plus-harmonics note whose pitch is mapped to wedge size (smaller wedge, higher pitch), so a dying wedge audibly panics; pop is a noise burst with a falling pitch sweep; a quiet bass pulse speeds up with the push multiplier; the finish is a major chord built from additive sines.
- **Risk:** Search on 2026-11-16 showed several circle or territory pick-a-colour Shorts at about 1K to 15K views (for example a which-colour-takes-over-the-circle title at about 1.4K); none of the visible titles described moving spokes, but whether this exact rule exists is unknown. The comeback effect can look rigged unless a small icon legend shows the rule (bigger wedge, bigger push). The mean-reverting or neutral behaviour is my expectation, not tested; the exponent may need tuning so rounds end on time. It is close in theme to video 1 (territory), so schedule it several days apart. Not child-directed.

## 6. FUSE RACE  (complexity: low)

- **Hook:** Which rocket launches first?
- **Rules:** One spark lights a hub. Five tangled fuses (smooth random splines, lengths within 12% of each other) lead to five coloured rockets. Flame travels at constant speed along arc length; where two fuses touch, it jumps across both ways. First flame to arrive launches its rocket; the rest follow in arrival order. Seeded, outcome computed, never edited.
- **Tension:** Seconds 0-2: the viewer traces the tangle and picks. All sparks crawl at the same visible speed, so the apparent leader keeps changing as loops and detours burn away. Junction jumps create sudden shortcuts the viewer may or may not have spotted. Last 3 seconds: two sparks are within a short distance of their rockets.
- **Payoff:** Winner launches on a bright trail and bursts into a large firework in its colour across the upper frame; the other four fire in arrival order as a 3-second finale. Falling embers converge on the hub and become the opening spark, which closes the loop. Readable with sound off.
- **Variants (rule-changing):**
  - Two fuse thicknesses: thick fuse burns at half speed, so the shortest route is no longer the fastest (weighted shortest path).
  - One branching fuse tree with dead ends and one-way burn gates: only some rockets are reachable, question becomes which rocket is reached at all.
  - Two sparks lit at opposite ends of one network with numbered markers along it: question becomes where the flames meet; the meeting point detonates.
- **Audio:** Band-passed white-noise hiss per active spark, centre frequency rising as it nears its rocket; sparse random impulse crackle. Launch: sine sweep 200 to 1800 Hz with noise. Burst: 60 Hz sine thump with exponential decay, noise burst, then sparse high sine pings on a pentatonic chord.
- **Risk:** Fireworks can read as festive kids content: keep a dark technical look, no faces or characters. Never change burn speed mid-run; near-ties must come from seed selection only, stated as such. Tangle layouts look alike quickly, so treat as occasional and change the rule set each time. One search found no simulated Short of this premise in the top results (real firework clips dominated); that is a single-query heuristic, not proof. LENS NOTE from the 8 searches run on 2026-11-16: mousetrap chain reaction, Chladni sand plate, Fourier epicycle drawing and raindrop race were considered and dropped because each already has several simulated Shorts with near-identical question titles sitting at roughly 50 to 7.5K views (one epicycle video at about 16K); obvious brainstorm ideas appear to be already mass-produced by small code channels.

## 7. MAZE FLOOD  (complexity: medium)

- **Hook:** Who escapes the flood?
- **Rules:** Four runners start at the bottom of one maze, exit at the top. Each obeys one rule: follow left wall, follow right wall, always take the unexplored branch nearest the exit, or random at junctions. All move one cell per tick. Water rises row by row, accelerating. A flooded runner is out. First through the exit wins.
- **Tension:** A gauge beside each runner shows its height above the water, and that number goes down as often as up, because the wall followers take long detours and dead ends often point downward. Viewers can see a dead end before the runner does. The water accelerates, so a detour that was safe at 8 seconds is fatal at 20. The lead is the shortest-path distance to the exit and it changes whenever a leader commits to a wrong branch. Seed-search targets (heuristic): at least two runners alive at 80 percent of the runtime, winner not leading at halfway, winner exits with 3 or fewer cells of margin.
- **Payoff:** One runner slips through the exit with the water one or two cells behind, the maze fills completely, and the winner's whole route is drawn over the flooded maze. An occasional honest episode where nobody escapes is a usable twist.
- **Variants (rule-changing):**
  - Closing from both ends: hazard advances from top and bottom toward an exit in the middle row, so heading upward is no longer always right.
  - Timed doors: several doors open and close on a visible cycle; two rules wait at a closed door and two reroute, making patience the variable.
  - Breadcrumbs: runners leave coloured trails and one runner's rule is to follow the freshest trail it meets, so a follower competes against three explorers.
- **Audio:** Each runner has a step tick at its own pitch, so four runners form a shifting polyrhythm that thins as runners are lost; the water is low-passed noise plus a slowly rising sine drone that accelerates with the water level; a lost runner is a short downward sine blip; the escape is a bright rising arpeggio.
- **Risk:** Stakes read as drowning, so runners must be abstract dots or arrows, never characters, to avoid a distressing or child-directed look. Maze-solver visualisations are an established genre; whether the flood-plus-rules version exists is unknown (not searched). Rule labels need icons instead of English words. If the exit-seeking rule wins most seeds the series becomes predictable, so check the winner distribution over a few hundred seeds and never claim one rule is generally best. Rising-lava ball Shorts are crowded (many seen at about 1K views), so the hazard should look like water in a maze, not lava under balls.

## 8. TIPPING POINT  (complexity: medium)

- **Hook:** Left or right?
- **Rules:** A beam balances on a central pivot with small rotational damping. Every 0.4 s a block of random mass (1 to 5, shown by size) drops at a random spot. Blocks stick by friction until tilt exceeds 12 degrees, then slide outward and add torque. The side whose tip touches the floor first loses its load and the round.
- **Tension:** The tilt angle is a live suspense meter, shown with a thin arc gauge against the 12-degree slide threshold. Every drop visibly swings the beam one way or the other; a heavy block near a tip can reverse the lead. Near the threshold blocks start to creep, so the viewer sees the runaway coming a moment before it happens.
- **Payoff:** Runaway: all blocks avalanche down the beam and pour into a pit in a clattering cascade. The emptied beam whips back, oscillates and settles level, which is the first frame.
- **Variants (rule-changing):**
  - Off-centre pivot (unequal arms): the short arm needs more mass, so the lever rule changes what heavier means.
  - A continuous sand stream replaces blocks and switches sides on a visible schedule; sand flows downhill at any tilt, so there is no friction threshold, only feedback.
  - Two stacked beams: whatever slides off the top beam lands on the lower one; question becomes which pit collects more.
- **Audio:** Landing: short sine thunk pitched by mass (heavy is low). Quiet continuous sine rising in pitch with absolute tilt. Creak: low-amplitude FM saw scaled by angular velocity. Avalanche: dense burst of the thunk notes over brown-noise rumble. Settle: one soft bell tone.
- **Risk:** Surface similarity to the tipping and edge-balance videos gaining in the tracker: this is a 2D lever with different rules; do not imitate their look or titles. A binary outcome repeated often looks templated, so use sparingly and change the rule each time. Do not select only seeds with last-second reversals and present them as typical. Seesaw imagery can look like playground content: draw it as lab apparatus. Search showed seesaw content is mostly physics lessons with low views; demand for this framing is unproven.

## 9. STORM RACE  (complexity: low)

- **Hook:** Which bolt strikes first?
- **Rules:** Four coloured lightning leaders start at the cloud line. Every tick each leader extends from one of its tips by a short random-angle step biased downward; 12% of steps fork a new tip; tips stall inside drifting dry-air patches. All leaders get the same steps per second. First leader to touch the ground wins.
- **Tension:** Markers at the screen edge show each colour's lowest tip, so the lead is readable at a glance. Growth is bursty: a lean leader with few forks drops fast but can stall in a patch, while a heavily forked one is slower per tip but always has a branch that keeps going, so a side branch from a trailing colour can suddenly take the lead. The ground line gets closer every second and the last tenth of the height is usually two tips a few pixels apart. Seed-search targets (heuristic): at least 5 lead changes in lowest tip, winner not leading at 70 percent of the descent, final gap under 3 percent of the height.
- **Payoff:** The winning leader's full path from cloud to ground lights up as one bright return stroke with a single soft bloom, the other three fade, and the runner-up's remaining gap is shown in pixels or metres of screen height.
- **Variants (rule-changing):**
  - Rod strike: the ground has four coloured towers of different heights that attract tips within a radius; one leader descends and viewers bet on which tower is hit.
  - Two-way connection: streamers also grow upward from four ground points; the strike happens where any leader meets any streamer, and the pair that connects wins.
  - Field-driven growth: random steps are replaced by a diffusion-limited rule on a relaxed potential grid, so neighbours shield each other and an early lead tends to hold unless a patch blocks it.
- **Audio:** Sparse noise clicks for each growth step, with click density following the number of active tips and panning following x position; a rising filtered-noise swell as altitude falls; the strike is a broadband burst into a low-passed brown-noise rumble tail with a falling sub sine.
- **Risk:** The outcome is driven by seeded random steps, so it can feel like a lottery; show the same-steps-per-second rule and print the seed. It is a stylised growth model inspired by stepped leaders, not an electrostatic simulation, and must not be described as real lightning physics. Bright flashing is a photosensitivity risk: use one smooth bloom, no strobing. Saturation is unknown (not searched). Not child-directed.

## 10. MAGNET CHAOS  (complexity: medium)

- **Hook:** Which magnet catches the pendulum?
- **Rules:** Top view: a pendulum bob hangs above three coloured magnets at the corners of a triangle. Forces: spring pull to the centre, inverse-square attraction to each magnet (softened by height), linear friction. Release from a marked point; it stops over one magnet. Then run every start point on a 270 by 480 grid and colour each pixel by its winner.
- **Tension:** Act 1 (about 11 s): the bob sweeps past each magnet with a fading trail, almost settles on one and is pulled away again; the apparent winner changes several times as energy drains. Act 2: a ghost bob released a hair's width away ends on a different magnet, which raises the question of what every other start point would do.
- **Payoff:** The map of all outcomes fills in as a three-colour fractal: solid basins near each magnet, endlessly interleaved bands between them, with the release point sitting on a boundary. The camera zooms into that point until one colour fills the screen, then cuts to a new release for the loop.
- **Variants (rule-changing):**
  - Four or five magnets in an irregular layout, one of them repelling and drawn with a different symbol: the bob can never rest there and the basins warp around it.
  - Double pendulum instead of magnets: question becomes how many flips before it stops, and the map colours each pair of start angles by flip count.
  - Magnets on a slowly rotating turntable, simulated in the rotating frame with the extra swirl term: basins become spirals.
- **Audio:** Bob: continuous sine with pitch following speed and pan following x. Each magnet owns a note (root, fifth, octave) that swells when the bob is near, so lead changes are audible. Capture: that note rings out. Map fill: granular shimmer built from the three notes at the fill rate. Zoom: falling Shepard-style glide.
- **Risk:** Outcome is extremely sensitive to integrator and step size: the result shown must be the one computed, and the description should call it a numerical model. A three-colour pick can resemble pick-a-colour channels in a thumbnail; the fractal reveal is the differentiator. The second act is more explanatory than suspenseful, so a retention dip at the act change is possible (unknown). The full map is about 130,000 trajectories: vectorise and cache it. Shorts saturation was not checked (unknown).

## 11. DOMINO GIANT  (complexity: medium)

- **Hook:** How many dominoes to topple a tower?
- **Rules:** Side view. Domino 1 is 5 cm tall; each next one is 1.5 times bigger in every dimension, spaced at half its height. Each is a rigid block pivoting on its lower edge under gravity; contact transfers the push. The camera zooms out to keep the newest domino in frame, past scale markers, until one out-tops the 300 m tower.
- **Tension:** A counter ticks up with each fall while scale markers (coin, hand, person outline, house, tree, tower) slide into frame as the camera pulls back. Each fall is slower and heavier than the last, so the rhythm decelerates while the scale escalates. Viewers hold their guessed number against the counter the whole time.
- **Payoff:** The final giant leans, hangs for a beat and lands beside the tower with a screen shake and dust ring; the counter freezes on the answer (by arithmetic the 23rd domino is about 374 m, the first above 300 m; recompute in code). The camera keeps pulling out until that giant is the size of domino 1 in the opening composition, which closes the loop.
- **Variants (rule-changing):**
  - Rising growth factor (1.2, 1.3, 1.4 and so on) until a domino is too heavy to be tipped by the previous one, decided by a computed energy criterion. Question: which domino refuses to fall.
  - Reverse chain: starts with a giant, each next domino is 1.5 times smaller and faster, the camera zooms in toward a grain. Question: how small before it stops.
  - Branching top view: every domino knocks over two equal ones in a binary tree. Question: how many rounds to cover the floor (exponential count instead of exponential size).
- **Audio:** Each impact: sine thump dropping one whole tone per domino (about 1.2 kHz down to under 100 Hz across the chain) with a noise click that gets longer and duller. Sub-bass rumble grows with size. Final hit: 40 Hz sine with long decay under a brown-noise wash. Counter: soft click.
- **Risk:** Based on a real demonstration (each domino can topple one about 1.5 times larger, attributed to L. Whitehead, 1983; citation from memory, verify before publishing it). Our rigid-block model is simplified: say so, and show a time-speed indicator because large dominoes fall far slower in real time. The tower must be a generic outline, not a real landmark. The outcome is fixed by arithmetic, so this is a one-off plus a few rule variants, not a series. Saturation of simulated versions was not checked (unknown).

## 12. RIVER DRIFT  (complexity: medium)

- **Hook:** Which float wins the river?
- **Rules:** Four floats start side by side at the top of a river flowing downward. The water is one velocity field: steady current plus ten slowly drifting whirlpools and three rocks. Floats follow the local flow with slight inertia and bump each other. No float is steered. First across the bottom line wins.
- **Tension:** Thousands of tracer streaks make the current visible, so viewers can see a whirlpool ahead of the leader and anticipate the trap. A float caught in an eddy circles while the others pass, then is released as the whirlpool drifts, so the order reshuffles repeatedly. A rank strip and a progress bar keep the standings readable while the camera scrolls with the pack. The final stretch is laid out with one whirlpool near the line. Seed-search targets (heuristic): at least 5 lead changes, winner not leading at 75 percent distance, winning margin under 0.4 s, no float trapped for more than 6 s.
- **Payoff:** A photo-finish freeze with the gaps in hundredths of a second, then the winner's full path lights up along the river as a single line showing where it avoided or escaped each whirlpool.
- **Variants (rule-changing):**
  - Last place sinks: every 6 seconds the float in last position is removed, turning the race into survival.
  - Upstream: floats have constant self-propulsion against the current, so the back-flow side of each whirlpool becomes a boost lane and the slow-looking route can win.
  - Fork: the river splits around an island into a short slow branch and a long fast branch; the flow alone decides which branch each float takes.
- **Audio:** A filtered-noise water bed whose cutoff follows the leader's speed; each float has a soft sine voice, and an overtake triggers a two-note blip in the overtaker's pitch; capture by a whirlpool is a wobbling descending tone made with an LFO; the finish is a short chord.
- **Risk:** It can look like a marble or country race if the floats are round balls or carry flags, and that format is saturated: use abstract leaf or arrow shapes, four plain colours, a dark data-visualisation look. Start lane affects the result, so rotate lanes between episodes and show the seed. A float can be trapped indefinitely in a steady eddy; drifting whirlpools and seed rejection are required. A paper-boat styling would read as child-directed. Saturation of flow-field races is unknown (not searched).

## 13. STAR FORGE  (complexity: medium)

- **Hook:** Which core becomes a star?
- **Rules:** Three coloured cores sit in a rotating disc of 600 dust grains. Gravity is softened inverse-square, all pairs. Touching bodies merge, conserving mass and momentum; mixed colour goes to the heavier body. Bodies change form at mass tiers: rock, moon, planet, giant. The first core to reach mass 250 ignites and blows away what is left.
- **Tension:** Three mass bars fill unevenly. A core can swallow a rival's largest clump, or be slung outward by a close pass and starve. Each tier change is a small visual reward every few seconds. Late on, two giants spiral toward each other and a single merger can decide the result.
- **Payoff:** Ignition: the winner flashes into a glowing star with radial rays and a ring shockwave pushes the remaining dust outward. The dust drifts back into a rotating disc with three small cores, which is frame one.
- **Variants (rule-changing):**
  - No cores: plain dust, question becomes how many bodies survive in orbit after a fixed time, with a final count.
  - A fixed heavy central star swallows anything that falls in: question becomes whether any planet survives.
  - Impacts above a speed threshold shatter both bodies into fragments instead of merging, so growth can reverse.
- **Audio:** Each merge: soft sine pluck, pitch falling with merged mass, panned by x position. Tier change: two-note rising interval. Low sine pad whose loudness tracks the leading core's mass. Ignition: one-second rising noise sweep into a major chord of pure sines with slow decay.
- **Risk:** A near-identical premise already exists from a small channel (a Short titled about 256 specks of dust growing into a star, about 1.7K views when checked), so the rival-cores rule and tier forms must carry the difference; do not reuse that title. The high-view space clips in the same search looked like 3D sandbox footage, so demand for a 2D version is unproven. Label it a toy model: real stars do not form by pairwise merging and the mass numbers are arbitrary. Tier icons must be our own drawings.

## 14. UNSHATTER  (complexity: medium)

- **Hook:** What was it before it broke?
- **Rules:** Draw one of our own silhouettes as a polygon, fracture it into 40 to 70 Voronoi shards around an impact point, simulate shards as rigid pieces under gravity with spin and damped floor bounces until they rest. Play that recording backwards: shards rise and lock together. Then play it forwards: the hammer hits, it shatters again.
- **Tension:** It opens on a heap of glinting shards. They twitch, lift and the outline becomes guessable piece by piece. The reverse starts slow and accelerates, so the guessing window is roughly the first 12 seconds. A few shards carry telling details such as a handle or a hole and rise at different times.
- **Payoff:** The last shard seats, seams vanish and the object gleams whole for one beat. Then the hammer strikes and it shatters forward in real time, coming to rest exactly in the opening heap: the loop is exact because both halves are the same recording.
- **Variants (rule-changing):**
  - Shards of two different objects are mixed in one heap and only separate by colour as they assemble: question becomes which two things these are.
  - Hinged dissection with no gravity: the same pieces rotate about hinges between two shapes (the classic triangle-to-square dissection). Question: can a triangle become a square.
  - Forward-only counting rule: the object is struck at three marked points in sequence and cracks spread by a branching rule. Question: how many pieces, with a live counter.
- **Audio:** Each shard gets a short sine ping with fast decay at a random pitch between 2 and 6 kHz. During the rebuild the shatter track is rendered reversed so pings swell into silence. Assembly: one clean sine tone with slow tremolo. Shatter: high-passed noise burst plus all pings forward.
- **Risk:** Silhouettes must be generic objects we draw ourselves (bulb, key, vase), never a logo, character or recognisable product shape; toys or animals would push toward a child-directed look. State that the rebuild is a reversed recording. Shatter clips are the fastest-growing item in the tracker, so titles and look must not imitate that channel. Skipping shard-to-shard collisions can show as overlaps: QA for it. Saturation of reverse-shatter guessing was not checked (unknown).

## 15. BRIDGE LOAD TEST  (complexity: high)

- **Hook:** Which bridge breaks last?
- **Rules:** Four truss bridges span equal gaps, each built from the same total beam length. Beams are springs that snap above 6% strain. A weight hangs from each mid-span and all four grow together, one step per second. A bridge is out when its deck touches the water. Last bridge standing wins.
- **Tension:** Every beam is coloured by strain from cool to hot, so viewers watch red spread through each design and guess where it will give. Single beams snap while the bridge still holds, the load redistributes, the deck sags and a new beam turns red. A shared weight counter climbs once per second and a percent-of-limit bar per bridge reorders as loads shift. Design-search targets (heuristic): first collapse after 8 s, at least one bridge loses a beam and survives two more load steps, the last two bridges fail within 2 load steps of each other.
- **Payoff:** A half-second slow motion on the final snap, then the survivor stands alone with its held weight shown as a record number and its critical beam outlined. The best episodes are those where the sturdiest-looking design goes first, found by search rather than forced.
- **Variants (rule-changing):**
  - Rolling load: one heavy ball rolls across all four bridges in turn and gets heavier each lap, so stress travels along the deck as a wave.
  - Damage tolerance: load stays fixed and every 3 seconds the most-stressed beam is removed from every bridge; the design that survives the most removals wins.
  - Resonance sweep: no weight; the supports shake with a frequency rising from low to high, and each design fails near its own resonance.
- **Audio:** A low drone whose pitch rises with the load; creaks made from short filtered-noise bursts whose rate follows peak strain; a beam snap is a sharp click plus a plucked-string twang (Karplus-Strong); collapse is a rumble and a noise splash; the survivor gets one clean sustained tone.
- **Risk:** Several which-bridge-breaks-first Shorts were seen on 2026-11-16 at about 0.5K to 2K views and one at about 33K, so this needs the equal-material rule and strain colours to stand apart. A spring-mass truss is a simplification and must not be presented as engineering guidance; no real bridge names. Designs can look rigged unless the fixed rules (same beam length, same snap strain) are shown. It is adjacent to the saturated hydraulic-press idea (rising force until failure), so avoid press-like framing. Tuning for stable simulation and close finishes is the main cost. Not child-directed.

## 16. BREAKING POINT  (complexity: high)

- **Hook:** Which beam snaps first?
- **Rules:** Side-view truss bridge of point masses joined by springs (Verlet integration). Four beams are marked A, B, C, D. A weight on the deck grows steadily. Each spring is coloured by strain, blue to red, and breaks when strain exceeds 4%. Broken beams shed load to neighbours, which can cascade. The first marked beam to break is the answer.
- **Tension:** Strain colours creep from blue through yellow to red at different rates in different beams while a weight counter climbs; the viewer can reason about which member carries most. After the first snap the load redistributes visibly: neighbouring beams jump in colour, and the question shifts to how far the failure will run.
- **Payoff:** Cascade collapse: beams snap in sequence, the deck folds and falls with the weight. Debris leaves the frame and a clean blueprint of the bridge redraws itself in about 1.5 s, matching frame one.
- **Variants (rule-changing):**
  - Resonance: no weight; the ground shakes at a slowly rising frequency under three towers of different heights, each with its own natural frequency. Question: which tower falls first.
  - Moving load: a heavy roller crosses the deck at constant speed. Question: does it make it across.
  - One beam is visibly doubled in strength: question becomes whether the strong beam saves the bridge, showing the load path moving to the next weakest member.
- **Audio:** Tension drone from two detuned sines whose pitch follows maximum strain. Creaks: short downward-chirped saw grains when any spring passes 3% strain. Snap: click plus bright sine ping pitched by beam length. Collapse: cluster of pings over brown-noise rumble. Rebuild: quick ascending sine arpeggio.
- **Risk:** Collapsing bridges can read as disaster imagery: keep it abstract, no vehicles, people or real landmarks, never name a real bridge. A mass-spring model is a simplification: call it a spring model, not engineering advice. Explicit integration can blow up numerically and look fake: QA should check energy stays bounded. Avoid the visual style of bridge-building games. Saturation was not checked (unknown).

## 17. STRIKE  (complexity: medium)

- **Hook:** Which tower gets struck?
- **Rules:** Dark sky above five coloured towers of different heights and spacing. A leader starts at a random cloud point and grows in steps on a grid: each step picks a frontier cell with probability proportional to local electric potential squared (dielectric breakdown model, potential relaxed by Jacobi iterations). Branches fork. First branch touching a tower triggers the return stroke.
- **Tension:** Faint branches creep down in jerky steps and several tips compete. The tallest tower is the obvious pick, but an off-centre start can favour a shorter one. As the leader nears the ground, short upward streamers appear on each tower tip, lengthening with local field strength, which shows who is still in contention. In the last 2 seconds two tips are within a few cells of two different towers.
- **Payoff:** Return stroke at full brightness along the winning channel only, one soft white flash, the struck tower glows in its colour with a ring shockwave. Afterglow fades to the dark-sky first frame for the loop.
- **Variants (rule-changing):**
  - Add a visible grounded rod or overhead wire: question becomes whether the rod protects the tower beside it (shielding rule).
  - Radial version: discharge grows from the centre of a disc toward coloured electrodes on the rim, with an insulating obstacle in the way (Lichtenberg-figure geometry).
  - Inverted rule: upward leaders start from every tower toward a charged cloud band; first tower to reach the cloud wins.
- **Audio:** Low filtered-noise wind bed. Each growth step: very quiet high click, click rate rising as the leader descends. Strike: 30 ms full-band noise crack, then thunder from brown noise through a closing low-pass plus a 40 to 60 Hz sine rumble decaying over 2 s.
- **Risk:** Flash must respect photosensitivity guidance: one flash, softened to about 80% white, never repeated rapidly. It is a simplified stochastic model: say so, and do not claim real lightning always hits the tallest object. Changing only the tower layout would look templated. One comparable Short was found (3D, about 12K views); not saturated on that single query (heuristic). Per-step potential relaxation is the CPU cost: keep the grid near 108 by 192 and relax incrementally.
