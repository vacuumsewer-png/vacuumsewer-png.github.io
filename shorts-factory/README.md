# shorts-factory

Code-only YouTube Shorts: every frame is drawn with pycairo, every sound is synthesised with numpy,
ffmpeg encodes. No footage, no samples, no third-party material except one open-licensed font.

Slice 1 contains one scene, `territory` (four balls paint a tiled arena, most tiles wins).

## Setup

Needs Python 3.11+, `ffmpeg`/`ffprobe` on the PATH, and the cairo development headers for pycairo
(`apt-get install -y ffmpeg libcairo2-dev pkg-config` on Debian/Ubuntu).

    pip install -r requirements.txt

## Make a video

    python make.py territory --out out/                 # score 600 seeds for drama, render the best
    python make.py territory --seed 197 --out out/      # render one specific seed
    python make.py territory --seeds 2000 --out out/    # search wider

Writes `out/territory_seed<N>.mp4` (1080x1920, 30 fps, H.264 High + AAC 192k, about -14 LUFS) and
`out/territory_seed<N>.json` (title, description, tags, disclosure flags, seed, drama stats).
The simulation is deterministic: the same seed always gives the same video.

Quick look without encoding (PNG stills at the given seconds):

    python make.py territory --seed 197 --stills 0,0.5,12,25.5 --out /tmp/stills

## Check a video

    python qa.py out/territory_seed197.mp4
    python qa.py out/territory_seed197.mp4 --frames 0,0.5,4,9,14,19,23.5,24.5,25.5,26.8,27.97 --frames-dir /tmp/review

`qa.py` exits non-zero if any check fails: ffprobe conformance, length, black frames, loudness and
true peak (ebur128), long silences, motion on the first frame, and how cleanly the last frame cuts
back to the first. `--frames` extracts stills to look at. Nothing here can judge how the audio
sounds; listen before publishing.

## Layout

    engine/gfx.py        easing, colour, shapes, text drawn from the TTF outlines (fontTools)
    engine/audio.py      synth voices, drums, mixer, reverb, EQ, limiter, loudness mastering
    engine/pipeline.py   parallel frame rendering piped into ffmpeg
    scenes/territory.py  simulation, drama score, drawing, sound and metadata for this scene
    make.py              CLI: seed search and render
    qa.py                conformance and sanity checks, frame grabs
    fonts/               Anton (SIL Open Font License 1.1) and its licence text

## Font

`fonts/Anton-Regular.ttf` is Anton by The Anton Project Authors, SIL OFL 1.1 (`fonts/Anton-OFL.txt`).
It is the Latin subset from the npm package `@fontsource/anton` 5.3.0, converted from woff2:

    npm install @fontsource/anton
    python -c "from fontTools.ttLib import TTFont; f = TTFont('node_modules/@fontsource/anton/files/anton-latin-400-normal.woff2'); f.flavor = None; f.save('fonts/Anton-Regular.ttf')"
