# Phased plan

Develop on a **5-10 min clip** first; scale to a full match only once the spine works.

## MVP - the spine
- [x] **Step 0 - env & test clip.** Env + deps built; `data/raw/clip.mp4` (Belgium v Senegal highlights, 720p) via download.py.
- [~] **Step 1 - scoreboard OCR.** Read score; score change => candidate goal.
      - [x] `frames.py` to dump frames for calibration
      - [x] `SCOREBOARD_BOX` calibrated (exclude the clock - its colon misreads and leaks digits)
      - [x] OCR reads scores cleanly on visible frames (0-2, 2-2, 3-2 on the test clip)
      - [ ] confirm chronological goal timestamps on a CONTINUOUS match (highlights clip is non-chronological)
- [x] **Step 2 - audio spikes.** ffmpeg -> per-second RMS(dB) -> spikes vs local rolling-median
      baseline with robust (median/MAD) threshold. Global stats fail (silence + loudness
      compression); local baseline is the fix. 7 candidates on the test clip.
- [x] **Step 3 - fuse.** Cluster OCR + audio timestamps within 10s into candidate events;
      both-sources = high confidence. Writes data/events.json for Step 4. 6 candidates on
      the test clip, incl. one high-confidence goal (score change + crowd surge at 03:22).
- [x] **Step 4 - caption windows.** BLIP-base (now GPU) captions frames per event. Captions are
      generic ("a soccer game...") - superseded by commentary ASR for the recap; kept for later.
- [x] **Step 5 - LLM recap.** qwen3:8b via Ollama (think=false, num_ctx=16384). **MVP spine done.**

## v2 upgrade - commentary ASR (done)
- [x] **GPU:** torch cu128 for the RTX 5060 (Blackwell); OCR + BLIP + Whisper all run on GPU.
- [x] **Whisper ASR (asr.py):** transcribe full commentary -> data/transcript.json (447 segments).
- [x] **OCR robustness:** score changes need a 2-read confirmation (kills 1-frame misreads like
      "17-1"); OCR sampling dropped to 0.5fps.
- [x] **Recap v2:** OCR score changes DEFINE the goals; commentary only names the scorers. Fixes
      the hallucination (LLM had invented a 3rd goal / wrong 2-1). Now "Morocco 1-1 Norway" with
      correct scorers (Brahim Diaz, Oscar Bob).

## v3 - generic scoreboard + board time + analytic recap (done)
- [x] **Generic top-left reader:** OCR the whole top-left corner, anchor on the MM:SS clock,
      read the score off the same bar row, skip letter tokens. No per-broadcast box - verified
      on BOTH SuperSport (BEL 2 2 SEN, no dash) and ESPN (MAR 1-0 NOR, dashed). Rejects the
      "ROAD to 26" banner.
- [x] **Board time:** each goal carries its match-clock time (7:39, 75:03), used in the recap.
- [x] **Recap v3:** ## Summary (5-8 sentence analytic paragraph with commentator trends -
      pressing, defensive shape, second-half shift) + ## Goals with board time.

## v4 - generic reader, frontend, one-shot pipeline (done)
- [x] **Clockless / absent scoreboards:** anchor on team codes (MUN..EVE), read the score between
      them; mag_ratio=2 recovers small single digits. Handles clocked, clockless, and no board.
- [x] **Seek-based OCR sampling:** decode only sampled frames (Barca fusion ~90s, was minutes).
- [x] **Whisper medium** default (better names); **commentary-primary recap** with title final-score
      anchor + explicit goal split.
- [x] **One-shot pipeline** (src/pipeline.py): URL or file -> download -> asr -> fuse -> recap.
- [x] **Web frontend** (app.py): paste link / upload video, duration-based ETA, live progress,
      rendered recap. Per-video outputs under data/out/<name>/.

## v5 - stacked scoreboards, teams, non-English (done)
- [x] **Stacked scoreboards:** read_scoreboard handles both inline (MUN 1-1 EVE) and stacked
      (GIR 0 / OVI 2, teams on separate rows) layouts. Validated on LaLiga Girona 3-3 Oviedo.
- [x] **Alphanumeric team codes:** CODE_RE allows digits (B04, S04) so Bundesliga boards read.
- [~] **Highlights vs continuous:** montages give non-monotonic OCR (0-0 -> 1-1 -> 2-2), so per-goal
      board times can't be mapped and scorer attribution suffers. Pipeline is built for continuous
      footage - the persistent limitation flagged since day one.
- [x] **Board time on every goal:** carry the most recent clock forward for changes whose exact
      frame missed the clock read.
- [x] **Team names:** URL title / upload filename passed to the recap -> real names (Girona FC,
      Real Oviedo), never "Team 1/2"; falls back to commentary when the title has none.
- [x] **Non-English commentary:** asr auto-detects language (was forced English -> "Diola");
      recap translates to English. Spanish LaLiga now yields plausible Spanish names.
- [x] **Two-pass score:** recap asks the model the final score first, so the count is pinned
      without any manual input (uploads no longer anchor to garbage OCR).

## Polish / next
- [x] Goal-count/names: root cause is TRANSCRIPT quality, not the recap model. Man Utd with the
      medium transcript -> correct 4-4 (8 goals), real scorers (Jelavic, Pienaar, Welbeck, Rooney).
      qwen3:8b/gemma3:12b/gpt-oss:20b all undercounted equally on the small transcript.
- [x] deno installed so yt-dlp solves YouTube JS challenges (fixes 403s); download.py has retries.
- [x] Default recap model gemma3:12b (~60s); qwen3:8b (~25s) via --model for speed.
- [ ] Ollama must be running for the recap step - have the pipeline auto-start it.
- [ ] Console print shows mojibake for curly quotes (recap.md file is fine, utf-8) - force ASCII.
- [ ] Validate on a truly continuous full match.
- [ ] BLIP LoRA for football vocabulary (the training flex).

## After the MVP runs end-to-end
- [ ] Highlight clip cutting (ffmpeg around high-signal moments).
- [ ] YOLO on ball/players for goal-mouth activity.
- [ ] BLIP LoRA for football vocabulary (hand-label a few hundred frames).

## Watch-outs
- Generic captioners struggle with broadcast football (small, fast players) - OCR + audio
  carry more load early; that's fine.
- GPU torch for BLIP needs a CUDA build for the RTX 5060 (Blackwell, sm_120): cu128.
  Step 1 OCR runs on CPU torch, so defer the CUDA install to Step 4.
- Ollama already has qwen3:8b / llama3 / gemma3 pulled - Step 5 needs no downloads.
