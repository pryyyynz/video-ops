"""Step 5: match recap from commentary transcript + detected score changes.

Commentary (Whisper) is the PRIMARY source for the goals and narrative. OCR score
changes are only weak corroboration - they can be partial, empty, or plain wrong on
unusual scoreboards, so they are NEVER used to anchor the final score. A trusted final
score (from the title/filename via --final) anchors the result and pins the goal count;
without one, the model works the score out from the commentary. Reads
data/out/<video>/{events,transcript}.json, writes .../recap.md.

Usage:
    python src/recap.py data/raw/manutd_everton.mp4 --final 4-4
    python src/recap.py data/raw/manutd_everton.mp4          # infer score from commentary
"""
import argparse
import json
import re

import config
import llm

# gemma3:12b - richer and better at instruction-following than the 8B models
# (~60s/recap; partial CPU offload on 8GB). Transcript quality still drives the
# goal-count and name accuracy more than the recap model does.
MODEL = "gemma3:12b"
NUM_CTX = 16384  # must hold the whole transcript; Ollama's 4096 default truncates it

PROMPT = """You are a football analyst. Write the recap in ENGLISH from the evidence below.
The commentary may be in another language; translate names and facts as needed.

{title_line}{final_line}
{split}

TEAMS: name the two teams from the title and commentary, and use their real names
throughout - never "Team 1" / "Team 2".

SCOREBOARD READINGS (OCR - score and match-clock time; may be PARTIAL, empty, or WRONG,
so treat as hints and never above the commentary):
{scores}

COMMENTARY (MM:SS video time) - the PRIMARY and most reliable source. Identify every
goal and its scorer from here. Names may be misheard; fix obvious errors from context.

{commentary}

Write markdown, plain text, no emoji, in two sections:

## Summary
5-8 sentences using the real team names: the final result and score, how the match
unfolded, and the general trends the commentators point to (defensive shape, attacking
form, momentum, standout or off-colour players). Ground every claim in the commentary.

## Goals
One bullet per goal, in order: "[time] Player Name (Team) - one-line description". For the
time, use the MATCH-CLOCK time from the scoreboard readings when available (e.g. 62'),
otherwise the commentary's time cues. List every goal the commentary describes; the goals
must tally to the final score. If a scorer is not identifiable, use "Unknown".
"""


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def summarize_goals(events):
    changes = sorted(
        (s["t"], s["detail"], s.get("board")) for e in events for s in e["signals"]
        if s["source"] == "score_change"
    )
    goals, final = [], "0-0"
    for t, detail, board in changes:
        old, new = (p.strip() for p in detail.split("->"))
        side = "home" if int(new.split("-")[0]) > int(old.split("-")[0]) else "away"
        goals.append((t, new, side, board))
        final = new
    return goals, final


def format_scores(goals):
    if not goals:
        return "(none - scoreboard unreadable or absent in this video)"
    return "\n".join(
        f"- {new} at {board or 'video ' + mmss(t)} (the {side} team scored)"
        for t, new, side, board in goals
    )


def format_commentary(segments):
    return "\n".join(f"[{mmss(s['start'])}] {s['text']}" for s in segments if s["text"])


def ask_ollama(prompt, model=MODEL):
    return llm.ask(prompt, model, num_ctx=NUM_CTX)


FINAL_PROMPT = """Read the football commentary below and work out the final score.
Count every goal each team scored across the whole match. Reply with ONLY the final
score as two numbers "home-away" (home team first), e.g. 4-4. No other text.

COMMENTARY:
{commentary}
"""


def determine_final(commentary, model=MODEL):
    """First pass: ask the model for just the final score, so the recap pass can pin
    the exact goal count without any manual input."""
    m = re.search(r"(\d{1,2})\s*-\s*(\d{1,2})",
                  ask_ollama(FINAL_PROMPT.format(commentary=commentary), model))
    if m and int(m.group(1)) <= 20 and int(m.group(2)) <= 20:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def run(video, model=MODEL, final=None, title=None):
    d = config.out_dir(video)
    events = json.loads((d / "events.json").read_text())["events"]
    transcript = json.loads((d / "transcript.json").read_text(encoding="utf-8"))["segments"]
    goals, _ = summarize_goals(events)   # OCR final is deliberately NOT used to anchor
    commentary = format_commentary(transcript)

    if not final:
        final = determine_final(commentary, model)   # work the score out from the commentary

    split = ""
    if final:
        try:
            h, a = (int(x) for x in final.split("-"))
            final_line = f"KNOWN FINAL SCORE: {final} - authoritative; the recap must reflect exactly this."
            split = (f"That is EXACTLY {h + a} goals in total ({h} for the home/first team, {a} for "
                     f"the away/second team). You MUST list all {h + a} goals - go through the commentary "
                     f"in order and include every one, even when the same player scores more than once. "
                     f"Do not list saves, chances, or disallowed goals.")
        except ValueError:
            final = None
    if not final:
        final_line = ("FINAL SCORE: not provided - work it out yourself by counting the goals the "
                      "commentary describes. The scoreboard readings may be wrong, so rely on the commentary.")

    title_line = f"VIDEO TITLE: {title}\n" if title else ""
    prompt = PROMPT.format(
        title_line=title_line, final_line=final_line, split=split,
        scores=format_scores(goals), commentary=commentary,
    )
    print(f"Asking {model} (final {final or 'unknown'}, {len(goals)} OCR goals, {len(transcript)} segments)...\n")
    recap = ask_ollama(prompt, model)
    out = d / "recap.md"
    out.write_text(recap, encoding="utf-8")   # save first: a console encoding error must not lose it
    try:
        print(recap)
    except UnicodeEncodeError:
        print(recap.encode("ascii", "replace").decode())
    print(f"\n\nSaved to {out}")
    return recap


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--final", help="known final score like 3-3 (from the title); anchors the recap")
    p.add_argument("--title", help="video title / filename, used to name the teams")
    a = p.parse_args()
    run(a.video, a.model, a.final, a.title)
