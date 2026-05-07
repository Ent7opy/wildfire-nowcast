# Dashboard accessibility audit — 2026-05-07

Scope: every authed page in the A' dashboard plus the public share view. Out
of scope: Clerk-hosted `/sign-in`, `/sign-up`. Method: JSX read + mental
keyboard / screen-reader simulation. No real browser, no axe-core run.

## What shipped in this PR

Low-risk markup-only fixes. None changed visual design. All under 50 LOC.

| File | Fix | Why |
|---|---|---|
| `app/layout.tsx:13` | `role="status"` on auth-config banner | Banner content (read-only mode) wasn't announced to AT users on first paint. |
| `app/dashboard/layout.tsx:19` | `aria-label="Primary"` on `<nav>` | Disambiguates from any future secondary nav for screen-reader landmark navigation. |
| `app/dashboard/_components/aoi-list.tsx:30-35` | `scope="col"` on table headers | Required for screen-reader table navigation; AT cannot infer column association without it. |
| `app/dashboard/aoi/new/page.tsx:78` | `role="tablist"` + `role="tab"` + `aria-selected` on input-method tabs | Buttons styled as tabs but lacked the ARIA pattern; screen readers announced "button" three times instead of "tab 1 of 3, selected". |
| `app/dashboard/aoi/new/page.tsx` (form error) | `role="alert"` on the error `<p>` | Validation failure is dynamic; without an alert role, AT users get no announcement and silently believe the submit succeeded. |
| `app/dashboard/_components/share-toggle.tsx` | `aria-label` on the share URL anchor (`target="_blank"`) | Communicates "opens in new tab" verbally — `target="_blank"` alone has no AT signal. Also `role="alert"` on error message. |
| `app/dashboard/_components/rules-form.tsx` (save msg) | `role="status"` + `aria-live="polite"` on the save-result span | Save confirmation/failure is now announced. |
| `app/dashboard/_components/aoi-map.tsx` (container div) | `role="img"` + descriptive `aria-label` | MapLibre canvas is unlabelled by default; AT users heard nothing. Now they hear "Map of the area of interest with matched FIRMS detections" or the draw-mode equivalent. |

## Findings deferred (need design or larger refactor)

### F1 — Map drawing is mouse-required (real gap)

`installDrawHandlers` in `app/dashboard/_components/aoi-map.tsx` binds vertex
addition to `click` and ring-close to `dblclick`. Keyboard-only users
cannot create an AOI via the **Draw on map** tab. The Paste and Upload tabs
are accessible alternatives, so this is not a blocker — but the Draw tab
should either:

- Have a visible note ("Mouse required — use Paste or Upload for keyboard
  input") near the tab itself, or
- Be hidden from the tablist when no pointer device is detected, or
- Get a parallel keyboard interaction (e.g. arrow keys to nudge a virtual
  cursor, Enter to add vertex, Esc to finish). This would be ~60-100 LOC
  in the map component and probably needs visible UI cues, so it's a
  proper feature, not a chore.

Recommendation: add the visible note this week (cheap, ~5 LOC) and put the
keyboard interaction in the v1.2 backlog. Not blocking launch.

### F2 — Color contrast of `--warn` token

`app/globals.css:8` defines `--warn: oklch(0.85 0.13 80)` (a light gold).
The freshness banner (`freshness-banner.tsx:88`) uses
`bg-[color:var(--warn)]/10` with `text-[color:var(--foreground)]` — the
foreground token is near-black on near-white, so contrast against the
**page** background is fine. But the **border** (`border-[color:var(--warn)]`)
at full opacity against `--background: #faf8f4` is roughly 1.3:1 — well
below WCAG AA's 3:1 minimum for non-text UI components (SC 1.4.11). In
dark mode (`oklch(0.7 0.13 80)` border on `#12100d` background) the
contrast is fine.

Recommendation: bump light-mode `--warn` darker, e.g.
`oklch(0.65 0.15 60)`, and re-verify visually. Out of scope for a markup
PR — needs a design eyeball.

### F3 — Status colors `text-yellow-700` / `text-green-700`

`aoi-list.tsx:54-56` — the only signal of "active" vs "paused" is text
color and the literal word. Text is already there, so AT users get the
information. Sighted color-blind users get the word too. Not a real gap;
keeping as-is.

### F4 — Brief view markdown

`app/dashboard/brief/[id]/page.tsx` and `app/brief/share/[token]/page.tsx`
both use `dangerouslySetInnerHTML` to inject AI-generated markdown via
`renderMarkdownToHtml`. The heading hierarchy depends entirely on what the
LLM emits. Spec says briefs have an h1 ("Brief: <AOI name>"), so on the
share page that becomes the document's h1 — good. On the authed page the
dashboard layout has no `<h1>`, so the brief's h1 is the page h1 — also
good. But this is **not enforced** anywhere. If the prompt drifts, headings
break.

Requires manual verification: render real briefs and inspect the heading
tree. Recommend adding a unit test on `renderMarkdownToHtml` that asserts
the rendered HTML starts with `<h1>` for the standard prompt fixtures.
~15 LOC; not blocking.

### F5 — Public share page `<main>` is the only landmark

`app/brief/share/[token]/page.tsx` has `<main>` + `<article>` + `<footer>`.
No `<header>` and no skip link. For a single-purpose shared brief view this
is fine — there's nothing to skip past. Leaving alone.

### F6 — `dangerouslySetInnerHTML` and inline interactive content

The brief markdown could in principle contain links. `marked`/whichever
renderer we use should be sanitising output. Worth a security AND a11y
read of `lib/notify/markdown.ts` — links inside the brief should pick up
`rel="noreferrer"` automatically if external. Not audited here; flagging
for a follow-up scout pass.

## Manual-verification needed (cannot audit without browser)

- Tab order through `RulesForm` (especially around the conditional
  `quietHours` panel — when the checkbox is toggled, where does focus
  land?)
- Focus management in `ShareToggle` after mint: does focus move to the
  newly-rendered URL link, or stay on the now-disabled "Create public
  link" button?
- MapLibre keyboard interactions in view mode (pan via arrow keys?) — not
  audited; MapLibre defaults are usually OK but unverified here.
- Color contrast measurements on the deployed Vercel preview, with both
  light and dark color schemes, using a real picker.

## Summary

11 markup-only fixes shipped in this branch, no visual change, no behavior
change. 6 findings deferred — F1 (keyboard drawing) and F2 (warn-token
contrast in light mode) are the two worth queueing. Everything else is
either non-blocking or requires actual screen-reader / browser
verification.
