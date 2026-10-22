"""The look of the chat window (design-references/FINAL-DESIGN-INSTRUCTIONS.md): colors, the window, the header, the
greeting area, the message bubbles and the input box. Only the design - what the bot says and decides stays in answer.py.

The page is Streamlit's own, restyled by CSS: the window is the main block (rounded corners, soft shadow), the input
box is st.chat_input with the send button moved outside the pill, and the messages are HTML bubbles built here.
Images: static/ (cut from the reference by design-references/make_assets.py) - the big robot is served by Streamlit
static serving (app/static/robot.png), the small icon and the send icon are embedded in the CSS (they repeat).
"""
import base64
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
ROBOT_URL = "app/static/robot.png"  # relative: also right when the app sits under another path (Streamlit Cloud)

NAVY, ORANGE, BG, GREY = "#111C3F", "#FF5A23", "#F7F8FB", "#8A8FA0"  # the only colors (white aside)
WINDOW = 476  # px - the width at which the sizes of the instructions (16px text, 64px header, ...) fit the reference
RADIUS = 28
# app.py טוען מחדש את המודול הזה כשהגרסה בזיכרון ישנה יותר (Streamlit Cloud לא טוען מחדש מודולים אחרי git pull)
UI_VERSION = 3


def _icon_uri() -> str:
    return "data:image/png;base64," + base64.b64encode((STATIC / "robot-icon-small.png").read_bytes()).decode()


def _send_uri() -> str:
    return "data:image/svg+xml," + quote((STATIC / "send-icon.svg").read_text(encoding="utf-8").strip())


def css() -> str:
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;600;700&display=swap');
:root {{ --navy:{NAVY}; --orange:{ORANGE}; --bg:{BG}; --grey:{GREY}; }}
html, body, .stApp, .stAppViewContainer {{ background:#fff !important; }}
.stApp {{ direction:rtl; text-align:right; color:var(--navy); }}
.stApp, .stApp textarea, .stApp button, .px-row, .px-hero, .px-header {{ font-family:'Heebo',sans-serif; }}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], footer {{ display:none !important; }}

/* the window: the main block on top, the input row under it - one rounded window with a soft shadow */
.stMain {{ max-width:{WINDOW}px; width:100%; margin:24px auto 0 !important; background:var(--bg);
  border-radius:{RADIUS}px {RADIUS}px 0 0; box-shadow:0 14px 44px rgba(17,28,63,.16); clip-path:inset(-80px -80px 0 -80px);
  overflow-x:hidden; }}
.stMainBlockContainer {{ max-width:none !important; width:100%; padding:0 !important; }}
.stMainBlockContainer [data-testid="stVerticalBlock"] {{ gap:0; }}
.stMainBlockContainer [data-testid="stMarkdownContainer"] {{ margin-bottom:0 !important; }}
.stBottom, .stBottom > div {{ background:transparent !important; }}
[data-testid="stBottomBlockContainer"] {{ max-width:{WINDOW}px !important; width:100%; margin:0 auto 24px; background:var(--bg);
  padding:12px 12px 12px !important; border-radius:0 0 {RADIUS}px {RADIUS}px; box-shadow:0 14px 44px rgba(17,28,63,.16);
  clip-path:inset(0 -80px -80px -80px); }}
@media (max-width:{WINDOW + 40}px) {{
  .stMain {{ margin:0 !important; border-radius:0; box-shadow:none; }}
  [data-testid="stBottomBlockContainer"] {{ margin:0; border-radius:0; box-shadow:none; }}
  .px-header {{ border-radius:0 !important; }}
  .px-user .px-bubble, .px-bot .px-bubble {{ max-width:78%; }}
}}

/* the header: sticks to the top of the window (the sticky element is its container: the parent of a sticky element is its limit) */
.stMainBlockContainer [data-testid="stElementContainer"]:has(.px-header) {{ position:sticky; top:0; z-index:30; }}
.px-header {{ height:64px; box-sizing:border-box; background:var(--navy);
  border-radius:{RADIUS}px {RADIUS}px 0 0; display:flex; align-items:center; gap:12px; padding:0 20px; }}
.px-icon {{ width:40px; height:40px; border-radius:50%; flex:none; background:#fff url("{_icon_uri()}") center/cover; }}
.px-header-title {{ color:#fff; font-weight:700; font-size:18px; line-height:1; }}
.px-new {{ margin-inline-start:auto; color:#fff !important; text-decoration:none !important; font-size:13px; font-weight:500;
  line-height:1.2; white-space:nowrap; border:1.5px solid rgba(255,255,255,.85); border-radius:999px; padding:5px 12px; }}
.px-new:hover {{ background:#fff; color:var(--navy) !important; }}
.px-close {{ display:flex; align-items:center; justify-content:center; width:32px; height:32px; padding:0; border:none;
  background:transparent; cursor:pointer; }}
.px-close:hover {{ opacity:.8; }}

/* closed (the X): the window is hidden and the owner's icon waits at the bottom left; a click shows the same window again,
   with the same conversation (nothing is reloaded). The launcher is added to <body> by script_html(). */
.px-launcher {{ display:none; position:fixed; left:20px; bottom:20px; width:76px; height:76px; padding:0; border:none;
  background:transparent; cursor:pointer; z-index:1000; filter:drop-shadow(0 6px 14px rgba(17,28,63,.25)); }}
.px-launcher img {{ width:100%; height:100%; display:block; }}
.px-launcher:hover {{ transform:scale(1.05); }}
body.px-closed .px-launcher {{ display:block; }}
body.px-closed #root {{ display:none !important; }}  /* all of Streamlit; the launcher is outside it, in <body> */

/* the greeting area: the robot, the decorations (in the image) and the card that overlaps its bottom */
.px-hero {{ display:flex; flex-direction:column; align-items:center; padding:16px 0 12px; }}
.px-robot {{ width:{round(650 * .62)}px; max-width:100%; height:auto; display:block; }}
.px-card {{ position:relative; z-index:2; margin-top:-48px; background:#fff; border-radius:24px; padding:16px 28px;
  box-shadow:0 10px 30px rgba(17,28,63,.10); text-align:center; color:var(--navy); font-weight:700; font-size:20px;
  line-height:1.4; min-width:270px; box-sizing:border-box; }}
.px-orange {{ color:var(--orange); }}

/* the messages: the customer on the right (navy), the bot on the left (white with a navy frame) */
.px-row {{ display:flex; gap:8px; padding:0 14px; margin-bottom:16px; }}
.px-user {{ justify-content:flex-start; align-items:flex-end; }}
.px-bot {{ justify-content:flex-end; align-items:flex-start; }}
.px-bubble {{ box-sizing:border-box; border-radius:22px; padding:9px 15px; font-size:16px; line-height:1.55;
  overflow-wrap:anywhere; text-align:right; }}
.px-user .px-bubble {{ background:var(--navy); color:#fff; border-bottom-right-radius:0; max-width:56%; }}
.px-bot .px-bubble {{ background:#fff; color:var(--navy); border:1.5px solid var(--navy); border-bottom-left-radius:0;
  max-width:56%; }}
.px-bubble a {{ color:inherit; text-decoration:underline; }}
.px-bubble .answer {{ line-height:1.55; }}
.px-bubble .answer bdi {{ white-space:nowrap; }}  /* a phone does not break after the dash ("073-" and "2076100") */
.px-sources {{ margin-top:8px; font-size:14px; line-height:1.5; }}
.px-checked {{ margin-top:4px; font-size:12px; line-height:1.4; color:var(--grey); }}
.px-hours {{ margin-top:6px; font-size:13px; line-height:1.4; color:var(--navy); font-weight:500; }}
.px-meta {{ direction:ltr; display:flex; align-items:center; gap:4px; font-size:12px; line-height:1; color:var(--grey);
  white-space:nowrap; padding-bottom:3px; }}
.px-side {{ display:flex; flex-direction:column; align-items:center; gap:4px; flex:none; width:56px; }}
.px-avatar {{ width:56px; height:56px; box-sizing:border-box; border-radius:50%; border:1px solid var(--navy);
  background:#fff url("{_icon_uri()}") center/cover; }}
.px-time {{ font-size:12px; line-height:1; color:var(--grey); }}
.stage-main {{ animation:stage-pulse 1.6s ease-in-out infinite; }}  /* the bot is working - not stuck */
@keyframes stage-pulse {{ 50% {{ opacity:.55; }} }}
@media (prefers-reduced-motion:reduce) {{ .stage-main {{ animation:none; }} }}

/* the buttons of the bot (insurance type, more questions, human agent): white pills with a navy frame, under the bot's bubble */
.stMain [data-testid^="stBaseButton"], .stMain [data-testid^="stBaseLinkButton"] {{ background:#fff !important; color:var(--navy) !important;
  border:1.5px solid var(--navy) !important; border-radius:999px !important; min-height:0 !important; padding:6px 16px !important;
  font-size:14px; font-weight:500; line-height:1.3; box-shadow:none !important; }}
.stMain [data-testid^="stBaseButton"] p, .stMain [data-testid^="stBaseLinkButton"] p {{ font-size:14px; margin:0; }}
.stMain .stButton, .stMain [data-testid="stLinkButton"], .stMain [data-testid^="stBaseLinkButton"] {{ width:auto !important; }}
.stMain [data-testid^="stBaseButton"]:hover, .stMain [data-testid^="stBaseLinkButton"]:hover {{ background:var(--navy) !important; color:#fff !important; }}
.stMain [data-testid^="stBaseButton"]:hover p, .stMain [data-testid^="stBaseLinkButton"]:hover p {{ color:#fff !important; }}
.stMainBlockContainer [data-testid="stElementContainer"]:has(> .stButton, > [data-testid="stLinkButton"]):not([data-testid="stHorizontalBlock"] > *) {{
  width:100%; box-sizing:border-box; padding:0 14px 16px 78px; display:flex; justify-content:flex-end; }}

.stMainBlockContainer [data-testid="stHorizontalBlock"] {{ width:100%; box-sizing:border-box; padding:0 14px 16px 78px; display:flex;
  justify-content:flex-end; }}
.stMainBlockContainer [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; gap:8px; }}
/* the example questions (before the first question): on the customer's side - the right, like her bubbles */
.stMainBlockContainer .st-key-px_examples {{ justify-content:flex-start !important; padding:0 14px 16px 78px !important; }}

/* 👍/👎 under the bot's bubble (at its left edge): small, grey, the chosen one navy. The icons are a Material Symbols font. */
.stMainBlockContainer [data-testid="stElementContainer"]:has([data-testid="stFeedback"]) {{ width:100%; box-sizing:border-box;
  padding:0 14px 12px 78px; display:flex; justify-content:flex-end; margin-top:-8px; }}
[data-testid="stFeedback"] button {{ background:transparent !important; border:none !important; box-shadow:none !important; }}
[data-testid="stFeedback"] button, [data-testid="stFeedback"] button span {{ color:var(--grey) !important; }}
[data-testid="stFeedback"] button:hover, [data-testid="stFeedback"] button:hover span,
[data-testid="stFeedback"] button[aria-checked="true"], [data-testid="stFeedback"] button[aria-checked="true"] span {{ color:var(--navy) !important; }}

/* the input row: a pill with an orange frame, and the send button outside it, on its right */
[data-testid="stChatInput"] {{ background:transparent; width:100%; }}
[data-testid="stChatInput"] > div {{ border:none !important; background:transparent !important; box-shadow:none !important;
  border-radius:0 !important; padding:0 !important; overflow:visible !important; }}
[data-testid="stChatInput"] > div > div {{ flex-direction:row-reverse; align-items:center; gap:10px; }}
[data-testid="stChatInput"] > div > div > div:first-child {{ flex:1; min-width:0; height:56px; box-sizing:border-box;
  border:2px solid var(--orange); border-radius:999px; background:#fff; padding:0 24px; align-items:center; }}
[data-testid="stChatInput"] > div > div > div:first-child > div {{ width:100%; }}
[data-testid="stChatInputTextArea"] {{ direction:rtl; text-align:right; color:var(--navy); font-size:16px; padding:0 !important;
  min-height:0 !important; line-height:24px; }}
[data-testid="stChatInputTextArea"]::placeholder {{ color:var(--grey); opacity:1; }}
[data-testid="stChatInput"] > div > div > div:last-child {{ flex:none; }}
[data-testid="stChatInputSubmitButton"] {{ width:48px; height:48px; border-radius:50%; padding:0; border:none;
  background:url("{_send_uri()}") center/48px 48px no-repeat !important; opacity:1 !important; }}
[data-testid="stChatInputSubmitButton"] svg {{ display:none; }}
/* ---- accessibility (22/10, the owner's request): nothing below changes the normal look - only the small "Aa" button that
   hangs off the header's bottom-left corner (it takes no room in the header: on a 360px phone there are 0 free pixels).
   Text size and high contrast are classes on <body> (outside Streamlit's redraws), set by a11y_script_html(). ---- */
.px-header {{ position:relative; }}
.px-a11y {{ position:absolute; left:14px; bottom:-18px; width:36px; height:36px; box-sizing:border-box; padding:0;
  border-radius:50%; border:1.5px solid var(--navy); background:#fff; color:var(--navy); font:700 14px/1 'Heebo',sans-serif;
  cursor:pointer; z-index:31; box-shadow:0 2px 8px rgba(17,28,63,.18); }}
.px-a11y:hover, .px-a11y[aria-expanded="true"] {{ background:var(--navy); color:#fff; border-color:#fff; }}
.px-a11y-panel {{ position:absolute; left:14px; top:calc(100% + 24px); display:flex; gap:8px; padding:8px; z-index:32;
  background:#fff; border:1.5px solid var(--navy); border-radius:16px; box-shadow:0 10px 30px rgba(17,28,63,.16); direction:rtl; }}
.px-a11y-panel[hidden] {{ display:none; }}
.px-a11y-panel button {{ border:1.5px solid var(--navy); background:#fff; color:var(--navy); border-radius:999px; padding:6px 12px;
  font:500 14px/1.2 'Heebo',sans-serif; cursor:pointer; white-space:nowrap; }}
.px-a11y-panel button:hover, .px-a11y-panel button[aria-pressed="true"] {{ background:var(--navy); color:#fff; }}
.px-a11y-panel button[aria-disabled="true"] {{ opacity:.45; cursor:default; }}
/* the keyboard frame - only for keyboard navigation (:focus-visible), never on a mouse click */
.stApp :focus-visible:not([data-testid="stChatInputTextArea"]), .px-launcher:focus-visible {{ outline:3px solid var(--navy) !important;
  outline-offset:2px; }}  /* the input box: its orange frame already shows it (a text box is "focus-visible" even by mouse) */
.px-header :focus-visible {{ outline-color:#fff !important; }}
/* text size: large / very large - the bubbles (and everything inside them) and the input box */
body.px-size-1 .px-bubble {{ font-size:18px; }}
body.px-size-2 .px-bubble {{ font-size:20px; }}
body.px-size-1 .px-sources {{ font-size:16px; }}
body.px-size-2 .px-sources {{ font-size:18px; }}
body.px-size-1 .px-hours, body.px-size-1 .px-checked {{ font-size:15px; }}
body.px-size-2 .px-hours, body.px-size-2 .px-checked {{ font-size:17px; }}
body.px-size-1 [data-testid="stChatInputTextArea"] {{ font-size:18px; line-height:26px; }}
body.px-size-2 [data-testid="stChatInputTextArea"] {{ font-size:20px; line-height:28px; }}
/* high contrast: only navy and white (16.6:1, WCAG AA needs 4.5:1) - the grey and orange text become navy, frames thicker */
body.px-contrast .stMain, body.px-contrast [data-testid="stBottomBlockContainer"] {{ background:#fff !important; }}
body.px-contrast .px-time, body.px-contrast .px-meta, body.px-contrast .px-checked, body.px-contrast .px-hours,
body.px-contrast [data-testid="stChatInputTextArea"]::placeholder {{ color:var(--navy) !important; opacity:1; }}
body.px-contrast .px-meta svg, body.px-contrast .px-meta svg path {{ stroke:var(--navy); }}
body.px-contrast .px-orange {{ color:var(--navy); text-decoration:underline; }}
body.px-contrast .px-bot .px-bubble, body.px-contrast .px-avatar {{ border-width:2px; }}
body.px-contrast .px-bubble a {{ text-decoration-thickness:2px; }}
body.px-contrast .px-card {{ box-shadow:none; border:2px solid var(--navy); }}
body.px-contrast [data-testid="stChatInput"] > div > div > div:first-child {{ border-color:var(--navy); }}
body.px-contrast .stMain [data-testid^="stBaseButton"], body.px-contrast .stMain [data-testid^="stBaseLinkButton"] {{ border-width:2px !important; }}
body.px-contrast [data-testid="stFeedback"] button, body.px-contrast [data-testid="stFeedback"] button span {{ color:var(--navy) !important; }}
</style>"""


def esc_attr(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def header_html(new_chat_href: str) -> str:
    """The top bar: the robot icon and the title on the right; on the left "שיחה חדשה" (a link to a new conversation id) and
    the close mark (a button: script_html() hides the window and shows the launcher; the conversation stays)."""
    close = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" '
             'stroke-linecap="round"><path d="M4 4l16 16M20 4L4 20"/></svg>')
    return (f'<div class="px-header"><div class="px-icon"></div><div class="px-header-title">הבוט האישי שלך</div>'
            f'<a class="px-new" href="{esc_attr(new_chat_href)}" target="_self">שיחה חדשה</a>'
            f'<button class="px-close" type="button" aria-label="סגירת החלון" title="סגירה">{close}</button>'
            '<button class="px-a11y" type="button" aria-label="נגישות: גודל טקסט וניגודיות" title="נגישות"'
            ' aria-expanded="false" aria-controls="px-a11y-panel">Aa</button>'
            '<div class="px-a11y-panel" id="px-a11y-panel" role="group" aria-label="נגישות" hidden>'
            '<button class="px-a11y-smaller" type="button" aria-label="הקטנת הטקסט">א-</button>'
            '<button class="px-a11y-bigger" type="button" aria-label="הגדלת הטקסט">א+</button>'
            '<button class="px-a11y-contrast" type="button" aria-pressed="false">ניגודיות גבוהה</button></div></div>')


LAUNCHER_URL = "app/static/launcher.png"  # the owner's icon ("עוזר אישי"), as she sent it (design-references/launcher-icon.png)


def script_html(start_closed: bool = False) -> str:
    """For st.html(..., unsafe_allow_javascript=True), installed once per page (window.__pxUI):
    - the X hides the window (body.px-closed) and shows the launcher at the bottom left; the launcher shows the window again;
    - a short "send" sound when the customer sends a message (Enter in the box, or the send button), made by the browser
      (Web Audio) - no file. window.__pxSends counts the sounds (for the browser test).
    - start_closed (22/10, the owner): an entry from the public link starts closed - only the launcher; a click opens it.
    - a new conversation opens on the robot illustration: until the customer's first message Streamlit's scroll-to-bottom
      container is kept at the top (from the first message on, Streamlit scrolls to the answers as before).
    Event delegation on the document: Streamlit can redraw the header and the box, the listeners stay.
    No "<" followed by a letter or "/" inside the script: Streamlit's HTML sanitizer drops such a script completely (checked
    21/10 - the launcher was first written as an "<img ...>" string, and nothing ran)."""
    return """<script>
(function () {
  if (window.__pxUI) return;
  window.__pxUI = true;
  var doc = document, audio = null;
  function launcher() {
    var b = doc.querySelector('.px-launcher');
    if (!b) {
      b = doc.createElement('button');
      b.type = 'button'; b.className = 'px-launcher';
      b.setAttribute('aria-label', 'פתיחת הבוט'); b.title = 'פתיחת הבוט';
      var img = doc.createElement('img');
      img.src = 'LAUNCHER'; img.alt = 'עוזר אישי';
      b.appendChild(img);
      doc.body.appendChild(b);
    }
    return b;
  }
  function pop() {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === 'suspended') audio.resume();
      var t = audio.currentTime, o = audio.createOscillator(), g = audio.createGain();
      o.type = 'sine';
      o.frequency.setValueAtTime(520, t);
      o.frequency.exponentialRampToValueAtTime(1050, t + 0.12);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.16, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);
      o.connect(g); g.connect(audio.destination);
      o.start(t); o.stop(t + 0.22);
    } catch (e) {}
    window.__pxSends = (window.__pxSends || 0) + 1;
  }
  function box() { return doc.querySelector('textarea[data-testid="stChatInputTextArea"]'); }
  function top() {
    if (doc.querySelector('.px-user')) return false;
    var m = doc.querySelector('[data-testid="stAppScrollToBottomContainer"]');
    if (m && m.scrollTop !== 0) m.scrollTop = 0;
    return true;
  }
  function settle() { [60, 250, 600, 1200, 2000, 3000].forEach(function (ms) { setTimeout(top, ms); }); }
  /* until the first question: a scroll the customer did not make (no wheel, finger or key in the last 0.8 s) is
     Streamlit's scroll-to-bottom - back to the illustration. The customer can still scroll down herself. */
  var userAt = 0;
  ['wheel', 'touchmove', 'keydown', 'mousedown'].forEach(function (name) {
    doc.addEventListener(name, function () { userAt = Date.now(); }, {capture: true, passive: true});
  });
  doc.addEventListener('scroll', function (e) {
    var m = e.target;
    if (m && m.getAttribute && m.getAttribute('data-testid') === 'stAppScrollToBottomContainer' &&
        Date.now() - userAt > 800) top();
  }, true);
  var watch = new MutationObserver(function () { if (!top()) watch.disconnect(); });
  watch.observe(doc.body, {childList: true, subtree: true});
  settle();
  if (STARTCLOSED) { launcher(); doc.body.classList.add('px-closed'); }
  doc.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.closest) return;
    if (t.closest('.px-close')) {
      e.preventDefault(); var l = launcher(); doc.body.classList.add('px-closed');
      try { l.focus(); } catch (x) {}
      return;
    }
    if (t.closest('.px-launcher')) {
      doc.body.classList.remove('px-closed');
      settle();  /* Streamlit scrolls to the bottom again when the window becomes visible - keep the illustration */
      var bx = box(); if (bx) { try { bx.focus({preventScroll: true}); } catch (x) {} }
      return;
    }
    var send = t.closest('[data-testid="stChatInputSubmitButton"]');
    if (send && !send.disabled && box() && box().value.trim()) pop();
  }, true);
  doc.addEventListener('keydown', function (e) {
    var t = e.target;
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && t && t.matches &&
        t.matches('textarea[data-testid="stChatInputTextArea"]') && t.value.trim()) pop();
  }, true);
})();
</script>""".replace("LAUNCHER", LAUNCHER_URL).replace("STARTCLOSED", "true" if start_closed else "false")


def a11y_script_html() -> str:
    """Accessibility (22/10), for st.html(..., unsafe_allow_javascript=True) - a separate script from script_html(), so a
    problem here cannot stop the X, the launcher or the send sound. Installed once per page (window.__pxA11y):
    - "Aa" opens a small panel: "א-" / "א+" (3 sizes: normal, large, very large) and "ניגודיות גבוהה" / "מצב רגיל"
      (aria-pressed). The choice is saved in localStorage - also after a refresh; when it cannot be saved: normal mode.
    - Escape or a click outside closes the panel. After the bot's answer appears, the cursor returns to the input box
      (only with a mouse/keyboard - on a phone it would open the keyboard every time).
    - Streamlit redraws the header on every message: a MutationObserver puts the state back on the new buttons.
    No "<" in the script at all (Streamlit's sanitizer drops a script with "<" followed by a letter)."""
    return """<script>
(function () {
  if (window.__pxA11y) return;
  window.__pxA11y = true;
  var doc = document, body = doc.body, NAMES = ['רגיל', 'גדול', 'גדול מאוד'];
  var state = {size: 0, contrast: false, open: false}, bots = -1, pending = false;
  function load() {
    try {
      var s = parseInt(localStorage.getItem('pxTextSize') || '0', 10);
      state.size = (s === 1 || s === 2) ? s : 0;
      state.contrast = localStorage.getItem('pxContrast') === '1';
    } catch (e) { state.size = 0; state.contrast = false; }
  }
  function save() {
    try {
      localStorage.setItem('pxTextSize', String(state.size));
      localStorage.setItem('pxContrast', state.contrast ? '1' : '0');
    } catch (e) {}
  }
  function attr(el, name, value) { if (el && el.getAttribute(name) !== value) el.setAttribute(name, value); }
  function text(el, value) { if (el && el.textContent !== value) el.textContent = value; }
  function apply() {
    body.classList.toggle('px-size-1', state.size === 1);
    body.classList.toggle('px-size-2', state.size === 2);
    body.classList.toggle('px-contrast', state.contrast);
    attr(doc.querySelector('.px-a11y'), 'aria-expanded', state.open ? 'true' : 'false');
    var panel = doc.querySelector('.px-a11y-panel');
    if (!panel) return;
    if (state.open) panel.removeAttribute('hidden'); else if (!panel.hasAttribute('hidden')) panel.setAttribute('hidden', '');
    var c = panel.querySelector('.px-a11y-contrast');
    attr(c, 'aria-pressed', state.contrast ? 'true' : 'false');
    text(c, state.contrast ? 'מצב רגיל' : 'ניגודיות גבוהה');
    var big = panel.querySelector('.px-a11y-bigger'), small = panel.querySelector('.px-a11y-smaller');
    attr(big, 'aria-disabled', state.size === 2 ? 'true' : 'false');
    attr(small, 'aria-disabled', state.size === 0 ? 'true' : 'false');
    attr(big, 'aria-label', 'הגדלת הטקסט, עכשיו: ' + NAMES[state.size]);
    attr(small, 'aria-label', 'הקטנת הטקסט, עכשיו: ' + NAMES[state.size]);
  }
  function close(focusBadge) {
    state.open = false; apply();
    if (focusBadge) { var b = doc.querySelector('.px-a11y'); if (b) b.focus(); }
  }
  doc.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.closest) return;
    if (t.closest('.px-a11y')) { e.preventDefault(); state.open = !state.open; apply(); return; }
    if (t.closest('.px-a11y-bigger')) { if (state.size !== 2) { state.size += 1; save(); } apply(); return; }
    if (t.closest('.px-a11y-smaller')) { if (state.size !== 0) { state.size -= 1; save(); } apply(); return; }
    if (t.closest('.px-a11y-contrast')) { state.contrast = !state.contrast; save(); apply(); return; }
    if (state.open && !t.closest('.px-a11y-panel')) close(false);
  }, true);
  doc.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && state.open) { e.preventDefault(); close(true); }
  }, true);
  function afterRender() {
    apply();
    var n = doc.querySelectorAll('.px-bot').length;
    if (bots === -1 || n === bots) { bots = n; return; }
    var grew = n > bots;
    bots = n;
    if (!grew || body.classList.contains('px-closed')) return;
    var fine = window.matchMedia && window.matchMedia('(pointer: fine)').matches;
    var box = doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
    var a = doc.activeElement;
    if (fine && box && !(a && a.closest && a.closest('.px-a11y-panel'))) { try { box.focus({preventScroll: true}); } catch (e) {} }
  }
  new MutationObserver(function () {
    if (pending) return;
    pending = true;
    setTimeout(function () { pending = false; afterRender(); }, 60);
  }).observe(body, {childList: true, subtree: true});
  load(); apply(); afterRender();
})();
</script>"""


def hero_html() -> str:
    """The greeting area: the big robot and the white greeting card, with "הפניקס" in orange."""
    return (f'<div class="px-hero"><img class="px-robot" src="{ROBOT_URL}" alt="">'
            '<div class="px-card">שלום! 👋<br>אני הבוט של <span class="px-orange">הפניקס</span>,<br>'
            'ואני עונה על כל שאלה<br>לפי המידע שבאתר.</div></div>')


TICKS = ('<svg width="18" height="12" viewBox="0 0 18 12" fill="none" stroke="#8A8FA0" stroke-width="1.4" '
         'stroke-linecap="round" stroke-linejoin="round"><path d="M1 6.5l3.2 3.2L11 2.5"/><path d="M7 9.2l.9.8L16 2.5"/></svg>')


def user_html(content_html: str, clock: str) -> str:
    """A customer message: on the right; the time and the double tick on its left, at the bottom."""
    meta = f'<div class="px-meta">{clock}{TICKS}</div>' if clock else ""
    return f'<div class="px-row px-user"><div class="px-bubble"><div class="utext">{content_html}</div></div>{meta}</div>'


def bot_html(content_html: str, sources_html: str = "", clock: str = "") -> str:
    """A bot message: on the left; the robot icon on its left, and under the icon the time."""
    time_html = f'<div class="px-time">{clock}</div>' if clock else ""
    return (f'<div class="px-row px-bot"><div class="px-bubble"><div class="answer">{content_html}</div>{sources_html}</div>'
            f'<div class="px-side"><div class="px-avatar"></div>{time_html}</div></div>')
