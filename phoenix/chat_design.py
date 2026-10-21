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
            f'<button class="px-close" type="button" aria-label="סגירה" title="סגירה">{close}</button></div>')


LAUNCHER_URL = "app/static/launcher.png"  # the owner's icon ("עוזר אישי"), as she sent it (design-references/launcher-icon.png)


def script_html() -> str:
    """For st.html(..., unsafe_allow_javascript=True), installed once per page (window.__pxUI):
    - the X hides the window (body.px-closed) and shows the launcher at the bottom left; the launcher shows the window again;
    - a short "send" sound when the customer sends a message (Enter in the box, or the send button), made by the browser
      (Web Audio) - no file. window.__pxSends counts the sounds (for the browser test).
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
  doc.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.closest) return;
    if (t.closest('.px-close')) { e.preventDefault(); launcher(); doc.body.classList.add('px-closed'); return; }
    if (t.closest('.px-launcher')) { doc.body.classList.remove('px-closed'); return; }
    var send = t.closest('[data-testid="stChatInputSubmitButton"]');
    if (send && !send.disabled && box() && box().value.trim()) pop();
  }, true);
  doc.addEventListener('keydown', function (e) {
    var t = e.target;
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && t && t.matches &&
        t.matches('textarea[data-testid="stChatInputTextArea"]') && t.value.trim()) pop();
  }, true);
})();
</script>""".replace("LAUNCHER", LAUNCHER_URL)


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
