"""
NiceGUI frontend for LangGraph Multi-HITL FastAPI backend.
Run:  python app.py -> http://localhost:8080
Backend must be running on http://localhost:8008

Hardened version:
- UI stays alive when backend goes offline
- Clear online/offline status
- Automatic recovery when backend returns
- All network calls are exception-safe
"""

import json
from datetime import datetime
from html import escape
from typing import Optional

import httpx
from nicegui import ui

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
API_BASE = "http://localhost:8008".strip()

NODE_NAMES = [
    "agent",
    "ragger",
    "web_scraper",
    "memer",
    "finalizer",
]

ACCEPTED_FILES = ".txt,.pdf,.md,.json,.csv,.py,.docx,.xlsx"

DECISION_LABELS = {
    "Y": "✅ Approve",
    "N": "❌ Reject",
    "E": "✏️ Edit Arguments",
}

# How many times the UI may automatically repeat an identical approved tool request.
MAX_AUTO_REPEAT_IDENTICAL_INTERRUPTS = 3

CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Space+Grotesk:wght@400;500;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root {
  --bg-void: #030507;
  --bg-deep: #05080d;
  --bg-panel: rgba(10, 16, 24, 0.55);
  --bg-panel-solid: #0a1018;
  --bg-glass: rgba(14, 22, 33, 0.45);
  --bg-glass-hover: rgba(20, 32, 48, 0.65);
  --border-faint: rgba(0, 255, 200, 0.06);
  --border-subtle: rgba(0, 255, 200, 0.12);
  --border-glow: rgba(0, 255, 200, 0.35);
  --border-cyan: rgba(34, 211, 238, 0.25);
  --shadow-sm: 0 4px 20px rgba(0, 0, 0, 0.5);
  --shadow-md: 0 12px 40px rgba(0, 0, 0, 0.6);
  --shadow-glow: 0 0 30px rgba(0, 255, 200, 0.08);
  --radius-card: 12px;
  --radius-btn: 6px;
  --radius-input: 6px;
  --accent: #00ffc8;
  --accent-dim: rgba(0, 255, 200, 0.10);
  --accent-2: #22d3ee;
  --accent-3: #a78bfa;
  --warn: #ffb224;
  --danger: #ff4d6d;
  --accent-glow: 0 0 24px rgba(0, 255, 200, 0.30);
  --cyan-glow: 0 0 24px rgba(34, 211, 238, 0.25);
  --text-primary: #d8e6f0;
  --text-secondary: #7d93a8;
  --text-muted: #4a5d70;
  --font: 'JetBrains Mono', 'Fira Code', monospace;
  --font-display: 'Space Grotesk', sans-serif;
  --font-hack: 'Share Tech Mono', monospace;
}

* { box-sizing: border-box; }

html, body {
  background: var(--bg-void) !important;
}

body {
  font-family: var(--font);
  color: var(--text-primary);
  line-height: 1.55;
  margin: 0;
  min-height: 100vh;
  background-image:
    radial-gradient(ellipse 100% 60% at 50% -20%, rgba(0, 255, 200, 0.06), transparent),
    radial-gradient(ellipse 60% 40% at 100% 100%, rgba(34, 211, 238, 0.05), transparent),
    radial-gradient(ellipse 40% 30% at 0% 80%, rgba(167, 139, 250, 0.04), transparent),
    linear-gradient(rgba(0, 255, 200, 0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 255, 200, 0.02) 1px, transparent 1px);
  background-size: auto, auto, auto, 44px 44px, 44px 44px;
  animation: gridDrift 30s linear infinite;
}
@keyframes gridDrift {
  from { background-position: 0 0, 0 0, 0 0, 0 0, 0 0; }
  to   { background-position: 0 0, 0 0, 0 0, 44px 44px, 44px 44px; }
}

/* ── CRT scanline + vignette overlay ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: repeating-linear-gradient(
    0deg, transparent 0px, transparent 3px, rgba(0, 0, 0, 0.18) 4px
  );
  z-index: 9998;
  opacity: 0.45;
}
body::after {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: radial-gradient(ellipse 120% 100% at 50% 50%, transparent 55%, rgba(0,0,0,0.5) 100%);
  z-index: 9997;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.015); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(0,255,200,0.35), rgba(34,211,238,0.25));
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent); box-shadow: var(--accent-glow); }

/* ── Quasar overrides ── */
.q-field__control {
  border-radius: var(--radius-input) !important;
  background: rgba(255, 255, 255, 0.025) !important;
  backdrop-filter: blur(14px);
  border: 1px solid var(--border-faint);
  transition: all 0.25s ease;
}
.q-field__control:hover { border-color: var(--border-subtle); }
.q-field__control:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent-dim), var(--accent-glow);
  background: rgba(0, 255, 200, 0.03) !important;
}
.q-field__label, .q-field__native, .q-field__prefix { color: var(--text-secondary) !important; font-family: var(--font) !important; }
.q-field--focused .q-field__label { color: var(--accent) !important; }

.q-btn {
  border-radius: var(--radius-btn) !important;
  font-family: var(--font) !important;
  font-weight: 500;
  text-transform: none !important;
  letter-spacing: 0.04em;
  transition: all 0.2s ease !important;
}
.q-btn--unelevated {
  background: linear-gradient(135deg, rgba(0,255,200,0.14), rgba(34,211,238,0.08)) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border-glow);
  position: relative;
  overflow: hidden;
}
.q-btn--unelevated::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 60%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  transition: left 0.4s ease;
}
.q-btn--unelevated:hover::before { left: 140%; }
.q-btn--unelevated:hover {
  transform: translateY(-1px);
  box-shadow: var(--accent-glow);
  border-color: var(--accent);
}
.q-btn--outline { border: 1px solid currentColor; color: var(--text-secondary) !important; }
.q-btn--outline:hover { color: var(--accent) !important; box-shadow: var(--accent-glow); border-color: var(--accent); }
.q-btn--flat { color: var(--text-secondary) !important; }
.q-btn--flat:hover { background: var(--accent-dim) !important; color: var(--accent) !important; }
.q-btn[disabled] { opacity: 0.3 !important; }

.q-radio__inner--truthy { color: var(--accent) !important; }
.q-checkbox__inner--truthy { color: var(--accent) !important; }

.q-separator { background: var(--border-faint) !important; }

.q-expansion-item__container {
  border-radius: var(--radius-card);
  background: var(--bg-glass);
  backdrop-filter: blur(16px);
  border: 1px solid var(--border-faint);
  transition: all 0.25s ease;
}
.q-expansion-item__container:hover { border-color: var(--border-subtle); }
.q-expansion-item__toggle-icon { color: var(--accent); }

/* ── Header ── */
.q-header {
  background: rgba(4, 8, 12, 0.82) !important;
  backdrop-filter: blur(20px) saturate(1.4) !important;
  border-bottom: 1px solid var(--border-subtle) !important;
  box-shadow: 0 4px 30px rgba(0, 255, 200, 0.06), 0 1px 0 rgba(0,255,200,0.1) inset;
  padding: 6px 14px;
}

/* ── Footer ── */
.q-footer {
  background: rgba(4, 8, 12, 0.88) !important;
  backdrop-filter: blur(20px) saturate(1.4) !important;
  border-top: 1px solid var(--border-subtle) !important;
  box-shadow: 0 -4px 30px rgba(0, 0, 0, 0.4);
}

/* ── Drawer / Sidebar ── */
.q-drawer {
  background: rgba(5, 9, 14, 0.88) !important;
  backdrop-filter: blur(28px) saturate(1.3) !important;
  border-right: 1px solid var(--border-subtle) !important;
  box-shadow: 4px 0 40px rgba(0, 0, 0, 0.5);
  transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1), transform 0.3s ease !important;
}
.q-drawer .q-label { color: var(--text-secondary) !important; font-family: var(--font) !important; font-size: 0.78rem; }

/* ── Header typography ── */
.main-header {
  font-family: var(--font-display);
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--accent) !important;
  text-shadow: 0 0 30px rgba(0, 255, 200, 0.5), 0 0 60px rgba(0, 255, 200, 0.2);
  letter-spacing: 0.06em;
  margin: 0;
  animation: fadeInDown 0.5s ease;
  position: relative;
}
.main-header::after {
  content: '';
  position: absolute;
  bottom: -4px; left: 0;
  width: 40%; height: 1px;
  background: linear-gradient(90deg, var(--accent), transparent);
}

.avatar-emoji {
  font-size: 1.3rem;
  filter: drop-shadow(0 0 10px rgba(0, 255, 200, 0.4));
  display: inline-block;
  animation: float 3.5s ease-in-out infinite;
}
@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-3px); }
}

/* ── Chat bubbles ── */
.user-msg {
  background: linear-gradient(135deg, rgba(0,255,200,0.10), rgba(34,211,238,0.05));
  color: var(--text-primary);
  border: 1px solid var(--border-glow);
  padding: 10px 16px;
  border-radius: 10px 10px 2px 10px;
  margin: 6px 0;
  max-width: 85%;
  backdrop-filter: blur(12px);
  box-shadow: var(--shadow-sm);
  word-wrap: break-word;
  animation: slideIn 0.35s ease;
  position: relative;
}
.user-msg::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,255,200,0.4), transparent);
}
.user-msg:hover { box-shadow: var(--accent-glow); }

.assistant-msg {
  background: var(--bg-glass);
  color: var(--text-primary);
  border: 1px solid var(--border-faint);
  padding: 10px 16px;
  border-radius: 10px 10px 10px 2px;
  margin: 6px 0;
  max-width: 85%;
  backdrop-filter: blur(14px);
  box-shadow: var(--shadow-sm);
  word-wrap: break-word;
  animation: slideIn 0.35s ease;
  transition: border-color 0.2s ease;
}
.assistant-msg:hover { border-color: var(--border-subtle); }
.assistant-msg p { color: var(--text-primary) !important; }
.assistant-msg code {
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border-faint);
  border-radius: 3px;
  padding: 0 4px;
  font-family: var(--font-hack);
}
.assistant-msg pre {
  background: rgba(0,0,0,0.5) !important;
  border: 1px solid var(--border-faint);
  border-radius: 8px;
}
.assistant-msg strong { color: var(--accent-2); }

.node-msg {
  background: rgba(34, 211, 238, 0.04);
  color: var(--text-secondary);
  padding: 6px 12px;
  border-radius: 6px;
  border-left: 2px solid var(--accent-2);
  font-size: 0.76rem;
  max-width: 85%;
  margin: 3px 0;
  font-family: var(--font-hack);
  animation: slideIn 0.3s ease;
  letter-spacing: 0.02em;
}
.node-msg b { color: var(--accent-2); }

@keyframes slideIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── Typing cursor ── */
.typing::after {
  content: '▌';
  color: var(--accent);
  animation: blink 1s steps(2) infinite;
  text-shadow: 0 0 10px var(--accent);
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Interrupt box ── */
.interrupt-box {
  background: rgba(25, 18, 4, 0.45);
  backdrop-filter: blur(18px);
  border: 1px solid rgba(255, 178, 36, 0.22);
  border-radius: var(--radius-card);
  padding: 18px 22px;
  margin: 12px 0;
  box-shadow: 0 0 40px rgba(255, 178, 36, 0.06), var(--shadow-md);
  position: relative;
  overflow: hidden;
  animation: slideIn 0.4s ease;
}
.interrupt-box::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--warn), transparent);
  animation: scanLine 3s linear infinite;
  pointer-events: none;
}
@keyframes scanLine {
  0% { top: 0; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

.cycle-badge {
  background: rgba(0, 255, 200, 0.07);
  color: var(--accent);
  padding: 3px 12px;
  border-radius: 3px;
  font-size: 0.66rem;
  font-weight: 700;
  border: 1px solid var(--border-glow);
  font-family: var(--font-hack);
  letter-spacing: 0.08em;
  animation: pulseBadge 2.5s ease-in-out infinite;
}
@keyframes pulseBadge {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 200, 0.15); }
  50% { box-shadow: 0 0 14px rgba(0, 255, 200, 0.3); }
}

.tool-card {
  background: rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(10px);
  padding: 12px 16px;
  border-radius: 8px;
  border-left: 2px solid var(--accent);
  border-top: 1px solid var(--border-faint);
  border-right: 1px solid var(--border-faint);
  border-bottom: 1px solid var(--border-faint);
  margin: 10px 0;
  font-family: var(--font);
  transition: all 0.2s ease;
}
.tool-card:hover {
  border-left-color: var(--accent-2);
  box-shadow: var(--accent-glow);
  transform: translateX(4px);
}
.tool-card p { color: var(--text-secondary) !important; font-size: 0.84rem; }
.tool-card code {
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
  padding: 0 4px;
  border-radius: 3px;
  font-family: var(--font-hack);
}
.tool-card strong { color: var(--accent-2) !important; }

/* ── Edit rows / file badges ── */
.edit-row {
  background: rgba(0, 0, 0, 0.3);
  padding: 10px 14px;
  border-radius: 6px;
  margin: 6px 0;
  border: 1px solid var(--border-faint);
  transition: border-color 0.2s ease;
}
.edit-row:hover { border-color: var(--border-subtle); }

.edit-label {
  font-weight: 700;
  color: var(--accent-2);
  font-size: 0.78rem;
  display: block;
  margin-bottom: 2px;
  font-family: var(--font-hack);
  letter-spacing: 0.05em;
}
.edit-current {
  color: var(--text-muted);
  background: rgba(0, 255, 200, 0.04);
  padding: 2px 8px;
  border-radius: 3px;
  font-family: var(--font-hack);
  font-size: 0.74rem;
  display: inline-block;
  margin: 4px 0;
  border: 1px solid var(--border-faint);
}

.file-badge {
  background: var(--accent-dim);
  color: var(--accent);
  padding: 3px 12px;
  border-radius: 3px;
  font-size: 0.7rem;
  display: inline-block;
  margin: 3px 6px 3px 0;
  border: 1px solid var(--border-faint);
  font-family: var(--font-hack);
  transition: all 0.2s ease;
  letter-spacing: 0.03em;
}
.file-badge:hover {
  background: rgba(0, 255, 200, 0.18);
  box-shadow: var(--accent-glow);
  transform: translateY(-1px);
}

/* ── Notifications ── */
.q-notification {
  border-radius: 8px !important;
  backdrop-filter: blur(18px) !important;
  background: rgba(8, 13, 20, 0.94) !important;
  border: 1px solid var(--border-glow) !important;
  color: var(--text-primary) !important;
  font-family: var(--font) !important;
  box-shadow: var(--shadow-md), var(--accent-glow) !important;
}
.q-notification .q-notification__message { color: var(--text-primary) !important; }

/* ── Spinner ── */
.q-spinner { color: var(--accent) !important; filter: drop-shadow(0 0 6px rgba(0,255,200,0.5)); }

/* ── Page fade ── */
.q-page-container { animation: fadeIn 0.6s ease; }
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── Status dot ── */
.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 6px;
}
.status-dot.online { background: var(--accent); box-shadow: 0 0 12px var(--accent); animation: pulseBadge 2s infinite; }
.status-dot.offline { background: var(--danger); box-shadow: 0 0 12px var(--danger); }

/* ── Sidebar section labels ── */
.side-label {
  font-family: var(--font-hack);
  font-size: 0.68rem;
  letter-spacing: 0.15em;
  color: var(--text-muted);
  text-transform: uppercase;
  padding: 4px 0;
  border-bottom: 1px solid var(--border-faint);
  margin-bottom: 6px;
}

/* ── Responsive ── */
@media (max-width: 768px) {
  .main-header { font-size: 1.1rem; }
  .user-msg, .assistant-msg, .node-msg { max-width: 95%; }
  .interrupt-box { padding: 12px 14px; }
}
/* ══════════════════════════════════════════════════
   NEXUS//AGENT — FX PACK v2
   cursor zoom · spotlight · tilt · glow trails
   ══════════════════════════════════════════════════ */

/* ── 1. CURSOR ZOOM LENS ──
   A subtle "zooming" ring that expands toward the cursor */
:root {
  --mx: 50vw;
  --my: 50vh;
}

/* soft radial spotlight that follows the mouse */
body::before {
  background:
    radial-gradient(600px circle at var(--mx) var(--my),
      rgba(0, 255, 200, 0.045), transparent 70%),
    repeating-linear-gradient(0deg, transparent 0px, transparent 3px,
      rgba(0, 0, 0, 0.18) 4px);
}

/* ── 2. MAGNIFY CURSOR RING (pure CSS, no JS) ── */
.cursor-ring {
  position: fixed;
  top: 0; left: 0;
  width: 34px; height: 34px;
  border: 1.5px solid rgba(0, 255, 200, 0.55);
  border-radius: 50%;
  pointer-events: none;
  z-index: 9999;
  transform: translate(-50%, -50%);
  transition: width 0.18s ease, height 0.18s ease,
              border-color 0.18s ease, opacity 0.3s ease;
  box-shadow:
    0 0 20px rgba(0, 255, 200, 0.25),
    inset 0 0 12px rgba(0, 255, 200, 0.10);
  mix-blend-mode: screen;
  opacity: 0;
}
.cursor-ring.visible { opacity: 1; }
.cursor-ring.hovering {
  width: 56px; height: 56px;
  border-color: rgba(34, 211, 238, 0.8);
  box-shadow:
    0 0 35px rgba(34, 211, 238, 0.35),
    inset 0 0 18px rgba(34, 211, 238, 0.15);
}

/* ── 3. 3D TILT ON CARDS (JS-driven, CSS transition) ── */
.tilt-card {
  transform-style: preserve-3d;
  transition: transform 0.15s ease-out, box-shadow 0.2s ease;
  will-change: transform;
}
.tilt-card:hover {
  box-shadow:
    0 20px 50px rgba(0, 0, 0, 0.55),
    0 0 40px rgba(0, 255, 200, 0.10);
}

/* ── 4. ZOOM-ON-HOVER for messages ── */
.user-msg, .assistant-msg {
  transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1),
              box-shadow 0.25s ease, border-color 0.25s ease;
}
.user-msg:hover, .assistant-msg:hover {
  transform: scale(1.015) translateZ(0);
  border-color: var(--border-glow);
}

/* ── 5. STAGGERED MESSAGE ENTRANCE ── */
@keyframes msgIn {
  from { opacity: 0; transform: translateY(14px) scale(0.97); }
  to   { opacity: 1; transform: translateY(0)    scale(1);    }
}
.user-msg, .assistant-msg, .node-msg {
  animation: msgIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) backwards;
}

/* ── 6. AMBIENT FLOATING ORBS ── */
.orb {
  position: fixed;
  border-radius: 50%;
  filter: blur(90px);
  pointer-events: none;
  z-index: 0;
  opacity: 0.35;
  animation: orbFloat 18s ease-in-out infinite alternate;
}
.orb-1 {
  width: 420px; height: 420px;
  background: radial-gradient(circle, rgba(0,255,200,0.14), transparent 70%);
  top: -10%; left: -8%;
}
.orb-2 {
  width: 380px; height: 380px;
  background: radial-gradient(circle, rgba(34,211,238,0.12), transparent 70%);
  bottom: -12%; right: -6%;
  animation-delay: -6s;
}
.orb-3 {
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(167,139,250,0.10), transparent 70%);
  top: 40%; left: 55%;
  animation-delay: -12s;
}
@keyframes orbFloat {
  from { transform: translate(0, 0) scale(1); }
  to   { transform: translate(60px, -40px) scale(1.15); }
}

/* ── 7. GLITCH FLICKER on main header ── */
.main-header {
  animation: fadeInDown 0.5s ease, glitchFlicker 7s infinite;
}
@keyframes glitchFlicker {
  0%, 91%, 94%, 98%, 100% { opacity: 1; }
  92% { opacity: 0.75; transform: translateX(1px); }
  93% { opacity: 0.9;  transform: translateX(-1px); }
  96% { opacity: 0.8; }
}

/* ── 8. SHIMMER TEXT on status ── */
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
.status-dot.online {
  animation: pulseBadge 2s infinite;
}

/* ── 9. SMOOTH PAGE-WIDE PARALLAX GRID ── */
body {
  background-attachment: fixed;
}

/* ── 10. FOCUS ZOOM ON INPUTS ── */
.q-field--focused .q-field__control {
  transform: scale(1.005);
  transition: transform 0.2s ease, box-shadow 0.25s ease,
              border-color 0.25s ease, background 0.25s ease;
}

/* ── 11. TOOLTIP STYLING ── */
.q-tooltip {
  background: rgba(8, 13, 20, 0.95) !important;
  border: 1px solid var(--border-glow) !important;
  border-radius: 4px !important;
  font-family: var(--font-hack) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.05em;
  box-shadow: var(--accent-glow) !important;
}

/* ── 12. SCROLL-REVEAL (add .reveal class via JS if wanted) ── */
.reveal {
  opacity: 0;
  transform: translateY(24px);
  transition: opacity 0.6s ease, transform 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.reveal.visible {
  opacity: 1;
  transform: translateY(0);
}

/* ── 13. HOLO BUTTON PULSE on primary actions ── */
@keyframes holoPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,200,0.20); }
  50%      { box-shadow: 0 0 0 6px rgba(0,255,200,0.0); }
}
.q-btn--unelevated {
  animation: holoPulse 3s ease-in-out infinite;
}

/* ── 14. DISABLED STATE SHIMMER ── */
.q-btn[disabled] {
  background: rgba(255,255,255,0.03) !important;
  border-color: var(--border-faint) !important;
  color: var(--text-muted) !important;
  animation: none;
}

/* ── 15. NOTIFICATION SLIDE-IN ── */
.q-notification {
  animation: notifIn 0.35s cubic-bezier(0.22, 1, 0.36, 1);
}
@keyframes notifIn {
  from { opacity: 0; transform: translateX(40px); }
  to   { opacity: 1; transform: translateX(0); }
}
</style>
<script>
// ── Cursor ring + spotlight tracker ──
(function () {
  const ring = document.createElement('div');
  ring.className = 'cursor-ring';
  document.body.appendChild(ring);

  let rx = -100, ry = -100, tx = -100, ty = -100;

  document.addEventListener('mousemove', e => {
    tx = e.clientX; ty = e.clientY;
    document.documentElement.style.setProperty('--mx', e.clientX + 'px');
    document.documentElement.style.setProperty('--my', e.clientY + 'px');
    ring.classList.add('visible');
  });

  document.addEventListener('mouseleave', () => ring.classList.remove('visible'));

  // enlarge ring over interactive elements
  document.addEventListener('mouseover', e => {
    if (e.target.closest('button, a, .q-btn, .tool-card, input, [role="button"]'))
      ring.classList.add('hovering');
    else
      ring.classList.remove('hovering');
  });

  // smooth trailing animation
  (function animate() {
    rx += (tx - rx) * 0.18;
    ry += (ty - ry) * 0.18;
    ring.style.transform = `translate(${rx - 17}px, ${ry - 17}px)`;
    requestAnimationFrame(animate);
  })();

  // ambient orbs
  ['orb-1', 'orb-2', 'orb-3'].forEach(c => {
    const o = document.createElement('div');
    o.className = 'orb ' + c;
    document.body.appendChild(o);
  });
})();
</script>
"""


# ─────────────────────────────────────────────
# Interrupt normalization helpers
# ─────────────────────────────────────────────
def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def interrupt_signature(value) -> str:
    """Stable signature for an interrupt value (used to detect duplicates)."""
    try:
        if isinstance(value, dict):
            return json.dumps(
                {
                    "type": value.get("type"),
                    "tool_name": value.get("tool_name"),
                    "tool_args": value.get("tool_args"),
                    "question": value.get("question"),
                },
                sort_keys=True,
                default=str,
            )
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)


def normalize_interrupts(items) -> list:
    """Normalize interrupt payloads from stream / state / interrupts endpoints."""
    normalized = []
    seen_ids = set()

    def add_item(item, idx):
        if isinstance(item, list):
            for i, x in enumerate(item):
                add_item(x, f"{idx}_{i}")
            return

        if not isinstance(item, dict):
            return

        iid = str(
            item.get("id")
            or item.get("interrupt_id")
            or item.get("uuid")
            or f"unknown_{idx}_{datetime.now().microsecond}"
        )

        if iid in seen_ids:
            return

        value = item.get("value", item)

        # Some serializers put the interrupt value fields at the top level.
        if isinstance(value, dict) and "tool_name" not in value and "tool_name" in item:
            value = {
                "tool_name": item.get("tool_name"),
                "tool_args": item.get("tool_args"),
                "status": item.get("status", "N"),
            }

        entry = {
            "id": iid,
            "ids": [iid],
            "value": value,
            "ns": item.get("ns"),
            "signature": interrupt_signature(value),
            "missing_id": iid.startswith("unknown_"),
        }

        seen_ids.add(iid)
        normalized.append(entry)

    for idx, item in enumerate(_as_list(items)):
        add_item(item, idx)

    return normalized


def merge_interrupts(*lists) -> list:
    """Merge interrupt lists safely by interrupt id."""
    merged = []
    seen = set()

    for lst in lists:
        for item in normalize_interrupts(lst):
            iid = item.get("id")
            if iid in seen:
                continue
            seen.add(iid)
            merged.append(item)

    return merged


def extract_interrupts_from_state_snapshot(snapshot) -> list:
    """Extract raw interrupt objects from a LangGraph state snapshot."""
    raw = []

    if not isinstance(snapshot, dict):
        return raw

    for key in ("interrupts", "interrupt", "__interrupt__"):
        if snapshot.get(key) is not None:
            raw.extend(_as_list(snapshot.get(key)))

    values = snapshot.get("values") or {}
    if isinstance(values, dict):
        for key in ("interrupt", "__interrupt__"):
            if values.get(key) is not None:
                raw.extend(_as_list(values.get(key)))

    tasks = snapshot.get("tasks") or []
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, dict) and task.get("interrupts") is not None:
                raw.extend(_as_list(task.get("interrupts")))

    if isinstance(snapshot.get("state"), dict):
        raw.extend(extract_interrupts_from_state_snapshot(snapshot.get("state")))

    return raw


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
class SessionState:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.thread_id: Optional[str] = None
        self.events: list[dict] = []
        self.show_events: bool = False
        self.pending_interrupts: list[dict] = []
        self.uploaded_files: list[dict] = []
        self.custom_events: list[dict] = []
        self.max_iterations: int = 7
        self.interrupt_cycle: int = 0

        self.decisions: dict[str, dict] = {}
        self.event_filter: list[str] = []

        self.busy: bool = False
        self.resolved_interrupt_ids: set[str] = set()
        self.uploaded_file_keys: set[str] = set()

        # Authoritative online/offline flag – UI never dies when this becomes False
        self.api_online: bool = True


# ─────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────
def safe_notify(message: str, type: str = "info") -> None:
    try:
        ui.notify(message, type=type)
    except RuntimeError:
        print(f"[NOTIFY {type.upper()}] {message}")


def parse_sse_line(line: str) -> Optional[dict]:
    if not line or not line.startswith("data: "):
        return None
    try:
        return json.loads(line[6:])
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────
@ui.page("/")
async def index() -> None:
    state = SessionState()
    ui.add_head_html(CUSTOM_CSS)
    ui.dark_mode(True)  # forces Quasar dark palette so inputs/toggles render correctly

    # ─────────────────────────────────────────────
    # Authoritative interrupt fetch (never raises)
    # ─────────────────────────────────────────────
    async def fetch_authoritative_interrupts() -> tuple[bool, list]:
        if not state.thread_id:
            return True, []

        # Primary: thread state snapshot
        try:
            async with httpx.AsyncClient(timeout=6) as client:
                resp = await client.get(f"{API_BASE}/threads/{state.thread_id}/state")
                if resp.status_code == 200:
                    snapshot = resp.json().get("state", {})
                    raw = extract_interrupts_from_state_snapshot(snapshot)
                    return True, normalize_interrupts(raw)
        except Exception:
            pass

        # Fallback: /interrupts endpoint
        try:
            async with httpx.AsyncClient(timeout=6) as client:
                resp = await client.get(
                    f"{API_BASE}/threads/{state.thread_id}/interrupts"
                )
                resp.raise_for_status()
                raw = resp.json().get("interrupts", [])
                return True, normalize_interrupts(raw)
        except Exception:
            return False, []

    def filter_resolved_interrupts(interrupts: list) -> list:
        return [
            i for i in interrupts if i.get("id") not in state.resolved_interrupt_ids
        ]

    # ─────────────────────────────────────────────
    # UI availability helper
    # ─────────────────────────────────────────────
    def update_ui_availability() -> None:
        """Enable / disable controls based on online + busy + pending interrupts."""
        can_chat = (
            state.api_online
            and not state.busy
            and not state.pending_interrupts
            and bool(state.thread_id)
        )
        status_html.set_content(
            '<span class="status-dot online">ONLINE</span>'
            if state.api_online
            else '<span class="status-dot offline">OFFLINE</span>'
        )
        try:
            if can_chat:
                msg_input.enable()
                send_btn.enable()

            else:
                msg_input.disable()
                send_btn.disable()
        except Exception:
            pass

        # Rebuild side panels so buttons also respect the new state
        rebuild_active_thread_section()
        rebuild_upload_section()

    # ─────────────────────────────────────────────
    # UI helpers
    # ─────────────────────────────────────────────
    def scroll_to_bottom() -> None:
        try:
            ui.run_javascript(
                'window.scrollTo({top: document.body.scrollHeight, behavior: "smooth"});'
            )
        except Exception:
            pass

    def render_message(msg: dict) -> None:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        avatar = {
            "user": "👤",
            "assistant": "🤖",
            "node": "⚙️",
            "custom": "📢",
        }.get(role, "🤖")

        with ui.row().classes("w-full items-start gap-2 no-wrap"):
            ui.html(f'<span class="avatar-emoji">{avatar}</span>')

            if role == "user":
                ui.html(f'<div class="user-msg">{escape(str(content))}</div>')
            elif role == "node":
                node_name = msg.get("node", "Agent")
                ui.html(
                    f'<div class="node-msg"><b>{escape(node_name)}</b>: '
                    f"{escape(str(content))}</div>"
                )
            elif role == "custom":
                ui.html(
                    f'<div class="node-msg"><b>Status</b>: {escape(str(content))}</div>'
                )
            else:
                with ui.element("div").classes("assistant-msg"):
                    ui.markdown(str(content))

    def add_message(msg: dict) -> None:
        state.messages.append(msg)
        try:
            with chat_container:
                render_message(msg)
            scroll_to_bottom()
        except Exception:
            pass

    def rebuild_chat() -> None:
        try:
            chat_container.clear()
            with chat_container:
                for m in state.messages:
                    render_message(m)
            scroll_to_bottom()
        except Exception as e:
            print(f"[rebuild_chat] ignored error: {e}")

    # ─────────────────────────────────────────────
    # Interrupt approval UI
    # ─────────────────────────────────────────────
    async def force_refresh_interrupts() -> None:
        if state.busy:
            safe_notify(
                "Please wait for the current operation to finish.", type="warning"
            )
            return
        if not state.api_online:
            safe_notify(
                "Backend is offline. Cannot refresh interrupts.", type="warning"
            )
            return

        state.busy = True
        update_ui_availability()
        try:
            ok, current = await fetch_authoritative_interrupts()
            if ok:
                state.pending_interrupts = filter_resolved_interrupts(current)
                state.decisions = {}
                refresh_interrupt_ui()

                if not state.pending_interrupts:
                    safe_notify("No active interrupts on backend.", type="positive")
                else:
                    safe_notify(
                        f"Refreshed: {len(state.pending_interrupts)} active interrupt(s).",
                        type="info",
                    )
            else:
                safe_notify("Could not fetch interrupts from backend.", type="negative")
                state.api_online = False
        finally:
            state.busy = False
            update_ui_availability()

    def build_interrupt_approval() -> None:
        interrupts = state.pending_interrupts

        with ui.element("div").classes("interrupt-box"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label(f"⏸️ {len(interrupts)} Tool Execution(s) Paused").classes(
                    "text-lg font-bold text-orange-700"
                )

                if state.interrupt_cycle > 0:
                    ui.html(
                        f'<span class="cycle-badge">Cycle #{state.interrupt_cycle}</span>'
                    )

                ui.button(icon="refresh", on_click=force_refresh_interrupts).props(
                    "dense flat round color=grey-8"
                ).tooltip("Force refresh from backend state")

            progress_label = ui.label("")

            def decided_count() -> int:
                count = 0
                for intr in interrupts:
                    iid = intr.get("id")
                    decision = state.decisions.get(iid, {}).get("decision", "pending")
                    if decision in ("Y", "N", "E"):
                        count += 1
                return count

            def collect_approvals() -> list:
                approvals = []

                for intr in interrupts:
                    iid = intr.get("id")
                    sig = intr.get("signature")
                    ids = intr.get("ids") or [iid]

                    dec = state.decisions.get(iid, {})
                    decision = dec.get("decision", "pending")

                    if decision not in ("Y", "N", "E"):
                        continue

                    tool_args = dec.get("edited_args") if decision == "E" else None
                    feedback = dec.get("feedback") if decision == "N" else None

                    for actual_id in ids:
                        approvals.append(
                            {
                                "interrupt_id": actual_id,
                                "decision": decision,
                                "tool_args": tool_args,
                                "feedback": feedback,
                                "signature": sig,
                            }
                        )

                return approvals

            for idx, interrupt in enumerate(interrupts):
                interrupt_id = interrupt.get("id", f"unknown_{idx}")
                value = interrupt.get("value", {})

                if isinstance(value, dict):
                    tool_name = value.get("tool_name", "Unknown Tool")
                    tool_args = value.get("tool_args", {}) or {}
                    status = value.get("status", "N")
                else:
                    tool_name = "Unknown Tool"
                    tool_args = {}
                    status = "N"

                dec = state.decisions.setdefault(
                    interrupt_id,
                    {
                        "decision": "pending",
                        "edited_args": {},
                        "feedback": "",
                    },
                )
                # ── Finalizer feedback branch ─────────────────────────
                is_final_feedback = isinstance(value, dict) and value.get("type") in (
                    "final_feedback",
                    "finalizer_feedback",
                )

                if is_final_feedback:
                    question = value.get("question", "Was that what you want?")
                    with ui.element("div").classes("tool-card"):
                        ui.markdown("**🎯 Finalizer Feedback**")
                        ui.markdown(f"**Question:** `{question}`")

                        decision_radio = ui.radio(
                            {
                                "Y": "✅ Yes, that's what I wanted",
                                "N": "❌ No, please revise",
                            },
                            value=dec["decision"]
                            if dec["decision"] in ("Y", "N")
                            else None,
                        ).props("inline")

                        feedback_box = ui.column().classes("w-full gap-1")

                        def rebuild_feedback(_c=feedback_box, _dec=dec):
                            _c.clear()
                            with _c:
                                if _dec["decision"] == "N":
                                    ui.input(
                                        label="What should be changed? (optional)",
                                        value=_dec.get("feedback", ""),
                                    ).classes("w-full").on_value_change(
                                        lambda e, d=_dec: d.update(feedback=e.value)
                                    )

                        rebuild_feedback()

                        def on_feedback_decision(e, _dec=dec, _rb=rebuild_feedback):
                            _dec["decision"] = (
                                e.value if e.value in ("Y", "N") else "pending"
                            )
                            _rb()
                            update_progress()

                        decision_radio.on_value_change(on_feedback_decision)

                    continue  # skip the standard tool card
                # ──────────────────────────────────────────────────────

                # Initialize edited_args with current values
                if isinstance(tool_args, dict):
                    for arg_key, arg_val in tool_args.items():
                        dec["edited_args"].setdefault(arg_key, str(arg_val))

                with ui.element("div").classes("tool-card"):
                    ui.markdown(f"**🔧 Tool #{idx + 1}** `{str(interrupt_id)[:12]}...`")
                    ui.markdown(f"**Tool:** `{tool_name}`")
                    ui.markdown(f"**Status:** `{status}`")

                    if not isinstance(value, dict):
                        ui.markdown(f"**Value:** `{str(value)}`")

                    decision_radio = ui.radio(
                        DECISION_LABELS,
                        value=dec["decision"]
                        if dec["decision"] in DECISION_LABELS
                        else None,
                    ).props("inline")

                    extras_container = ui.column().classes("w-full gap-1")

                    def rebuild_extras(
                        _c=extras_container, _dec=dec, _args=tool_args
                    ) -> None:
                        _c.clear()
                        with _c:
                            if _dec["decision"] == "E":
                                ui.markdown("**Edit each argument:**")

                                current = _args if isinstance(_args, dict) else {}

                                for arg_key, arg_val in current.items():
                                    with ui.element("div").classes("edit-row"):
                                        ui.label(str(arg_key)).classes("edit-label")
                                        ui.html(
                                            f'<span class="edit-current">'
                                            f"Current Value: {escape(str(arg_val))}"
                                            f"</span>"
                                        )

                                        ui.input(
                                            label=f"New Value for {arg_key}:",
                                            value=str(
                                                _dec["edited_args"].get(
                                                    arg_key, arg_val
                                                )
                                            ),
                                        ).classes("w-full").on_value_change(
                                            lambda e, k=arg_key, d=_dec: d[
                                                "edited_args"
                                            ].update({k: e.value})
                                        )

                            elif _dec["decision"] == "N":
                                ui.input(
                                    label="Rejection reason (optional):",
                                    value=_dec.get("feedback", ""),
                                ).classes("w-full").on_value_change(
                                    lambda e, d=_dec: d.update(feedback=e.value)
                                )

                    rebuild_extras()

                    def on_decision_change(
                        e, _dec=dec, _args=tool_args, _rb=rebuild_extras
                    ):
                        _dec["decision"] = (
                            e.value if e.value in DECISION_LABELS else "pending"
                        )
                        _rb()
                        update_progress()

                    decision_radio.on_value_change(on_decision_change)

            async def submit_all() -> None:
                if decided_count() < len(interrupts):
                    safe_notify("Decide on ALL tools first!", type="warning")
                    return
                await send_batch_resume(collect_approvals())

            async def approve_all() -> None:
                approvals = []
                for intr in interrupts:
                    iid = intr.get("id")
                    sig = intr.get("signature")
                    ids = intr.get("ids") or [iid]

                    for actual_id in ids:
                        approvals.append(
                            {
                                "interrupt_id": actual_id,
                                "decision": "Y",
                                "signature": sig,
                            }
                        )

                await send_batch_resume(approvals)

            async def reject_all() -> None:
                approvals = []
                for intr in interrupts:
                    iid = intr.get("id")
                    sig = intr.get("signature")
                    ids = intr.get("ids") or [iid]

                    for actual_id in ids:
                        approvals.append(
                            {
                                "interrupt_id": actual_id,
                                "decision": "N",
                                "feedback": "Batch rejected",
                                "signature": sig,
                            }
                        )

                await send_batch_resume(approvals)

            with ui.row().classes("w-full gap-2 no-wrap pt-1"):
                submit_btn = (
                    ui.button("🚀 Submit", on_click=submit_all)
                    .props("color=primary unelevated")
                    .classes("flex-grow")
                )

                ui.button("✅ Approve ALL", on_click=approve_all).props(
                    "color=positive outline unelevated"
                ).classes("flex-grow")

                ui.button("❌ Reject ALL", on_click=reject_all).props(
                    "color=negative outline unelevated"
                ).classes("flex-grow")

            def update_progress() -> None:
                decided = decided_count()
                total = len(interrupts)
                progress_label.set_text(
                    f"Decided: {decided}/{total} | Pending: {total - decided}"
                )

                if decided == total:
                    submit_btn.enable()
                else:
                    submit_btn.disable()

            update_progress()

    def refresh_interrupt_ui() -> None:
        try:
            interrupt_container.clear()
            if state.pending_interrupts:
                with interrupt_container:
                    build_interrupt_approval()
        except Exception as e:
            print(f"[refresh_interrupt_ui] ignored error: {e}")

    # ─────────────────────────────────────────────
    # Resume / approval handling
    # ─────────────────────────────────────────────
    async def send_batch_resume(approvals: list) -> None:
        if state.busy or not state.thread_id or not approvals:
            return
        if not state.api_online:
            safe_notify("Backend is offline. Cannot resume.", type="warning")
            return

        state.busy = True
        update_ui_availability()

        interrupt_container.clear()
        with interrupt_container:
            with ui.row().classes("w-full items-center gap-3"):
                ui.spinner("dots", color="primary", size="2em")
                ui.label("Resuming execution…").classes("text-grey-7")

        pending_approvals = approvals
        resolved_choice_by_signature = {}
        auto_cycle = 0

        try:
            while pending_approvals:
                approved_ids = {
                    a.get("interrupt_id")
                    for a in pending_approvals
                    if a.get("interrupt_id")
                }

                for a in pending_approvals:
                    sig = a.get("signature")
                    if sig:
                        resolved_choice_by_signature[sig] = {
                            "decision": a.get("decision"),
                            "tool_args": a.get("tool_args"),
                            "feedback": a.get("feedback"),
                        }

                full_response = ""
                node_messages = []
                streamed_interrupts = []

                backend_approvals = []
                for a in pending_approvals:
                    backend_approvals.append(
                        {
                            "interrupt_id": a.get("interrupt_id"),
                            "decision": a.get("decision"),
                            "tool_args": a.get("tool_args"),
                            "feedback": a.get("feedback"),
                        }
                    )

                payload = {
                    "thread_id": state.thread_id,
                    "approvals": backend_approvals,
                }

                try:
                    async with httpx.AsyncClient(timeout=120) as client:
                        async with client.stream(
                            "POST",
                            f"{API_BASE}/approve/batch",
                            json=payload,
                        ) as response:
                            response.raise_for_status()

                            async for line in response.aiter_lines():
                                data = parse_sse_line(line)
                                if data is None:
                                    continue

                                etype = data.get("type")

                                if etype == "custom":
                                    evt = data.get("data", {})
                                    state.custom_events.append(evt)
                                    state.messages.append(
                                        {
                                            "role": "custom",
                                            "content": evt.get("status", str(evt)),
                                        }
                                    )

                                elif etype == "chunk":
                                    chunk_data = data.get("data", {})
                                    if isinstance(chunk_data, dict):
                                        for node_name in NODE_NAMES:
                                            node_out = chunk_data.get(node_name, {})
                                            if isinstance(node_out, dict):
                                                for m in (
                                                    node_out.get("messages", []) or []
                                                ):
                                                    if isinstance(m, dict) and m.get(
                                                        "content"
                                                    ):
                                                        content = str(m["content"])
                                                        if len(content) > 3:
                                                            node_messages.append(
                                                                {
                                                                    "role": "node",
                                                                    "node": node_name,
                                                                    "content": content[
                                                                        :400
                                                                    ]
                                                                    + "..."
                                                                    if len(content)
                                                                    > 400
                                                                    else content,
                                                                }
                                                            )

                                        finalizer = chunk_data.get("finalizer", {})
                                        if isinstance(
                                            finalizer, dict
                                        ) and finalizer.get("final_response"):
                                            full_response = finalizer["final_response"]

                                elif etype == "interrupt":
                                    streamed_interrupts = merge_interrupts(
                                        streamed_interrupts,
                                        data.get("payload", []),
                                    )
                                    state.interrupt_cycle += 1

                                elif etype == "awaiting_approval":
                                    streamed_interrupts = merge_interrupts(
                                        streamed_interrupts,
                                        data.get("interrupts", []),
                                    )
                                    state.interrupt_cycle = data.get(
                                        "cycle",
                                        state.interrupt_cycle + 1,
                                    )

                                elif etype == "done":
                                    break

                except (
                    httpx.ConnectError,
                    httpx.ReadTimeout,
                    httpx.RemoteProtocolError,
                ) as e:
                    safe_notify(
                        "Connection to backend lost while resuming. "
                        "UI stays open – restart the LangGraph server and try again.",
                        type="warning",
                    )
                    state.api_online = False
                    break
                except Exception as e:
                    safe_notify(f"Resume stream error: {e}", type="negative")
                    state.api_online = False
                    break

                # Add collected messages
                for nm in node_messages:
                    state.messages.append(nm)

                if full_response:
                    state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                        }
                    )

                # Reconcile remaining interrupts
                streamed_remaining = [
                    i
                    for i in normalize_interrupts(streamed_interrupts)
                    if i.get("id") not in approved_ids
                    and i.get("id") not in state.resolved_interrupt_ids
                ]

                ok, remote = await fetch_authoritative_interrupts()

                if ok:
                    remote_remaining = [
                        i
                        for i in remote
                        if i.get("id") not in approved_ids
                        and i.get("id") not in state.resolved_interrupt_ids
                    ]
                    remaining = (
                        remote_remaining if remote_remaining else streamed_remaining
                    )
                else:
                    remaining = streamed_remaining
                    state.api_online = False

                # Detect repeated identical interrupts
                repeated = []
                new_items = []

                for item in remaining:
                    sig = item.get("signature")
                    val = item.get("value") or {}
                    is_final_fb = isinstance(val, dict) and val.get("type") in (
                        "final_feedback",
                        "finalizer_feedback",
                    )

                    if (
                        not is_final_fb
                        and sig in resolved_choice_by_signature
                        and auto_cycle < MAX_AUTO_REPEAT_IDENTICAL_INTERRUPTS
                    ):
                        repeated.append(item)
                    else:
                        new_items.append(item)

                if repeated:
                    next_approvals = []

                    for item in repeated:
                        choice = resolved_choice_by_signature.get(
                            item.get("signature"), {}
                        )
                        ids = item.get("ids") or [item.get("id")]

                        for iid in ids:
                            next_approvals.append(
                                {
                                    "interrupt_id": iid,
                                    "decision": choice.get("decision", "Y"),
                                    "tool_args": choice.get("tool_args"),
                                    "feedback": choice.get("feedback"),
                                    "signature": item.get("signature"),
                                }
                            )

                    auto_cycle += 1
                    state.resolved_interrupt_ids.update(approved_ids)

                    safe_notify(
                        "Duplicate identical tool request detected. "
                        f"Auto-resolving repeat #{auto_cycle}.",
                        type="warning",
                    )

                    pending_approvals = next_approvals
                    continue

                # Finalize
                state.resolved_interrupt_ids.update(approved_ids)
                state.decisions = {}
                state.pending_interrupts = remaining
                break

            rebuild_chat()
            refresh_interrupt_ui()
            refresh_side_panels()

            if state.pending_interrupts:
                safe_notify(
                    f"⚠️ {len(state.pending_interrupts)} more tool(s) need approval "
                    f"(cycle #{state.interrupt_cycle})",
                    type="warning",
                )

        except Exception as e:
            safe_notify(f"Resume failed: {e}", type="negative")
            refresh_interrupt_ui()
            state.api_online = False

        finally:
            state.busy = False
            update_ui_availability()

    # ─────────────────────────────────────────────
    # Chat send
    # ─────────────────────────────────────────────
    async def send_message() -> None:
        prompt = (msg_input.value or "").strip()
        if not prompt:
            return

        if state.busy:
            safe_notify(
                "Please wait for the current operation to finish.", type="warning"
            )
            return

        if not state.api_online:
            safe_notify(
                "Backend is offline. Start the LangGraph server first.", type="warning"
            )
            return

        if not state.thread_id:
            safe_notify("Create a thread first!", type="warning")
            return

        if state.pending_interrupts:
            safe_notify(
                "Resolve pending tool approvals before sending a new message.",
                type="warning",
            )
            return

        state.busy = True
        msg_input.value = ""
        update_ui_availability()

        for m in state.messages:
            m.pop("has_interrupt_after", None)

        add_message({"role": "user", "content": prompt})

        with chat_container:
            with ui.row().classes("w-full items-start gap-2 no-wrap"):
                ui.html('<span class="avatar-emoji">🤖</span>')
                with ui.element("div").classes("assistant-msg"):
                    placeholder = ui.markdown("▌")

        scroll_to_bottom()

        full_response = ""

        try:
            payload = {
                "message": prompt,
                "thread_id": state.thread_id,
                "max_iterations": int(state.max_iterations),
            }

            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream(
                        "POST",
                        f"{API_BASE}/chat/stream",
                        json=payload,
                        headers={"Accept": "text/event-stream"},
                    ) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            data = parse_sse_line(line)
                            if data is None:
                                continue

                            etype = data.get("type")

                            if etype == "thread_id":
                                if data.get("thread_id"):
                                    state.thread_id = data["thread_id"]

                            elif etype == "custom":
                                evt = data.get("data", {})
                                state.custom_events.append(evt)
                                msg = {
                                    "role": "custom",
                                    "content": evt.get("status", str(evt)),
                                }
                                state.messages.append(msg)
                                try:
                                    with chat_container:
                                        render_message(msg)
                                    scroll_to_bottom()
                                except Exception:
                                    pass
                            elif etype == "chunk":
                                chunk_data = data.get("data", {})

                                if isinstance(chunk_data, dict):
                                    for node_name in NODE_NAMES:
                                        node_out = chunk_data.get(node_name, {})
                                        if isinstance(node_out, dict):
                                            for m in node_out.get("messages", []) or []:
                                                if isinstance(m, dict) and m.get(
                                                    "content"
                                                ):
                                                    content = str(m["content"])
                                                    if len(content) > 3:
                                                        add_message(
                                                            {
                                                                "role": "node",
                                                                "node": node_name,
                                                                "content": content[:400]
                                                                + "..."
                                                                if len(content) > 400
                                                                else content,
                                                            }
                                                        )

                                    finalizer = chunk_data.get("finalizer", {})
                                    if isinstance(finalizer, dict) and finalizer.get(
                                        "final_response"
                                    ):
                                        full_response = finalizer["final_response"]
                                        placeholder.set_content(f"{full_response}▌")

                                    agent_out = chunk_data.get("agent", {})
                                    if isinstance(agent_out, dict):
                                        msgs = agent_out.get("messages", [])
                                        if isinstance(msgs, list) and msgs:
                                            last = msgs[-1]
                                            if (
                                                isinstance(last, dict)
                                                and last.get("role") == "assistant"
                                                and last.get("content")
                                            ):
                                                full_response = last["content"]
                                                placeholder.set_content(
                                                    f"{full_response}▌"
                                                )

                                state.events.append(
                                    {
                                        "event": data.get("event", "unknown"),
                                        "data": chunk_data,
                                        "time": datetime.now().isoformat(),
                                    }
                                )

                            elif etype == "interrupt":
                                incoming = normalize_interrupts(data.get("payload", []))

                                ok, remote = await fetch_authoritative_interrupts()
                                if ok:
                                    if remote:
                                        incoming = merge_interrupts(remote, incoming)
                                    else:
                                        incoming = merge_interrupts(incoming)
                                else:
                                    incoming = merge_interrupts(incoming)

                                incoming = filter_resolved_interrupts(incoming)

                                if incoming:
                                    state.pending_interrupts = incoming
                                    state.decisions = {}
                                    state.interrupt_cycle += 1

                                    if state.messages:
                                        state.messages[-1]["has_interrupt_after"] = True

                                    refresh_interrupt_ui()
                                else:
                                    state.pending_interrupts = []
                                    refresh_interrupt_ui()

                            elif etype == "awaiting_approval":
                                incoming = normalize_interrupts(
                                    data.get("interrupts", [])
                                )
                                count = data.get("count", len(incoming))

                                ok, remote = await fetch_authoritative_interrupts()
                                if ok and len(remote) >= count:
                                    incoming = merge_interrupts(remote, incoming)
                                else:
                                    incoming = merge_interrupts(incoming)

                                incoming = filter_resolved_interrupts(incoming)

                                state.interrupt_cycle = data.get(
                                    "cycle",
                                    state.interrupt_cycle + 1,
                                )

                                if incoming:
                                    state.pending_interrupts = incoming
                                    state.decisions = {}

                                    if state.messages:
                                        state.messages[-1]["has_interrupt_after"] = True

                                    refresh_interrupt_ui()
                                else:
                                    state.pending_interrupts = []
                                    refresh_interrupt_ui()

                            elif etype == "done":
                                break

                            elif etype == "error":
                                safe_notify(
                                    f"Stream error: {data.get('message')}",
                                    type="negative",
                                )
                                break
                            elif etype == "final_state":
                                # Optional: cache the final state for the inspector
                                state.final_state_snapshot = data.get("state", {})
                                continue

            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ) as e:
                safe_notify(
                    "Connection to backend lost mid-stream. "
                    "UI stays open – restart the LangGraph server and try again.",
                    type="warning",
                )
                state.api_online = False
                try:
                    placeholder.delete()
                except Exception:
                    pass
                return

            # Final reconciliation
            if state.pending_interrupts:
                ok, remote = await fetch_authoritative_interrupts()
                if ok:
                    state.pending_interrupts = filter_resolved_interrupts(remote)
                    state.decisions = {}
                    refresh_interrupt_ui()

            if full_response and not state.pending_interrupts:
                placeholder.set_content(full_response)
                state.messages.append({"role": "assistant", "content": full_response})
            else:
                try:
                    placeholder.delete()
                except Exception:
                    pass

        except Exception as e:
            safe_notify(f"Streaming failed: {e}", type="negative")
            state.api_online = False

            try:
                placeholder.delete()
            except Exception:
                pass

            # Fallback to non-streaming endpoint
            try:
                safe_notify("Falling back to non-streaming endpoint…")

                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{API_BASE}/chat",
                        json={
                            "message": prompt,
                            "thread_id": state.thread_id,
                            "max_iterations": int(state.max_iterations),
                        },
                    )
                    resp.raise_for_status()
                    r = resp.json()

                full_response = r.get("response", "No response")

                if r.get("interrupted"):
                    interrupts = normalize_interrupts(r.get("pending_interrupts", []))

                    ok, remote = await fetch_authoritative_interrupts()
                    if ok and remote:
                        interrupts = merge_interrupts(remote, interrupts)

                    interrupts = filter_resolved_interrupts(interrupts)

                    state.pending_interrupts = interrupts
                    state.decisions = {}
                    state.interrupt_cycle += 1

                    if state.messages:
                        state.messages[-1]["has_interrupt_after"] = True

                    refresh_interrupt_ui()

                state.events.extend(r.get("events", []))

                if full_response and not state.pending_interrupts:
                    add_message({"role": "assistant", "content": full_response})

            except Exception as e2:
                safe_notify(f"Fallback also failed: {e2}", type="negative")
                state.api_online = False

        finally:
            state.busy = False
            update_ui_availability()
            refresh_side_panels()
            scroll_to_bottom()

    # ─────────────────────────────────────────────
    # Thread / file / sidebar actions
    # ─────────────────────────────────────────────
    async def create_new_thread() -> None:
        if state.busy:
            safe_notify(
                "Please wait for the current operation to finish.", type="warning"
            )
            return
        if not state.api_online:
            safe_notify(
                "Backend is offline. Start the LangGraph server first.", type="warning"
            )
            return

        state.busy = True
        update_ui_availability()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(f"{API_BASE}/threads/create")
                resp.raise_for_status()
                state.thread_id = resp.json()["thread_id"]

            state.messages = []
            state.events = []
            state.pending_interrupts = []
            state.custom_events = []
            state.uploaded_files = []
            state.interrupt_cycle = 0
            state.decisions = {}
            state.resolved_interrupt_ids = set()
            state.uploaded_file_keys = set()

            rebuild_chat()
            refresh_interrupt_ui()
            refresh_side_panels()
            await rebuild_thread_list()

            safe_notify("New thread created", type="positive")

        except Exception as e:
            safe_notify(f"Failed to create thread: {e}", type="negative")
            state.api_online = False
        finally:
            state.busy = False
            update_ui_availability()

    async def delete_thread() -> None:
        if state.busy or not state.thread_id:
            return
        if not state.api_online:
            safe_notify("Backend is offline.", type="warning")
            return

        state.busy = True
        update_ui_availability()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.delete(f"{API_BASE}/threads/{state.thread_id}")

            state.thread_id = None
            state.messages = []
            state.pending_interrupts = []
            state.decisions = {}
            state.interrupt_cycle = 0
            state.resolved_interrupt_ids = set()
            state.uploaded_file_keys = set()

            rebuild_chat()
            refresh_interrupt_ui()
            refresh_side_panels()
            await rebuild_thread_list()

            safe_notify("Thread deleted", type="positive")

        except Exception as e:
            safe_notify(f"Failed to delete thread: {e}", type="negative")
            state.api_online = False
        finally:
            state.busy = False
            update_ui_availability()

    async def load_thread(tid: str) -> None:
        if state.busy:
            safe_notify(
                "Please wait for the current operation to finish.", type="warning"
            )
            return
        if not state.api_online:
            safe_notify("Backend is offline.", type="warning")
            return

        state.busy = True
        update_ui_availability()

        state.thread_id = tid
        state.messages = []
        state.decisions = {}
        state.interrupt_cycle = 0
        state.resolved_interrupt_ids = set()
        state.uploaded_file_keys = set()

        try:
            ok, current = await fetch_authoritative_interrupts()
            state.pending_interrupts = filter_resolved_interrupts(current) if ok else []

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{API_BASE}/threads/{tid}/history")
                resp.raise_for_status()
                history = resp.json().get("history", [])

            for h in history:
                for m in (h.get("values", {}) or {}).get("messages", []):
                    if isinstance(m, dict) and m.get("content"):
                        state.messages.append(
                            {
                                "role": m.get("role", "user"),
                                "content": str(m.get("content", "")),
                            }
                        )
        except Exception as e:
            safe_notify(f"Could not load full history: {e}", type="warning")
            # still keep the thread_id so user can continue

        rebuild_chat()
        refresh_interrupt_ui()
        refresh_side_panels()

        safe_notify(f"Loaded thread {tid[:12]}…", type="positive")

        state.busy = False
        update_ui_availability()

    async def handle_upload(e) -> None:
        if state.busy:
            safe_notify(
                "Please wait for the current operation to finish.", type="warning"
            )
            return
        if not state.api_online:
            safe_notify("Backend is offline. Cannot upload.", type="warning")
            return
        if not state.thread_id:
            safe_notify("Create a thread first!", type="warning")
            return
        if state.pending_interrupts:
            safe_notify(
                "Resolve pending tool approvals before uploading another file.",
                type="warning",
            )
            return

        state.busy = True
        update_ui_availability()

        try:
            raw_bytes = await e.file.read()
            filename = e.file.name
            content_type = e.file.content_type or "application/octet-stream"

            if not raw_bytes:
                safe_notify("Uploaded file is empty!", type="warning")
                return

            file_key = f"{filename}:{len(raw_bytes)}"
            if file_key in state.uploaded_file_keys:
                safe_notify(
                    "This file was already uploaded in this session. "
                    "Duplicate uploads can cause repeated tool interrupts.",
                    type="warning",
                )
                return

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{API_BASE}/upload",
                    files={"file": (filename, raw_bytes, content_type)},
                    data={"thread_id": state.thread_id},
                )

            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                safe_notify(f"Upload failed: {detail}", type="negative")
                return

            r = resp.json()

            state.uploaded_file_keys.add(file_key)
            state.uploaded_files.append(
                {
                    "name": r.get("filename", filename),
                    "type": r.get("file_type", content_type),
                    "size": r.get("size", len(raw_bytes)),
                    "path": r.get("path", ""),
                }
            )

            # Re-sync interrupts after upload (side-effect on graph state)
            ok, current = await fetch_authoritative_interrupts()
            if ok:
                state.pending_interrupts = filter_resolved_interrupts(current)
                state.decisions = {}
                refresh_interrupt_ui()

            safe_notify(
                f"Uploaded & linked to thread: {r.get('filename', filename)}",
                type="positive",
            )

        except Exception as ex:
            safe_notify(f"Upload request failed: {ex}", type="negative")
            state.api_online = False
        finally:
            state.busy = False
            update_ui_availability()
            refresh_side_panels()

    # ─────────────────────────────────────────────
    # Health check – never crashes the UI
    # ─────────────────────────────────────────────
    async def check_health() -> None:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{API_BASE}/health")
                resp.raise_for_status()
                _ = resp.json()

            # Backend is reachable
            was_offline = not state.api_online
            state.api_online = True
            health_label.text = "● API Online"
            health_label.classes(
                "text-sm text-green-600 font-semibold",
                replace="text-sm text-red-600 font-semibold",
            )
            if was_offline:
                safe_notify("Backend is back online 🎉", type="positive")

        except Exception as e:
            was_online = state.api_online
            state.api_online = False
            health_label.text = "● API Offline"
            health_label.classes(
                "text-sm text-red-600 font-semibold",
                replace="text-sm text-green-600 font-semibold",
            )
            if was_online:
                safe_notify(
                    f"Backend offline ({type(e).__name__}). "
                    "UI stays alive – restart the LangGraph server when ready.",
                    type="warning",
                )
        finally:
            update_ui_availability()

    # ─────────────────────────────────────────────
    # Sidebar / panel rebuild functions
    # ─────────────────────────────────────────────
    def rebuild_active_thread_section() -> None:
        active_thread_container.clear()

        with active_thread_container:
            if state.thread_id:
                ui.markdown(f"**Active:** `{state.thread_id[:16]}...`").classes(
                    "text-sm"
                )
                btn = (
                    ui.button("🗑️ Delete Thread", on_click=delete_thread)
                    .props("color=negative outline unelevated")
                    .classes("w-full")
                )
                if not state.api_online or state.busy:
                    btn.disable()
            else:
                ui.label("No active thread").classes("text-sm text-orange-600")

    def rebuild_upload_section() -> None:
        upload_container.clear()

        with upload_container:
            if state.thread_id:
                ui.label("Upload file for agent").classes("text-xs text-grey-7")

                up = (
                    ui.upload(auto_upload=True, on_upload=handle_upload)
                    .props(f'accept="{ACCEPTED_FILES}"')
                    .classes("w-full")
                )

                if state.busy or state.pending_interrupts or not state.api_online:
                    up.disable()

                if not state.api_online:
                    ui.label("Backend offline – upload disabled").classes(
                        "text-xs text-red-7"
                    )
                elif state.pending_interrupts:
                    ui.label(
                        "Resolve pending tool approvals before uploading."
                    ).classes("text-xs text-orange-7")

                if state.uploaded_files:
                    ui.label("Attached files:").classes("text-xs text-grey-7 pt-1")
                    for f in state.uploaded_files:
                        ui.html(
                            f'<span class="file-badge">📄 {escape(f["name"])}</span>'
                        )
            else:
                ui.label("Create a thread first to upload files").classes(
                    "text-xs text-grey-7"
                )

    async def rebuild_thread_list() -> None:
        thread_list_container.clear()

        with thread_list_container:
            ui.separator()

            with ui.row().classes("w-full items-center justify-between no-wrap"):
                ui.label("📋 Recent Threads").classes("text-lg font-semibold")
                ui.button(icon="refresh", on_click=rebuild_thread_list).props(
                    "dense flat round color=grey-8"
                )

            if not state.api_online:
                ui.label("Backend offline – cannot list threads").classes(
                    "text-xs text-red-7"
                )
                return

            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"{API_BASE}/threads/list",
                        params={"limit": 5},
                    )
                    resp.raise_for_status()
                    threads = resp.json().get("threads", [])
            except Exception as e:
                ui.label(f"Unable to load: {e}").classes("text-xs text-grey-7")
                state.api_online = False
                return

            if not threads:
                ui.label("No threads yet").classes("text-xs text-grey-7")

            for t in threads:
                tid = t.get("thread_id", "unknown")

                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label(f"{tid[:12]}...").classes("font-mono text-xs flex-grow")

                    async def on_load(tid=tid):
                        await load_thread(tid)

                    btn = ui.button("Load", on_click=on_load).props(
                        "dense outline color=primary unelevated"
                    )
                    if state.busy:
                        btn.disable()

    def rebuild_custom_events_panel() -> None:
        custom_events_container.clear()

        if not state.custom_events:
            return

        with custom_events_container:
            with ui.expansion("📡 Agent Status Updates", icon="podcasts").props(
                "default-opened"
            ):
                for evt in state.custom_events[-10:]:
                    status = evt.get("status", str(evt))
                    ui.html(
                        f'<div class="node-msg"><b>Update</b>: {escape(str(status))}</div>'
                    )

    async def load_state_snapshot(e) -> None:
        if not e.value:
            return

        if not state.thread_id:
            state_json_view.set_content("*No active thread.*")
            return

        if not state.api_online:
            state_json_view.set_content("*Backend offline – cannot fetch state.*")
            return

        try:
            async with httpx.AsyncClient(timeout=6) as client:
                resp = await client.get(f"{API_BASE}/threads/{state.thread_id}/state")
                resp.raise_for_status()
                snapshot = resp.json()

            state_json_view.set_content(
                f"```json\n{json.dumps(snapshot.get('state', {}), indent=2, default=str)}\n```"
            )
        except Exception as ex:
            state_json_view.set_content(f"⚠️ Could not fetch state: `{escape(str(ex))}`")
            state.api_online = False

    def rebuild_events_panel() -> None:
        events_panel_container.clear()

        if not state.show_events or not state.events:
            return

        with events_panel_container:
            ui.separator()
            ui.label("📡 Raw Events Stream").classes("text-lg font-semibold")

            event_types = sorted({str(e.get("event", "unknown")) for e in state.events})

            if not state.event_filter:
                state.event_filter = event_types[:]

            selected = [t for t in state.event_filter if t in event_types]

            def on_filter_change(e) -> None:
                state.event_filter = list(e.value or [])
                rebuild_events_panel()

            ui.select(
                event_types,
                multiple=True,
                label="Filter by event type",
                value=selected,
                on_value_change=on_filter_change,
            ).classes("w-full")

            filtered = [
                e
                for e in state.events
                if str(e.get("event", "unknown")) in set(selected)
            ]

            ui.label(f"Showing {len(filtered)} of {len(state.events)} events").classes(
                "text-xs text-grey-7"
            )

            for evt in filtered[-20:]:
                ui.markdown(
                    f"**Event:** `{evt.get('event', 'N/A')}` | "
                    f"**Time:** `{evt.get('time', 'N/A')}`"
                )
                ui.markdown(
                    f"```json\n{json.dumps(evt.get('data', {}), indent=2, default=str)}\n```"
                )
                ui.separator()

    def refresh_side_panels() -> None:
        rebuild_active_thread_section()
        rebuild_upload_section()
        rebuild_custom_events_panel()
        rebuild_events_panel()

    async def on_enter(e) -> None:
        await send_message()

    # ─────────────────────────────────────────────
    # Layout — header
    # ─────────────────────────────────────────────
    left_drawer = (
        ui.left_drawer(value=True, bordered=False)
        .props("width=330 breakpoint=768")
        .classes("glass-drawer")
    )

    with ui.header().classes("items-center gap-2"):
        # Sidebar toggle — collapses/expands the drawer
        ui.button(icon="sym_o_menu").props("flat dense round").on_click(
            lambda: left_drawer.toggle()
        ).tooltip("Toggle sidebar (Ctrl+B)")
        ui.label("NEXUS//AGENT").classes("main-header")

        # Online / offline status pill
        status_html = ui.html('<span class="status-dot online">ONLINE</span>')

        # Keyboard shortcut Ctrl+B
        ui.keyboard(
            on_key=lambda e: left_drawer.toggle() if e.key == "b" and e.ctrl else None
        )

    # ─────────────────────────────────────────────
    # Layout — sidebar
    # ─────────────────────────────────────────────
    with left_drawer:
        with ui.row().classes("w-full items-center justify-between q-pa-xs"):
            ui.html('<span class="side-label">// CONTROL PANEL</span>')
            ui.button(
                icon="sym_o_chevron_left", on_click=lambda: left_drawer.hide()
            ).props("flat dense round size=sm").tooltip(
                "Hide sidebar (Ctrl+B to restore)"
            )

        with ui.column().classes("w-full gap-2 q-pa-sm"):
            ui.html(
                '<p class="main-header" style="font-size:1.05rem;">🤖 LangGraph Multi-HITL</p>'
            )
            ui.separator()

            health_label = ui.label("● checking API…").classes(
                "text-sm text-grey-6 font-semibold"
            )

            ui.separator()
            ui.html('<span class="side-label">// Thread Management</span>')

            new_thread_btn = (
                ui.button("➕ New Thread", on_click=create_new_thread)
                .props("color=primary unelevated")
                .classes("w-full")
            )

            active_thread_container = ui.column().classes("w-full gap-2")

            ui.separator()
            ui.html('<span class="side-label">// File Upload</span>')

            upload_container = ui.column().classes("w-full gap-1")
            thread_list_container = ui.column().classes("w-full gap-1")

            ui.separator()

            ui.switch("🔍 Show Raw Events").bind_value(
                state, "show_events"
            ).on_value_change(lambda e: rebuild_events_panel())

            ui.html('<span class="side-label">// Max Iterations</span>')
            ui.slider(min=1, max=20, step=1, value=state.max_iterations).bind_value(
                state, "max_iterations"
            )

    # ─────────────────────────────────────────────
    # Layout — main chat area
    # ─────────────────────────────────────────────
    with ui.column().classes("w-full max-w-3xl mx-auto gap-2 p-4 items-stretch"):
        ui.html(
            '<p class="main-header" style="margin-bottom:0.6rem;">💬 LangGraph Chat with Multi-HITL</p>'
        )

        custom_events_container = ui.column().classes("w-full")

        with ui.expansion("🔎 Thread State Inspector", icon="search").classes(
            "w-full"
        ) as state_expansion:
            state_json_view = ui.markdown("*Open to load thread state…*")

        state_expansion.on_value_change(load_state_snapshot)

        chat_container = ui.column().classes("w-full gap-1")
        interrupt_container = ui.column().classes("w-full gap-2")
        events_panel_container = ui.column().classes("w-full gap-1")

        ui.separator()
        ui.html(
            '🔗 <a href="http://localhost:8008/docs" target="_blank">API Docs</a>'
            " | Built with FastAPI + NiceGUI + LangGraph Multi-HITL"
        ).classes("text-xs text-grey-6")

    # ─────────────────────────────────────────────
    # Footer chat input
    # ─────────────────────────────────────────────
    with ui.footer():
        with ui.row().classes("w-full max-w-3xl mx-auto items-center gap-2 no-wrap"):
            ui.html('<span class="avatar-emoji">🤖</span>')

            msg_input = (
                ui.input(placeholder="Type your message…")
                .props("outlined dense rounded")
                .classes("flex-grow")
            )

            send_btn = ui.button(icon="send", on_click=send_message).props(
                "round dense unelevated color=primary"
            )

    msg_input.on("keydown.enter", on_enter)

    # ─────────────────────────────────────────────
    # Initial render + resilient startup
    # ─────────────────────────────────────────────
    rebuild_active_thread_section()
    rebuild_upload_section()
    rebuild_custom_events_panel()
    rebuild_events_panel()

    # Health check runs forever – never crashes the page
    ui.timer(15.0, check_health)

    # One-shot initial setup that tolerates a missing backend
    async def initial_setup():
        await check_health()
        if state.api_online:
            await rebuild_thread_list()
            # Recover any pending interrupts that survived a previous session
            if state.thread_id and not state.pending_interrupts:
                ok, recovered = await fetch_authoritative_interrupts()
                if ok and recovered:
                    state.pending_interrupts = filter_resolved_interrupts(recovered)
                    state.interrupt_cycle += 1
                    refresh_interrupt_ui()
        else:
            with chat_container:
                ui.label(
                    "⚠️ Backend is currently offline.\n"
                    "Start the LangGraph server (port 8008) – the UI will recover automatically."
                ).classes("text-orange-700 whitespace-pre-wrap")

        update_ui_availability()

    ui.timer(0.2, initial_setup, once=True)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="LangGraph Multi-HITL",
        favicon="🤖",
        port=8080,
        reload=False,
        show=False,
    )
