---
name: Main
base: none
---
```css template
/* Ambient undulating background */
.ambient-backdrop {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
  z-index: 0;
}

.ambient-blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(120px);
  opacity: 0.5;
  will-change: transform;
}

.ambient-blob-1 {
  top: -20%;
  left: 15%;
  width: 950px;
  height: 850px;
  background: radial-gradient(circle, rgba(49, 46, 129, 0.85) 0%, rgba(30, 27, 75, 0.45) 50%, transparent 70%);
  animation: drift-1 14s ease-in-out infinite alternate;
}

.ambient-blob-2 {
  bottom: -25%;
  right: 10%;
  width: 1100px;
  height: 950px;
  background: radial-gradient(circle, rgba(67, 56, 202, 0.65) 0%, rgba(30, 27, 75, 0.35) 50%, transparent 70%);
  animation: drift-2 18s ease-in-out infinite alternate-reverse;
}

@keyframes drift-1 {
  0%   { transform: translate(0, 0) scale(1); }
  50%  { transform: translate(160px, 90px) scale(1.18) rotate(12deg); }
  100% { transform: translate(-100px, 150px) scale(0.92); }
}

@keyframes drift-2 {
  0%   { transform: translate(0, 0) scale(1); }
  50%  { transform: translate(-140px, -80px) scale(1.15) rotate(-10deg); }
  100% { transform: translate(110px, -130px) scale(0.9); }
}

/* Ensure slide content sits cleanly above the background */
.slide-layout {
  position: relative;
  z-index: 1;
}

.deck-chrome-footer {
  position: absolute;
  bottom: 1.25rem;
  left: 2.5rem;
  right: 2.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
  color: var(--text-muted, #94a3b8);
  pointer-events: none;
  z-index: 10;
}
.deck-chrome-footer .deck-brand-left,
.deck-chrome-footer .deck-brand-right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.deck-chrome-footer img {
  pointer-events: auto;
}

.fade-edges {
  mask-image:
    linear-gradient(to right, transparent, black 10%, black 80%, transparent),
    linear-gradient(to bottom, transparent, black 10%, black 80%, transparent);
  mask-composite: intersect;
  -webkit-mask-image:
    linear-gradient(to right, transparent, black 10%, black 80%, transparent),
    linear-gradient(to bottom, transparent, black 10%, black 80%, transparent);
  -webkit-mask-composite: source-in;
}

.caption {
  font-size: 0.8rem;
  font-style: italic;
  color: var(--text-muted, #94a3b8);
}

.logo-name {
  font-size: 9rem;
  font-family: Verdana, Arial, sans-serif;
  color: #606060;
}
```

<div class="ambient-backdrop">
  <div class="ambient-blob ambient-blob-1"></div>
  <div class="ambient-blob ambient-blob-2"></div>
</div>

<slot />

<footer class="deck-chrome-footer">
  <div class="deck-brand-left">
    <img src="../../docs/logo.svg" alt="Channel" height="20" />
    <span>Channel Inference Engine</span>
  </div>
  <div class="deck-brand-right">
    <span>© 2026 Sullux LLC</span>
    <img src="https://sullux.com/images/logo-full-email-dark.svg" height="18" />
  </div>
</footer>
