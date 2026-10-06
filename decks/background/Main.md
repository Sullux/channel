---
name: Main
base: none
---
```css template
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
.deck-chrome-footer .deck-brand-left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.deck-chrome-footer .deck-brand-right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.deck-chrome-footer img {
  pointer-events: auto;
}
```

<slot />

<footer class="deck-chrome-footer">
  <div class="deck-brand-left">
    <img src="../../docs/logo.svg" alt="Channel" height="20" />
    <span>Channel Inference Engine</span>
  </div>
  <div class="deck-brand-right">
    <span>© 2026 Sullux LLC</span>
    <img src="https://sullux.com/images/logo-full-email-dark.svg" alt="Sullux" height="18" />
  </div>
</footer>
