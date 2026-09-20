'use strict';
document.documentElement.classList.add('js');
const toggle = document.querySelector('.menu-toggle');
const links = document.querySelector('.nav-links');
function closeMenu() { links.classList.remove('open'); toggle.setAttribute('aria-expanded', 'false'); }
toggle.addEventListener('click', () => { const open = links.classList.toggle('open'); toggle.setAttribute('aria-expanded', String(open)); });
links.addEventListener('click', event => { if (event.target.closest('a')) closeMenu(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape' && links.classList.contains('open')) { closeMenu(); toggle.focus(); } });
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      links.querySelectorAll('a').forEach(a => { if (a.hash === '#' + entry.target.id) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current'); });
    }
  }, { rootMargin: '-10% 0px -65% 0px', threshold: 0 });
  document.querySelectorAll('main section[id]').forEach(section => observer.observe(section));
}
const dialog = document.querySelector('#image-dialog');
const enlarged = document.querySelector('#dialog-image');
if (typeof dialog.showModal === 'function') {
  document.querySelectorAll('.image-zoom').forEach(button => button.addEventListener('click', () => {
    const source = button.querySelector('img');
    enlarged.src = source.currentSrc || source.src;
    enlarged.alt = source.alt;
    document.querySelector('#dialog-caption').textContent = source.alt;
    dialog.showModal();
  }));
  dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
} else {
  document.querySelectorAll('.image-zoom').forEach(button => button.addEventListener('click', () => { window.location.href = button.querySelector('img').src; }));
}
const bibtex = document.querySelector('#bibtex');
const copy = document.querySelector('#copy-bibtex');
copy.disabled = bibtex.dataset.available !== 'true';
copy.addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try { await navigator.clipboard.writeText(bibtex.textContent.trim()); status.textContent = 'BibTeX copied.'; copy.textContent = 'Copied!'; setTimeout(() => { copy.textContent = 'Copy BibTeX'; }, 2000); }
  catch { const range = document.createRange(); range.selectNodeContents(bibtex); const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range); status.textContent = 'Copy unavailable. Citation selected; press Control+C or Command+C.'; }
});
