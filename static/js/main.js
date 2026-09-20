'use strict';
const menu = document.querySelector('.nav-menu');
menu.querySelectorAll('a').forEach(link => link.addEventListener('click', () => { menu.open = false; }));
document.addEventListener('click', event => { if (!menu.contains(event.target)) menu.open = false; });
document.addEventListener('keydown', event => { if (event.key === 'Escape' && menu.open) { menu.open = false; menu.querySelector('summary').focus(); } });
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
