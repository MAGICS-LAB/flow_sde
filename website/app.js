"use strict";
const button = document.getElementById("copy-citation");
button.addEventListener("click", async () => {
  const citation = document.getElementById("citation-text");
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(citation.textContent);
    button.textContent = "Copied";
    status.textContent = "BibTeX copied to clipboard.";
    window.setTimeout(() => { button.textContent = "Copy BibTeX"; }, 2200);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Citation selected. Use your browser's copy command.";
  }
});
