let isHighlightingEnabled = false;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "toggleHighlight") {
    isHighlightingEnabled = !isHighlightingEnabled;
    if (isHighlightingEnabled) {
      document.addEventListener("mouseup", highlightSelection);
    } else {
      document.removeEventListener("mouseup", highlightSelection);
    }
  }
});

function highlightSelection() {
  const selection = window.getSelection();
  if (selection.rangeCount > 0) {
    const range = selection.getRangeAt(0);
    const span = document.createElement("span");
    span.className = "text-highlighter-highlight";
    range.surroundContents(span);
  }
}
