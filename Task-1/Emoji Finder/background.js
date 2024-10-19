chrome.action.onClicked.addListener((tab) => {
  chrome.tabs.sendMessage(tab.id, { action: "replaceWithEmojis" });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getEmojiCount") {
    chrome.storage.sync.get({ emojiCount: 5 }, (data) => {
      sendResponse({ emojiCount: data.emojiCount });
    });
    return true;
  }
});
