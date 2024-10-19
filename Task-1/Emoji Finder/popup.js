document.addEventListener('DOMContentLoaded', () => {
  const emojiCountInput = document.getElementById('emojiCount');
  const applyButton = document.getElementById('applyButton');

  chrome.storage.sync.get({ emojiCount: 5 }, (data) => {
    emojiCountInput.value = data.emojiCount;
  });

  applyButton.addEventListener('click', () => {
    const emojiCount = parseInt(emojiCountInput.value);
    chrome.storage.sync.set({ emojiCount: emojiCount }, () => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { action: "replaceWithEmojis" });
      });
      window.close();
    });
  });
});
