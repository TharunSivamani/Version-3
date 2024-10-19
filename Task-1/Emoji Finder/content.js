const emojiMap = {
  "happy": ["😊", "😃", "😄", "😁"],
  "sad": ["😢", "😭", "😞", "☹️"],
  "love": ["❤️", "😍", "🥰", "💖"],
  "angry": ["😠", "😡", "🤬", "💢"],
  "surprised": ["😮", "😲", "😯", "🙀"],
  "food": ["🍎", "🍕", "🍦", "🍫"],
  "animal": ["🐶", "🐱", "🐼", "🦁"],
  "nature": ["🌳", "🌺", "🌈", "☀️"],
  "sport": ["⚽", "🏀", "🎾", "🏈"],
  "travel": ["✈️", "🚗", "🏖️", "🗺️"]
};

function getRandomEmoji(word) {
  const lowercaseWord = word.toLowerCase();
  for (const [category, emojis] of Object.entries(emojiMap)) {
    if (lowercaseWord.includes(category)) {
      return emojis[Math.floor(Math.random() * emojis.length)];
    }
  }
  return "😀"; // Default emoji if no match found
}

function replaceWithEmojis() {
  chrome.runtime.sendMessage({ action: "getEmojiCount" }, (response) => {
    const emojiCount = response.emojiCount;
    const textNodes = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);

    while (walker.nextNode()) {
      if (walker.currentNode.textContent.trim().length > 0) {
        textNodes.push(walker.currentNode);
      }
    }

    const selectedNodes = textNodes
      .sort(() => 0.5 - Math.random())
      .slice(0, emojiCount);

    selectedNodes.forEach((node) => {
      const words = node.textContent.split(/\s+/);
      const randomIndex = Math.floor(Math.random() * words.length);
      const selectedWord = words[randomIndex];
      const emoji = getRandomEmoji(selectedWord);

      const span = document.createElement('span');
      span.textContent = emoji;
      span.title = selectedWord;
      span.style.cursor = 'pointer';
      span.dataset.originalText = selectedWord;

      span.addEventListener('click', function() {
        this.textContent = this.dataset.originalText;
        this.removeEventListener('click', arguments.callee);
      });

      const newTextNode = document.createTextNode(
        words.slice(0, randomIndex).join(' ') + 
        ' ' + 
        words.slice(randomIndex + 1).join(' ')
      );

      node.parentNode.insertBefore(newTextNode, node);
      node.parentNode.insertBefore(span, newTextNode.nextSibling);
      node.parentNode.removeChild(node);
    });
  });
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "replaceWithEmojis") {
    replaceWithEmojis();
  }
});
