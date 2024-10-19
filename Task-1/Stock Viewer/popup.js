document.addEventListener('DOMContentLoaded', () => {
  const fetchButton = document.getElementById('fetchStock');
  const stockSymbolInput = document.getElementById('stockSymbol');
  const resultDiv = document.getElementById('result');

  fetchButton.addEventListener('click', () => {
    const symbol = stockSymbolInput.value.toUpperCase();
    if (symbol) {
      fetchStockData(symbol);
    } else {
      resultDiv.innerHTML = '<p class="error">Please enter a stock symbol.</p>';
    }
  });
});

async function fetchStockData(symbol) {
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '<p>Loading...</p>';

  try {
    const response = await fetch(`http://localhost:5000/stock/${symbol}`);
    const data = await response.json();

    if (data.error) {
      resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
      return;
    }

    let html = `<h2>${symbol} - Last 5 Days</h2><ul>`;

    data.forEach(item => {
      html += `<li>${item.date}: $${item.close.toFixed(2)}</li>`;
    });

    html += '</ul>';
    resultDiv.innerHTML = html;
  } catch (error) {
    resultDiv.innerHTML = `<p class="error">Error fetching stock data: ${error.message}</p>`;
  }
}
