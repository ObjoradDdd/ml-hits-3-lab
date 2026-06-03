const source = document.querySelector("#source");
const target = document.querySelector("#target");
const port = 8080;

let debounceTimer = null;
let activeSocket = null;
let hasStreamingStarted = false;

const resetOutput = () => {
  target.textContent = "";
  hasStreamingStarted = false;
};

const abortActiveRequest = () => {
  if (activeSocket) {
    activeSocket.close(1000, "Client canceled");
    activeSocket = null;
  }
};

const appendToken = (text) => {
  if (!text) {
    return;
  }
  if (!hasStreamingStarted) {
    hasStreamingStarted = true;
  }
  target.textContent += text;
};

const buildSocketUrl = () => {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://localhost:${port}/translate`;
};

const sendTranslation = (text) => {
  abortActiveRequest();
  resetOutput();

  if (!text.trim()) {
    return;
  }

  const socket = new WebSocket(buildSocketUrl());
  activeSocket = socket;

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ text }));
  });

  socket.addEventListener("message", (event) => {
    appendToken(String(event.data));
  });

  socket.addEventListener("close", () => {
    if (activeSocket === socket) {
      activeSocket = null;
    }
  });

  socket.addEventListener("error", () => {
    if (activeSocket === socket) {
      activeSocket = null;
    }
    appendToken("Ошибка соединения.");
  });
};

source.addEventListener("input", (event) => {
  const text = event.target.value;

  if (!text.trim()) {
    clearTimeout(debounceTimer);
    abortActiveRequest();
    resetOutput();
    return;
  }

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    sendTranslation(text);
  }, 650);
});
