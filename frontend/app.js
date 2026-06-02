const source = document.querySelector("#source");
const target = document.querySelector("#target");
const loader = document.querySelector("#loader");

let debounceTimer = null;
let activeSocket = null;
let hasStreamingStarted = false;

const setLoading = (isLoading) => {
  loader.hidden = !isLoading;
  target.setAttribute("aria-busy", String(isLoading));
};

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
    setLoading(false);
  }
  target.textContent += text;
};

const buildSocketUrl = () => {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/translate`;
};

const sendTranslation = (text) => {
  abortActiveRequest();
  resetOutput();

  if (!text.trim()) {
    setLoading(false);
    return;
  }

  setLoading(true);

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
    setLoading(false);
  });

  socket.addEventListener("error", () => {
    if (activeSocket === socket) {
      activeSocket = null;
    }
    setLoading(false);
    appendToken("Ошибка соединения.");
  });
};

source.addEventListener("input", (event) => {
  const text = event.target.value;

  if (!text.trim()) {
    clearTimeout(debounceTimer);
    abortActiveRequest();
    resetOutput();
    setLoading(false);
    return;
  }

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    sendTranslation(text);
  }, 650);
});
