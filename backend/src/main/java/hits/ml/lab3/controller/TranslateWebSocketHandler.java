package hits.ml.lab3.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.Map;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

@Component
public class TranslateWebSocketHandler extends TextWebSocketHandler {
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws IOException {
        String payload = message.getPayload();
        String text = extractText(payload);
        if (text == null || text.trim().isEmpty()) {
            session.sendMessage(new TextMessage(""));
            session.close(CloseStatus.NORMAL);
            return;
        }

        String count = String.valueOf(text.length());
        for (int i = 0; i < count.length(); i++) {
            session.sendMessage(new TextMessage(String.valueOf(count.charAt(i))));
        }
        session.close(CloseStatus.NORMAL);
    }

    private String extractText(String payload) {
        try {
            Map<String, Object> parsed = objectMapper.readValue(payload, Map.class);
            Object text = parsed.get("text");
            return text == null ? null : String.valueOf(text);
        } catch (IOException ex) {
            return null;
        }
    }
}
