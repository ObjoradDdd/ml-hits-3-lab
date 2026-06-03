package hits.ml.lab3.config;

import hits.ml.lab3.controller.TranslateWebSocketHandler;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {
    private final TranslateWebSocketHandler translateWebSocketHandler;

    public WebSocketConfig(TranslateWebSocketHandler translateWebSocketHandler) {
        this.translateWebSocketHandler = translateWebSocketHandler;
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(translateWebSocketHandler, "/translate")
                .setAllowedOrigins("*");
    }
}
