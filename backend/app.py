import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

MODEL_PATH = "./akkadian-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            text = payload.get("text", "")

            if not text or not text.strip():
                await websocket.send_text("")
                await websocket.close()
                break

            # inputs = tokenizer(text, return_tensors="pt")
            # outputs = model.generate(**inputs)
            # translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            await websocket.send_text(str(len(text)))
            await websocket.close()
            break

    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)