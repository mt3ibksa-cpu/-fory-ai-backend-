from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import replicate
import os
import base64

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from Fory API!"}

@app.post("/edit-image")
async def edit_image(prompt: str = Form(...), image: UploadFile = File(...)):
    # جلب التوكن من المتغير البيئي
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        return JSONResponse(content={"error": "Missing REPLICATE_API_TOKEN"}, status_code=500)

    replicate_client = replicate.Client(api_token=replicate_token)

    # قراءة الصورة
    image_bytes = await image.read()

    # تحويل الصورة إلى base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        # استدعاء النموذج
        output = replicate_client.run(
            "fofr/face-to-many:5b58f94235c5fbe65ae9526f6763fd18299834deacbaeead316e121bbee96c18",
            input={
                "image": image_base64,
                "prompt": prompt
            }
        )

        return JSONResponse(content={"output": output})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
