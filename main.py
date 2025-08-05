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
    # قراءة التوكن من المتغير البيئي
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        return JSONResponse(content={"error": "Missing REPLICATE_API_TOKEN"}, status_code=500)

    replicate_client = replicate.Client(api_token=replicate_token)

    # قراءة الصورة
    image_bytes = await image.read()

    # رفع الصورة إلى replicate
    output = replicate_client.run(
        "stability-ai/sdxl:15d2403e41fe3652880582b93e35c8c42ea67802ff8f68e63a6837c8c80c6531",
        input={
            "image": image_bytes,
            "prompt": prompt,
        }
    )

    return {"output_url": output}
