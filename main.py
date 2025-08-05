from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import replicate
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from Fory API!"}


@app.post("/generate-image")
async def generate_image(prompt: str = Form(...)):
    # قراءة التوكن من البيئة
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        return JSONResponse(content={"error": "Missing REPLICATE_API_TOKEN"}, status_code=500)

    # إعداد العميل
    replicate_client = replicate.Client(api_token=replicate_token)

    try:
        # تشغيل الموديل المجاني
        output = replicate_client.run(
            "stability-ai/stable-diffusion:db21e45c5e2618896c0c47392f7af0d39c4f60f36a0136b45fdd0a9a1f40a4e4",
            input={
                "prompt": prompt
            }
        )

        return {"image_url": output}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
