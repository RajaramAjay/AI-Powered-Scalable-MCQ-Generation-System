# src/app.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from src.factory_main import MCQPipelineFactory
import json
import asyncio
import os
import uvicorn
import uuid
from logger_setup import setup_logger, start_request_logging, end_request_logging

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

# Logger setup (global)
logger = setup_logger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        os.makedirs("./assets", exist_ok=True)
        file_location = f"./assets/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        return {"filename": file_location}

    except Exception as e:
        # Raise the exception normally
        raise e


@app.get("/upload_sse")
async def upload_sse(filename: str, request: Request = None):
    request_id = str(uuid.uuid4())
    log_file = start_request_logging(request_id)
    file_path = f"./assets/{filename}"

    async def event_generator():
        try:
            logger.info(f"[{request_id}] Starting pipeline for {file_path}")
            pipeline = MCQPipelineFactory(input_path=file_path)

            # Run pipeline
            # questions_json = pipeline.run_pipeline()
            questions_json = pipeline.template_method_example()
            logger.info(f"[{request_id}] Pipeline generated {len(questions_json['questions'])} questions")

            # Adjust format (options -> choices)
            for q in questions_json["questions"]:
                if "options" in q:
                    q["choices"] = q.pop("options")

            # Stream questions
            for idx, q in enumerate(questions_json["questions"], start=1):
                await asyncio.sleep(0.5)  # simulate streaming
                logger.debug(f"[{request_id}] Streaming question {idx}: {q.get('question', 'unknown')}")
                yield f"data: {json.dumps(q)}\n\n"

            logger.info(f"[{request_id}] Finished streaming {len(questions_json['questions'])} questions")

        except Exception as e:
            logger.error(f"[{request_id}] Error in SSE pipeline: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        finally:
            end_request_logging(request_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
