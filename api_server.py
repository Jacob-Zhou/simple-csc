"""
This script implements an API for the LM Corrector Service using FastAPI and uvicorn.
It is adapted from the api_server.py file of the chatglm3-6b model (https://github.com/THUDM/ChatGLM3/blob/main/openai_api_demo/api_server.py).

Key Components:
1. Model Setup: Initializes the LMCorrector model.
2. FastAPI Configuration: Sets up the app with CORS middleware.
3. API Endpoints:
   - "/health": Simple health check.
   - "/correction": Handles text correction requests, supporting both streaming and non-streaming responses.
4. Pydantic Models: Defines request and response structures for type safety and API documentation.
5. Utility Functions: Includes predict_stream for handling streaming responses.
6. Main Execution: Parses command-line arguments, initializes the model, and starts the FastAPI server.

Features:
- Supports both streaming and non-streaming text correction.
- Configurable via command-line arguments (model, host, port, etc.).
- Implements proper error handling and request validation.

Note:
This script is designed for single-GPU usage. Multi-GPU support and special token setup
are not included by default and would require additional configuration.
"""

import os
import time
from lmcsc import LMCorrector
import argparse
import torch
import uvicorn

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from loguru import logger
from pydantic import BaseModel, Field

from sse_starlette.sse import EventSourceResponse

# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CorrectionMessage(BaseModel):
    content: str = None


class DeltaMessage(BaseModel):
    content: Optional[str] = None

# for CorrectionRequest

class CorrectionRequest(BaseModel):
    input: str
    contexts: Optional[str] = None
    prompt_split: Optional[str] = "\n"
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    n_beam: Optional[int] = None
    n_beam_hyps_to_keep: Optional[int] = None


class CorrectionResponseChoice(BaseModel):
    index: int
    message: CorrectionMessage


class CorrectionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    index: int


class CorrectionResponse(BaseModel):
    id: str
    model: str
    object: Literal["correction", "correction.chunk"]
    choices: List[Union[CorrectionResponseChoice, CorrectionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/correction", response_model=CorrectionResponse)
async def create_chat_completion(request: CorrectionRequest):
    global corrector
    global args

    if len(request.input) < 1:
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        src=request.input,
        contexts=request.contexts,
        prompt_split=request.prompt_split,
        n_beam=request.n_beam,
        n_beam_hyps_to_keep=request.n_beam_hyps_to_keep,
        stream=request.stream,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:

        # Use the stream mode to read the first few characters, if it is not a function call, direct stram output
        predict_stream_generator = predict_stream(gen_params)
        output = next(predict_stream_generator)
        return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
    else:
        # Here is the handling of stream = False
        response = corrector(**gen_params)[0][0]

        message = CorrectionMessage(
            content=response,
        )

        logger.debug(f"==== message ====\n{message}")

        choice_data = CorrectionResponseChoice(
            index=0,
            message=message,
        )

        return CorrectionResponse(
            model=args.model,
            id="",  # for open_source model, id is empty
            choices=[choice_data],
            object="correction",
        )

def predict_stream(gen_params):
    """
    The function call is compatible with stream mode output.

    The first seven characters are determined.

    :param model_id:
    :param gen_params:
    :return:
    """
    output = ""
    has_send_first_chunk = False
    for new_response in corrector(**gen_params):
        output = new_response[0][0]

        # Send an empty string first to avoid truncation by subsequent next() operations.
        if not has_send_first_chunk:
            message = DeltaMessage(
                content="",
            )
            choice_data = CorrectionResponseStreamChoice(
                index=0,
                delta=message,
            )
            chunk = CorrectionResponse(
                model=args.model,
                id="",
                choices=[choice_data],
                created=int(time.time()),
                object="correction.chunk"
            )
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

        has_send_first_chunk = True
        message = DeltaMessage(
            content=output,
        )
        choice_data = CorrectionResponseStreamChoice(
            index=0,
            delta=message,
        )
        chunk = CorrectionResponse(
            model=args.model,
            id="",
            choices=[choice_data],
            created=int(time.time()),
            object="correction.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    yield '[DONE]'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompted_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--config_path", type=str, default="configs/default_config.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    # Load LLM
    logger.info(f"Loading model {args.model} from {args.config_path}")
    corrector = LMCorrector(
        model=args.model,
        prompted_model=args.prompted_model,
        config_path=args.config_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    logger.info(f"Model {args.model} loaded successfully")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, reload=args.debug)
