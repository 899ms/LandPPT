"""Response helpers for streams with task-local resources."""

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Send


class ClosingStreamingResponse(StreamingResponse):
    """Close the body in its consuming task, including on disconnect/cancellation."""

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                # Starlette's disconnect listener cancels the streaming task.
                # Allow awaited cleanup to finish, but keep it in this task:
                # moving aclose() to another task breaks ContextVar token reset.
                with anyio.CancelScope(shield=True):
                    await close()
