import os
import asyncio
from legacy.avatar_main import create_talking_head

async def synth_and_render(text: str, audio_path: str, video_path: str) -> str:
    # ensure clean paths
    if os.path.exists(video_path): os.remove(video_path)
    if os.path.exists(audio_path): os.remove(audio_path)
    await create_talking_head(text, audio_path, video_path)
    return video_path