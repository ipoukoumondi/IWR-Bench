import os

API_CONFIG = {
    "claude":{
        "max_tokens": 16384,
        "max_images": 70,
        "video_fps": 1,
        "max_video_frames": 64,
        "timeout": 300,
        "max_single_image_size": 1000,
    },
    "glm": {
        "max_tokens": 16384,
        "max_images": 20,
        "video_fps": 1,
        "max_video_frames": 20,
        "timeout": 600
    },
    "qwen": {
        "max_tokens": 4096,
        "max_images": 20,
        "video_fps": 1,
        "max_video_frames": 20,
        "timeout": 300,
    },
    "gpt-4o-2024-05-13": {
        "max_tokens": 4096,
        "max_images": 70,
        "video_fps": 1,
        "max_video_frames": 64,
        "timeout": 600
    },
    "gpt-5": {
        "max_tokens": 16384,
        "max_images": 70,
        "video_fps": 1,
        "max_video_frames": 64,
        "timeout": 600
    },
    "doubao-seed-1-6-thinking-250615": {
        "max_tokens": 16384,
        "max_images": 70,
        "video_fps": 1,
        "max_video_frames": 64,
        "timeout": 600
    },
    "gpt-4.1-2025-04-14":{
        "max_tokens": 16384,
        "max_images": 70,
        "video_fps": 1,
        "max_video_frames": 64,
        "timeout": 600
    },
    "default": {
        "max_tokens": 16384,
        "max_images": 70,
        "video_fps": 1,
        "max_video_frames": 64,
        "timeout": 300
    }
}


PROMPT_TEMPLATE = """
You are an expert front-end developer. Your task is to create a pixel-perfect replica of a website from a video.
Generate a single `index.html` file that contains all HTML, CSS, and JavaScript necessary to replicate the UI, content, and interaction features shown. The webpage resolution in the video is {resolution}.

Instructions:
1. Single File Output: All HTML, CSS, and JS must be in one `index.html` file.
2. If backend logic is implied, mock it in JS with static data (e.g., a JS array for a fake API call).
3. For all clickable elements, please add the class name "btn" in the HTML source code so that the evaluation agent can perform click evaluation. 
4. Assets(Images and Videos in the webpage):
   - All images must use the provided stitched image assets.
   - The `src` attribute must start with the literal, unchanging string `__PLACEHOLDER_ASSETS_BASE_DIR__/`, followed by the actual filename identified from the stitched image.
   - For example: `src="__PLACEHOLDER_ASSETS_BASE_DIR__/logo.svg"`.
   - `<img>` tags must include `width` and `height` attributes.
   - The provided stitched image assets are before the video.
5. No External Dependencies: The generated code must be entirely self-contained. No External Libraries and no External Fonts.
6. Final Response: Return **only the complete HTML code** in a single ```html code block, with no additional text or explanations.
"""
