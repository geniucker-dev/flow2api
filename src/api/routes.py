"""API routes - Gemini v1beta endpoints"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional, Tuple, Dict, Any
import base64
import re
import time
from urllib.parse import urlparse

from curl_cffi.requests import AsyncSession

from ..core.auth import verify_api_key_header
from ..core.models import Task
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG
from ..core.logger import debug_logger

router = APIRouter()

# Dependency injection will be set up in main.py
generation_handler: GenerationHandler = None


UNIFIED_VIDEO_MODELS: Dict[str, Dict[str, str]] = {
    # Gemini-style unified Veo model ids -> internal model ids
    "veo-3.1-generate-preview": {
        "t2v_landscape": "veo_3_1_t2v_landscape",
        "t2v_portrait": "veo_3_1_t2v_portrait",
        "i2v_landscape": "veo_3_1_i2v_s_fast_fl",
        "i2v_portrait": "veo_3_1_i2v_s_fast_portrait_fl",
        "r2v_landscape": "veo_3_0_r2v_fast",
        "r2v_portrait": "veo_3_0_r2v_fast_portrait",
    },
    "veo-3.1-fast-generate-preview": {
        "t2v_landscape": "veo_3_1_t2v_fast_landscape",
        "t2v_portrait": "veo_3_1_t2v_fast_portrait",
        "i2v_landscape": "veo_3_1_i2v_s_fast_fl",
        "i2v_portrait": "veo_3_1_i2v_s_fast_portrait_fl",
    },
    "veo-2.0-generate-001": {
        "t2v_landscape": "veo_2_0_t2v_landscape",
        "t2v_portrait": "veo_2_0_t2v_portrait",
        "i2v_landscape": "veo_2_0_i2v_landscape",
        "i2v_portrait": "veo_2_0_i2v_portrait",
    },
}


def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler


async def retrieve_image_data(url: str) -> Optional[bytes]:
    """优先本地缓存读取，失败后回退网络下载"""
    try:
        if "/tmp/" in url and generation_handler and generation_handler.file_cache:
            path = urlparse(url).path
            filename = path.split("/tmp/")[-1]
            local_file_path = generation_handler.file_cache.cache_dir / filename
            if local_file_path.exists() and local_file_path.is_file():
                data = local_file_path.read_bytes()
                if data:
                    return data
    except Exception as e:
        debug_logger.log_warning(f"[CONTEXT] 本地缓存读取失败: {str(e)}")

    try:
        async with AsyncSession() as session:
            response = await session.get(url, timeout=30, impersonate="chrome110", verify=False)
            if response.status_code == 200:
                return response.content
            debug_logger.log_warning(f"[CONTEXT] 图片下载失败，状态码: {response.status_code}")
    except Exception as e:
        debug_logger.log_error(f"[CONTEXT] 图片下载异常: {str(e)}")

    return None


def _extract_prompt_and_images(payload: dict) -> Tuple[str, List[bytes]]:
    """解析 Gemini v1beta instances/parameters 输入"""
    prompt = payload.get("prompt", "")
    images: List[bytes] = []

    instances = payload.get("instances") if isinstance(payload.get("instances"), list) else []
    if instances:
        first = instances[0] if isinstance(instances[0], dict) else {}
        prompt = first.get("prompt") or first.get("text") or prompt

        raw_images = first.get("images") if isinstance(first.get("images"), list) else []
        if not raw_images:
            raw_images = [first]  # 兼容单图字段

        for item in raw_images:
            if not isinstance(item, dict):
                continue
            image_b64 = (
                item.get("imageBytes")
                or item.get("bytesBase64Encoded")
                or (item.get("image", {}).get("imageBytes") if isinstance(item.get("image"), dict) else None)
            )
            if image_b64:
                images.append(base64.b64decode(image_b64))

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    return prompt, images


def _decode_inline_data(item: Any) -> Optional[bytes]:
    """Decode Gemini-style inlineData or legacy base64 fields."""
    if not isinstance(item, dict):
        return None

    image_b64 = (
        item.get("imageBytes")
        or item.get("bytesBase64Encoded")
        or (item.get("inlineData", {}).get("data") if isinstance(item.get("inlineData"), dict) else None)
        or (item.get("image", {}).get("inlineData", {}).get("data") if isinstance(item.get("image"), dict) else None)
        or (item.get("image", {}).get("imageBytes") if isinstance(item.get("image"), dict) else None)
    )
    if not image_b64:
        return None

    return base64.b64decode(image_b64)


def _resolve_video_model_and_images(model: str, payload: dict) -> Tuple[str, str, List[bytes]]:
    """Resolve request into (prompt, resolved_model, images)."""
    if model not in UNIFIED_VIDEO_MODELS:
        prompt, images = _extract_prompt_and_images(payload)
        return prompt, model, images

    instances = payload.get("instances") if isinstance(payload.get("instances"), list) else []
    first = instances[0] if instances and isinstance(instances[0], dict) else {}
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}

    prompt = first.get("prompt") or first.get("text") or payload.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if first.get("video"):
        raise HTTPException(status_code=400, detail="Video extension mode is not supported yet")

    aspect_ratio = parameters.get("aspectRatio", "16:9")
    if aspect_ratio not in {"16:9", "9:16"}:
        raise HTTPException(status_code=400, detail="aspectRatio must be '16:9' or '9:16'")
    mode_key = "portrait" if aspect_ratio == "9:16" else "landscape"

    images: List[bytes] = []
    start_image = _decode_inline_data(first.get("image"))
    last_frame = _decode_inline_data(parameters.get("lastFrame"))
    if not start_image:
        images_field = first.get("images") if isinstance(first.get("images"), list) else []
        decoded_images = [_decode_inline_data(item) for item in images_field]
        decoded_images = [img for img in decoded_images if img]
        if decoded_images:
            start_image = decoded_images[0]
            if len(decoded_images) >= 2 and not last_frame:
                last_frame = decoded_images[1]

    reference_images_raw = parameters.get("referenceImages") if isinstance(parameters.get("referenceImages"), list) else []
    reference_images = [img for img in (_decode_inline_data(ref) for ref in reference_images_raw) if img]

    model_alias = UNIFIED_VIDEO_MODELS[model]
    if reference_images:
        selected_model = model_alias.get(f"r2v_{mode_key}")
        if not selected_model:
            raise HTTPException(status_code=400, detail=f"Model {model} does not support referenceImages")
        if len(reference_images) > 3:
            raise HTTPException(status_code=400, detail="referenceImages supports at most 3 images")
        return prompt, selected_model, reference_images

    if start_image:
        selected_model = model_alias.get(f"i2v_{mode_key}")
        if not selected_model:
            raise HTTPException(status_code=400, detail=f"Model {model} does not support image input")
        images.append(start_image)
        if last_frame:
            images.append(last_frame)
        return prompt, selected_model, images

    selected_model = model_alias.get(f"t2v_{mode_key}")
    if not selected_model:
        raise HTTPException(status_code=400, detail=f"Model {model} does not support text-only generation")
    return prompt, selected_model, images


async def _prepare_generation_context(model: str, generation_type: str):
    """选 token + 刷新 AT + 准备 project_id"""
    if generation_type == "image":
        token = await generation_handler.load_balancer.select_token(for_image_generation=True, model=model)
    else:
        token = await generation_handler.load_balancer.select_token(for_video_generation=True, model=model)

    if not token:
        raise HTTPException(status_code=503, detail="No available token")

    if not await generation_handler.token_manager.is_at_valid(token.id):
        raise HTTPException(status_code=503, detail="Token AT invalid or refresh failed")

    token = await generation_handler.token_manager.get_token(token.id)
    project_id = await generation_handler.token_manager.ensure_project_exists(token.id)
    return token, project_id


def _op_metadata(progress: int = 0) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {"createTime": now, "progressPercent": progress}


def _normalize_operation_id(operation_name: str) -> str:
    """支持传 operation_id 或 operations/{operation_id}"""
    return operation_name.removeprefix("operations/")


def _build_video_operation_response(operation_id: str, video_url: str, progress: int = 100) -> dict:
    """构造 Gemini 风格 operation.response，同时兼容旧字段"""
    return {
        "name": f"operations/{operation_id}",
        "metadata": _op_metadata(progress),
        "done": True,
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [{
                    "video": {
                        "uri": video_url,
                        "mimeType": "video/mp4"
                    }
                }]
            },
            # backward compatibility
            "generatedVideos": [{"video": {"uri": video_url, "mimeType": "video/mp4"}}],
            "raiMediaFilteredCount": 0,
            "raiMediaFilteredReasons": []
        }
    }


@router.get("/v1beta/models")
async def list_models(api_key: str = Depends(verify_api_key_header)):
    """Gemini v1beta list models"""
    models = []
    for model_id, cfg in MODEL_CONFIG.items():
        methods = ["predictLongRunning"] if cfg["type"] == "video" else ["predict"]
        models.append({
            "name": f"models/{model_id}",
            "baseModelId": model_id,
            "displayName": model_id,
            "description": f"Flow2API proxied {cfg['type']} model",
            "supportedGenerationMethods": methods,
        })
    for model_id in UNIFIED_VIDEO_MODELS:
        models.append({
            "name": f"models/{model_id}",
            "baseModelId": model_id,
            "displayName": model_id,
            "description": "Unified Veo model id (mode selected by input payload)",
            "supportedGenerationMethods": ["predictLongRunning"],
        })
    return {"models": models}


@router.get("/v1beta/models/{model}")
async def get_model(model: str, api_key: str = Depends(verify_api_key_header)):
    """Gemini v1beta get model"""
    if model in UNIFIED_VIDEO_MODELS:
        return {
            "name": f"models/{model}",
            "baseModelId": model,
            "displayName": model,
            "description": "Unified Veo model id (mode selected by input payload)",
            "supportedGenerationMethods": ["predictLongRunning"],
        }

    cfg = MODEL_CONFIG.get(model)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")

    methods = ["predictLongRunning"] if cfg["type"] == "video" else ["predict"]
    return {
        "name": f"models/{model}",
        "baseModelId": model,
        "displayName": model,
        "description": f"Flow2API proxied {cfg['type']} model",
        "supportedGenerationMethods": methods,
    }


async def _predict_image_internal(model: str, payload: dict) -> dict:
    """Gemini v1beta: models.predict for image models"""
    model_config = MODEL_CONFIG.get(model)
    if not model_config or model_config.get("type") != "image":
        raise HTTPException(status_code=400, detail=f"Model {model} is not an image model")

    prompt, images = _extract_prompt_and_images(payload)
    token, project_id = await _prepare_generation_context(model, "image")

    result = None
    async for chunk in generation_handler._handle_image_generation(  # pylint: disable=protected-access
        token, project_id, model_config, prompt, images if images else None, False
    ):
        result = chunk

    if not result:
        raise HTTPException(status_code=500, detail="Image generation failed")

    # 现有 handler 非流式返回 Markdown 图片，提取 URL 并转 base64 输出
    match = re.search(r"\((.*?)\)", result)
    if not match:
        raise HTTPException(status_code=500, detail="Cannot parse image URL from response")

    image_url = match.group(1)
    image_data = await retrieve_image_data(image_url)
    if not image_data:
        raise HTTPException(status_code=500, detail="Failed to fetch generated image")

    image_b64 = base64.b64encode(image_data).decode("utf-8")
    return {
        "predictions": [{
            "bytesBase64Encoded": image_b64,
            "mimeType": "image/png"
        }],
        # backward compatibility
        "generatedImages": [{
            "image": {
                "imageBytes": image_b64,
                "mimeType": "image/png"
            }
        }]
    }


@router.post("/v1beta/models/{model}:predict")
async def predict_image(
    model: str,
    payload: dict,
    api_key: str = Depends(verify_api_key_header)
):
    return await _predict_image_internal(model, payload)


# backward compatibility for previous implementation
@router.post("/v1beta/models/{model}:generateImages")
async def generate_images(
    model: str,
    payload: dict,
    api_key: str = Depends(verify_api_key_header)
):
    return await _predict_image_internal(model, payload)


@router.post("/v1beta/models/{model}:predictLongRunning")
async def predict_long_running(
    model: str,
    payload: dict,
    api_key: str = Depends(verify_api_key_header)
):
    """Gemini v1beta: Veo long-running task creation"""
    unified_request = model in UNIFIED_VIDEO_MODELS
    prompt, resolved_model, images = _resolve_video_model_and_images(model, payload)

    model_config = MODEL_CONFIG.get(resolved_model)
    if not model_config or model_config.get("type") != "video":
        raise HTTPException(status_code=400, detail=f"Model {model} is not a video model")

    token, project_id = await _prepare_generation_context(resolved_model, "video")

    video_type = model_config.get("video_type", "t2v")

    if video_type == "i2v":
        if len(images) < 1 or len(images) > 2:
            raise HTTPException(status_code=400, detail="i2v model requires 1-2 images")

        start_media_id = await generation_handler.flow_client.upload_image(
            token.at, images[0], model_config["aspect_ratio"]
        )
        if len(images) == 2:
            end_media_id = await generation_handler.flow_client.upload_image(
                token.at, images[1], model_config["aspect_ratio"]
            )
            result = await generation_handler.flow_client.generate_video_start_end(
                at=token.at,
                project_id=project_id,
                prompt=prompt,
                model_key=model_config["model_key"],
                aspect_ratio=model_config["aspect_ratio"],
                start_media_id=start_media_id,
                end_media_id=end_media_id,
                user_paygate_tier=token.user_paygate_tier or "PAYGATE_TIER_ONE"
            )
        else:
            actual_model_key = model_config["model_key"].replace("_fl_", "_")
            if actual_model_key.endswith("_fl"):
                actual_model_key = actual_model_key[:-3]
            result = await generation_handler.flow_client.generate_video_start_image(
                at=token.at,
                project_id=project_id,
                prompt=prompt,
                model_key=actual_model_key,
                aspect_ratio=model_config["aspect_ratio"],
                start_media_id=start_media_id,
                user_paygate_tier=token.user_paygate_tier or "PAYGATE_TIER_ONE"
            )

    elif video_type == "r2v" and images:
        refs = []
        for img in images:
            media_id = await generation_handler.flow_client.upload_image(token.at, img, model_config["aspect_ratio"])
            refs.append({"imageUsageType": "IMAGE_USAGE_TYPE_ASSET", "mediaId": media_id})
        result = await generation_handler.flow_client.generate_video_reference_images(
            at=token.at,
            project_id=project_id,
            prompt=prompt,
            model_key=model_config["model_key"],
            aspect_ratio=model_config["aspect_ratio"],
            reference_images=refs,
            user_paygate_tier=token.user_paygate_tier or "PAYGATE_TIER_ONE"
        )

    else:
        result = await generation_handler.flow_client.generate_video_text(
            at=token.at,
            project_id=project_id,
            prompt=prompt,
            model_key=model_config["model_key"],
            aspect_ratio=model_config["aspect_ratio"],
            user_paygate_tier=token.user_paygate_tier or "PAYGATE_TIER_ONE"
        )

    operations = result.get("operations", [])
    if not operations:
        raise HTTPException(status_code=500, detail="Failed to create operation")

    op = operations[0]
    task_id = _normalize_operation_id(op["operation"]["name"])
    scene_id = op.get("sceneId")

    await generation_handler.db.create_task(Task(
        task_id=task_id,
        token_id=token.id,
        model=model if unified_request else model_config["model_key"],
        prompt=prompt,
        status="processing",
        scene_id=scene_id
    ))

    return {
        "name": f"operations/{task_id}",
        "metadata": _op_metadata(0),
        "done": False
    }


@router.get("/v1beta/operations/{operation_id:path}")
async def get_operation(
    operation_id: str,
    api_key: str = Depends(verify_api_key_header)
):
    """Gemini v1beta: operation polling"""
    operation_id = _normalize_operation_id(operation_id)

    task = await generation_handler.db.get_task(operation_id)
    if not task:
        raise HTTPException(status_code=404, detail="Operation not found")

    if task.status == "completed":
        url = task.result_urls[0] if task.result_urls else ""
        return _build_video_operation_response(operation_id, url)

    if task.status == "failed":
        return {
            "name": f"operations/{operation_id}",
            "done": True,
            "error": {
                "code": 500,
                "message": task.error_message or "Video generation failed",
                "details": []
            }
        }

    token = await generation_handler.token_manager.get_token(task.token_id)
    if not token or not token.at:
        raise HTTPException(status_code=500, detail="Token unavailable for polling")

    status_result = await generation_handler.flow_client.check_video_status(token.at, [{
        "operation": {"name": operation_id},
        "sceneId": task.scene_id,
        "status": "RUNNING"
    }])

    checked = (status_result.get("operations") or [{}])[0]
    status = checked.get("status", "PENDING")

    if status == "SUCCEEDED":
        samples = checked.get("generatedSamples") or []
        video_obj = samples[0].get("video", {}) if samples else {}
        video_url = video_obj.get("uri") or video_obj.get("fifeUrl") or ""
        await generation_handler.db.update_task(operation_id, status="completed", progress=100, result_urls=[video_url])
        return _build_video_operation_response(operation_id, video_url)

    if status == "FAILED":
        err = checked.get("error", {}).get("message", "Video generation failed")
        await generation_handler.db.update_task(operation_id, status="failed", error_message=err)
        return {
            "name": f"operations/{operation_id}",
            "done": True,
            "error": {
                "code": 500,
                "message": err,
                "details": []
            }
        }

    return {
        "name": f"operations/{operation_id}",
        "metadata": _op_metadata(10),
        "done": False
    }
