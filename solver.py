"""解题模块：普通模式走本地 LM Studio，困难模式固定走云端。

后端规则（由模式决定，不再是全局开关）：
- 普通解题 → solve_backend（本地 solve_model / 可选云端）
- 困难模式 → **固定云端**，且**强制喂图片**（行列式、矩阵等二维结构 OCR 必崩）
- 翻译模式 → 本地 translate_model（由 translator.py 负责）

平台兼容：
- OpenAI 兼容接口（本地 LM Studio / 智谱 / 阿里云百炼 / 火山方舟 / DeepSeek 均可）
- `thinking` 参数：平台不认时自动去掉重试
- 429 / 5xx：指数退避自动重试

输出规则：
- 选择题：只输出「选项字母. 选项内容」
- 简答题：精简答案（题目要求字数/步骤时按要求）
- 详细解释：按需二次请求（图片题会带上原图）
"""
import base64
import io
import json
import os
import re
import subprocess
import time

import requests

import config

_MATH_SYMBOLS = set("∫∑∏√≤≥≠±×÷≈∞πθαβγΔλμσφω∂∇∈∉⊂⊃∪∩⇒⇔→←↑↓"
                    "[]{}=/^_%‰°′″")

_LMS = os.path.expandvars(r"%USERPROFILE%\.lmstudio\bin\lms.exe")

# 可重试的 HTTP 状态码（云端限流/抖动）
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# 中断控制：流式读取时每行检查一次，点「停止」立即断开
_cancel_flag = False


def reset_cancel():
    global _cancel_flag
    _cancel_flag = False


def request_cancel():
    global _cancel_flag
    _cancel_flag = True


_SOLVE_SYSTEM = (
    "你是一位严谨、简洁的解题助手。只输出最终答案，不要输出推理过程。\n"
    "输出规则：\n"
    "1. 若题目是选择题（有 A/B/C/D 等选项）：只输出「选项字母. 选项内容」，"
    "例如「C. 29」。不要解释、不要过程。\n"
    "2. 若题目是简答题/填空题/计算题：直接给精简答案（1-3 句、尽量不超过 120 字）；"
    "若题目明确要求字数、步骤或格式，则严格按要求作答。\n"
    "3. 若题目信息不足或看不清：用一句话说明缺少什么。"
)

_HARD_SYSTEM = (
    "你是一位严谨的解题专家。下面是一张题目截图，请仔细阅读图中的全部内容，"
    "特别注意公式、矩阵、行列式、分数、上下标、根式、表格等二维结构，"
    "不要把它们读错或漏掉。\n"
    "输出规则：\n"
    "1. 若题目是选择题（有 A/B/C/D 等选项）：只输出「选项字母. 选项内容」。\n"
    "2. 其他题型：直接给精简答案（1-3 句、尽量不超过 120 字）；"
    "若题目明确要求字数、步骤或格式，则严格按要求作答。\n"
    "3. 若图片模糊无法辨认：用一句话说明哪里看不清。\n"
    "只输出最终答案，不要输出推理过程。"
)

_EXPLAIN_SYSTEM = (
    "你是一位耐心的老师。针对给出的题目和答案，给出详细解释，包含：\n"
    "1. 考点/关键原理（1-2 句）\n"
    "2. 解题步骤（分点，每点一句话）\n"
    "3. 易错点（1 句）\n"
    "总长控制在 300 字以内，不要重复题目原文。"
)


# ---------------- 本地模型管理 ----------------
def _lms_path():
    return config.get("lms_exe") or _LMS


def _run_lms(args, timeout=420):
    subprocess.run([_lms_path()] + args, capture_output=True, timeout=timeout)


def unload_all():
    try:
        _run_lms(["unload", "--all"], timeout=120)
    except Exception:
        pass


def load_model(model_name, context=None):
    """卸载全部后加载指定模型。

    - context：上下文长度（-c），默认取 config.local_context（16000）
    - parallel 1：单用户场景，把整段上下文都给一个请求
      （LM Studio 默认 parallel=4，会把上下文切成 4 份，每份只有 1/4）
    """
    unload_all()
    ctx = int(context or config.get("local_context", 16000) or 16000)
    _run_lms(["load", model_name, "--gpu", "max",
              "-c", str(ctx), "--parallel", "1", "-y"], timeout=600)


def is_hard_mode():
    """困难模式开关（solve() 只在解题模式被调用，无需再校验 mode）。"""
    return bool(config.get("hard_mode", False))


def solve_backend():
    """当前解题该用哪个后端：困难模式固定云端，普通模式看设置。"""
    if is_hard_mode():
        return "cloud"
    return config.get("solve_backend", "local")


# ---------------- HTTP ----------------
def _endpoint(backend):
    """返回 (base_url, api_key, backend)。"""
    if backend == "cloud":
        url = (config.get("solve_cloud_url") or "").strip().rstrip("/")
        key = (config.get("solve_cloud_key") or "").strip()
        if not url:
            raise RuntimeError("未配置云端解题 API 地址（设置 → 解题 → 云端）")
        if not key:
            raise RuntimeError("未配置云端解题 API Key（设置 → 解题 → 云端）")
        return url, key, "cloud"
    url = (config.get("lmstudio_url") or "http://127.0.0.1:1234/v1").rstrip("/")
    return url, (config.get("api_token") or ""), "local"


def _session(backend):
    s = requests.Session()
    if backend == "local":
        # 本地 127.0.0.1 绝不过系统代理（用户开 VPN 时会被劫持成 401）
        s.trust_env = False
    return s


class _HTTPError(Exception):
    def __init__(self, status, body):
        super().__init__("HTTP %s：%s" % (status, body))
        self.status = status
        self.body = body


def _stream_once(payload, base, key, backend):
    """发一次流式请求并拼出 content（忽略 reasoning_content）。"""
    global _cancel_flag
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    sess = _session(backend)
    r = sess.post(base + "/chat/completions", headers=headers, json=payload,
                  timeout=(15, 600), stream=True)
    if r.status_code != 200:
        body = r.text[:200] if hasattr(r, "text") else ""
        r.close()
        raise _HTTPError(r.status_code, body)

    parts = []
    reasoning = []
    try:
        for raw in r.iter_lines(decode_unicode=False):
            if _cancel_flag:
                raise RuntimeError("已停止")
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", "ignore").strip()
            except ValueError:
                continue
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except ValueError:
                # 只忽略非 JSON 行；其它异常（如 NameError）必须暴露出来
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                parts.append(piece)
            rc = delta.get("reasoning_content") or ""
            if rc:
                reasoning.append(rc)
    finally:
        try:
            r.close()
        except Exception:
            pass

    if _cancel_flag:
        raise RuntimeError("已停止")
    text = "".join(parts).strip()
    if not text:
        if reasoning:
            raise RuntimeError(
                "模型只输出了思考内容、没给最终答案（共 %d 字思考）。"
                "请调大输出上限或改用困难模式。" % len("".join(reasoning)))
        raise RuntimeError("模型没有返回内容（可能被上游截断）")
    return text


def _post_chat(messages, model, max_tokens=3000, temperature=0.2,
               thinking=None, backend="local", retries=2):
    """流式调用，支持中途取消、thinking 开关、429/5xx 自动重试。"""
    global _cancel_flag
    base, key, be = _endpoint(backend)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if thinking is not None:
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}

    attempt = 0
    while True:
        try:
            return _stream_once(payload, base, key, be)
        except _HTTPError as e:
            # 平台不认 thinking 参数 → 去掉重试一次
            if thinking is not None and e.status in (400, 422) \
                    and "thinking" in (e.body or "").lower():
                payload.pop("thinking", None)
                thinking = None
                continue
            # 限流/服务端抖动 → 退避重试
            if e.status in _RETRY_STATUS and attempt < retries and not _cancel_flag:
                attempt += 1
                time.sleep(1.5 * attempt)
                continue
            raise RuntimeError(str(e))


def list_cloud_models(url=None, key=None):
    """拉取云端可用模型 ID 列表（OpenAI 兼容 GET /models）。"""
    base = (url if url is not None else config.get("solve_cloud_url") or "").strip().rstrip("/")
    k = (key if key is not None else config.get("solve_cloud_key") or "").strip()
    if not base:
        raise RuntimeError("请先填「云端 · 地址」")
    headers = {}
    if k:
        headers["Authorization"] = "Bearer " + k
    s = requests.Session()   # 云端允许走系统代理
    r = s.get(base + "/models", headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError("HTTP %s：%s" % (r.status_code, r.text[:150]))
    data = r.json()
    items = data.get("data") or data.get("models") or []
    ids = []
    for it in items:
        if isinstance(it, dict):
            mid = it.get("id") or it.get("name")
        else:
            mid = str(it)
        if mid:
            ids.append(mid)
    return sorted(set(ids))


def pil_to_data_url(pil, max_side=1600, quality=90):
    """PIL → data URL（JPEG，限制最大边长以控制 token）。"""
    img = pil.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------- 判断逻辑 ----------------
def should_use_image(text, pil):
    """自动判断：文字太少或数学符号密集 → 直接喂图片。"""
    cfg_input = config.get("solve_input", "auto")
    if cfg_input == "image":
        return True
    if cfg_input == "text":
        return False
    text = (text or "").strip()
    if len(text) < 15:
        return True
    sym = sum(1 for ch in text if ch in _MATH_SYMBOLS)
    if sym and sym / float(len(text)) > 0.12:
        return True
    return False


def looks_like_multiple_choice(text):
    """粗判是否选择题：出现 3 个以上「A. / B、 / C) / D：」式选项标号。"""
    t = text or ""
    hits = 0
    for letter in "ABCD":
        if re.search(letter + r"\s*[.．、:：)）]", t):
            hits += 1
    return hits >= 3


def _pick_model(use_image, hard, backend):
    """决定用哪个模型名；云端没配置对应模型时给出明确提示。"""
    if backend == "cloud":
        if use_image:
            name = (config.get("solve_cloud_vision_model") or "").strip()
            if not name:
                raise RuntimeError(
                    "困难模式走云端需要「云端 · 图片模型」（设置 → 解题，"
                    "要能看图的模型，如 glm-4.6v）")
            return name
        name = (config.get("solve_cloud_model") or "").strip()
        if not name:
            raise RuntimeError("云端解题未配置「云端 · 文字模型」（设置 → 解题）")
        return name
    return config.get("solve_model", "qwen3-vl-8b-instruct")


def _thinking_for(hard, backend):
    """云端推理模型：困难模式开思考保准确；普通模式默认用平台默认值。

    实测 glm-4.6v 同一道矩阵题：关思考 1.4s 答错 A，开思考 25.6s 答对 D。
    """
    if backend != "cloud":
        return None
    if hard:
        return True
    if config.get("cloud_fast_no_thinking", False):
        return False
    return None


# ---------------- 解题 ----------------
def solve(question_text, pil=None):
    """解题。返回 dict(answer, is_mc, used_image, hard, backend, model)。"""
    reset_cancel()
    hard = is_hard_mode()
    backend = solve_backend()
    use_image = pil is not None and (hard or should_use_image(question_text, pil))
    model = _pick_model(use_image, hard, backend)

    system = _HARD_SYSTEM if (hard or use_image) else _SOLVE_SYSTEM
    question_text = (question_text or "").strip()

    if use_image:
        user_parts = [
            {"type": "text", "text":
                "下面是题目截图，请据此作答。\n"
                + (("（OCR 文字仅供参考，二维排版可能有误：\n%s）\n" % question_text[:800])
                   if question_text else "")
                + "请直接给出答案。"},
            {"type": "image_url", "image_url": {"url": pil_to_data_url(pil)}},
        ]
    else:
        user_parts = [{"type": "text", "text":
            "题目：\n%s\n\n请直接给出答案。" % question_text}]

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_parts},
    ]
    answer = _post_chat(messages, model, backend=backend,
                        thinking=_thinking_for(hard, backend))
    return {
        "answer": answer,
        "is_mc": looks_like_multiple_choice(question_text),
        "used_image": use_image,
        "hard": hard,
        "backend": backend,
        "model": model,
    }


def explain(question_text, answer, pil=None, hard=None):
    """详细解释（二次请求）。图片题会把原图一起发过去，避免 OCR 乱码误导。"""
    reset_cancel()
    if hard is None:
        hard = is_hard_mode()
    backend = "cloud" if hard else config.get("solve_backend", "local")
    use_image = pil is not None and (hard or should_use_image(question_text, pil))
    model = _pick_model(use_image, hard, backend)

    head = ("【题目】\n%s\n\n【已给出的答案】\n%s\n\n请给出详细解释。"
            % ((question_text or "").strip() or "（见截图）", (answer or "").strip()))
    if use_image:
        # 图片题：文字可能是 OCR 乱码，让模型以图为准
        user_parts = [
            {"type": "text", "text": head + "\n（题目以截图为准，上面的文字仅供参考）"},
            {"type": "image_url", "image_url": {"url": pil_to_data_url(pil)}},
        ]
    else:
        user_parts = [{"type": "text", "text": head}]

    messages = [
        {"role": "system", "content": _EXPLAIN_SYSTEM},
        {"role": "user", "content": user_parts},
    ]
    return _post_chat(messages, model, max_tokens=1200, temperature=0.3,
                      backend=backend, thinking=_thinking_for(False, backend))
