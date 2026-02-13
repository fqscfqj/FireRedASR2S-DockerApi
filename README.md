# FireRedASR2S Docker API

基于 [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) 的工业级语音处理 API，提供：
- ASR（语音识别）
- VAD（端点检测）
- LID（语种识别）
- Punc（标点预测）

## 功能特性
- FastAPI 异步接口：`/v1/asr`、`/v1/vad`、`/v1/lid`、`/v1/punc`、`/v1/process_all`
- ModelScope 自动下载模型（启动时或首次调用）
- 懒加载：服务启动不预加载大模型
- VRAM TTL 回收：模型空闲超过 `VRAM_TTL` 秒自动卸载并 `torch.cuda.empty_cache()`
- API Key 鉴权（默认请求头：`X-API-Key`）
- Docker + Compose 一键部署
- GitHub Actions 手动触发镜像构建并推送到 GHCR

## 目录结构
```text
app/
  main.py            # FastAPI 入口
  model_manager.py   # 下载/懒加载/TTL 回收
  service.py         # ASR 与全流程编排
  audio.py           # FFmpeg 音频规范化
scripts/entrypoint.sh
Dockerfile
docker-compose.yml
.github/workflows/build-image.yml
```

## 环境变量（精简版）
日常部署只建议设置以下 6 个：
- `API_KEY`：鉴权密钥（生产环境建议必填）
- `MODEL_PATH`：模型目录，默认 `/models`
- `MODEL_DOWNLOAD_MODE`：`lazy` 或 `startup`，默认 `lazy`
- `VRAM_TTL`：模型空闲显存回收时间（秒），默认 `300`
- `USE_HALF`：是否启用半精度（显存紧张时设为 `true`）
- `LOG_LEVEL`：日志级别，默认 `info`

其余变量都可以不配，按默认值运行。

### 高级可选变量（需要微调时再用）
- `API_KEY_HEADER`（默认 `X-API-Key`）：
  - 用途：指定 API 鉴权使用的 HTTP 请求头名。常见场景为反向代理或云平台要求自定义 header。
  - 建议：一般保持默认，只有在平台/网关要求时修改。

- `FIRERED_REPO_DIR`（默认 `/opt/FireRedASR2S`）：
  - 用途：本地 FireRedASR2S 源码的挂载/查找目录（`AUTO_CLONE_FIRERED` 为 true 时会在此处 clone）。
  - 建议：开发时可挂载本地源码以便调试；容器部署可使用默认路径。

- `ASR_TYPE`（默认 `aed`，可选 `llm`）：
  - 用途：选择 ASR 解码器类型。`aed` = 本地 AED 解码器（低延迟、离线）；`llm` = 使用 LLM 解码（需要额外后端/模型，延迟/资源消耗更高）。
  - 建议：默认使用 `aed`，仅在已配置 LLM 后端且需要 LLM 能力时切换为 `llm`。

- `HALF_FALLBACK_FP32` / `ASR_HALF_FALLBACK_FP32` / `LID_HALF_FALLBACK_FP32`（默认开启）：
  - 用途：当 FP16 推理出现数值错误（NaN/overflow）时自动回退到 FP32 并重试一次。
  - 建议：保持启用以提高稳定性；若希望强制使用 FP32，可关闭半精度或将这些变量设置为 `false`。

- `ASR_USE_HALF` / `VAD_USE_HALF` / `LID_USE_HALF` / `PUNC_USE_HALF`：
  - 用途：为单个子模块覆盖 `USE_HALF`（启用半精度以节省显存）。
  - 建议：显存充足时保持 `false`；显存紧张时可针对性开启，但可能出现数值不稳定（配合 fallback 使用）。

- `PROCESS_ALL_FILTER_SCRIPT_MISMATCH`（默认 `true`） / `PROCESS_ALL_FILTER_MIN_CONFIDENCE`（默认 `0.80`）：
  - 用途：在 `process_all` 流程中，若 LID 判断为英文且置信度 ≥ 阈值，则从识别结果中过滤中文字符（避免中英混写误识）。
  - 建议：若 LID 不稳定或误杀中文，请降低阈值或关闭该功能。

- `ASR_REPEAT_FILTER_ENABLED`（默认 `true`）：
  - 用途：开启重复 token/字符压缩与“低信息文本”过滤（如大量重复词或无意义低信息输出会被丢弃）。

- `ASR_MAX_CONSECUTIVE_TOKEN_REPEATS`（默认 `8`） / `ASR_MAX_CONSECUTIVE_CHAR_REPEATS`（默认 `6`）：
  - 用途：允许的最大连续重复 token/字符数，超过则压缩丢弃。
  - 建议：对口播/字幕类重复多的场景可适当增大；对噪声或电台直播可减小以去除冗余。

- `ASR_LOW_INFO_MIN_CHARS`（默认 `24`） / `ASR_LOW_INFO_UNIQUE_RATIO`（默认 `0.16`）：
  - 用途：判定“低信息文本”的阈值。只有当字符（或词）数 ≥ `ASR_LOW_INFO_MIN_CHARS` 时才计算 unique_ratio = unique_units / total_units；若 unique_ratio < `ASR_LOW_INFO_UNIQUE_RATIO` 则视为低信息并过滤掉。
  - 调优参考：
    - 保守/避免误杀：`ASR_LOW_INFO_UNIQUE_RATIO = 0.08 - 0.12` 或 增大 `ASR_LOW_INFO_MIN_CHARS`。
    - 更严格过滤噪声/重复：`ASR_LOW_INFO_UNIQUE_RATIO = 0.20 - 0.30`。

- 快速调优建议：
  1. 先修改一个变量并观察日志/输出，再逐步调整。
  2. 若出现“误杀短句/歌词/重复用语”，优先减小 `ASR_LOW_INFO_UNIQUE_RATIO` 或 增大 `ASR_LOW_INFO_MIN_CHARS`。
  3. 若看到 FP16 数值异常（NaN/overflow），开启或保留 `*_HALF_FALLBACK_FP32`。

### 显存优化建议
- 首选 `MODEL_DOWNLOAD_MODE=lazy` + `VRAM_TTL=300`（已默认开启）
- GPU 显存紧张时，设置 `USE_HALF=true`；如需细粒度控制，再单独设置 `*_USE_HALF`
- 服务已固定为单进程 + 单线程运行（Uvicorn worker=1，Torch/AnyIO/BLAS 线程=1），避免并发导致显存额外占用
- 若 FP16 触发数值/断言异常，服务会按模型自动切换到 FP32 并重试一次（ASR/LID），避免接口因半精度不稳定而失败
- 某些上游模型配置类可能不支持 `use_half` 参数，服务会自动跳过该参数并继续加载

## 本地运行（Docker）
```bash
docker compose up --build
```

## API 示例

先设置：
```bash
export API_KEY="change_me_strong_key"
```

`/v1/asr`（返回识别文本+时间戳）
```bash
curl -X POST "http://127.0.0.1:8000/v1/asr?force_refresh=false" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@./demo.wav"
```

`/v1/vad`（仅端点检测）
```bash
curl -X POST "http://127.0.0.1:8000/v1/vad?force_refresh=false" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@./demo.wav"
```

`/v1/lid`（仅语种识别）
```bash
curl -X POST "http://127.0.0.1:8000/v1/lid?force_refresh=false" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@./demo.wav"
```

`/v1/punc`（仅标点预测）
```bash
curl -X POST "http://127.0.0.1:8000/v1/punc?force_refresh=false" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{"text":"hello world this is a test"}'
```

`/v1/process_all`（VAD -> LID -> ASR -> Punc）
```bash
curl -X POST "http://127.0.0.1:8000/v1/process_all?force_refresh=false" \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@./demo.wav"
```

查看模型状态：
```bash
curl -H "X-API-Key: ${API_KEY}" "http://127.0.0.1:8000/v1/models/status"
```

## 手动触发 GitHub Actions 构建镜像
1. 进入仓库 `Actions` 页面。
2. 选择 `build-image` 工作流。
3. 点击 `Run workflow`，填写 `image_tag`（例如 `v1.0.0`）。
4. 构建完成后镜像推送到：
   - `ghcr.io/<your-org>/fireredasr2s-api:<image_tag>`
   - `ghcr.io/<your-org>/fireredasr2s-api:sha-...`

## 注意事项
- FireRedASR2S 期望输入为 `16kHz/mono/PCM`，服务会自动调用 FFmpeg 预处理。
- 首次调用会触发模型下载，耗时取决于网络与模型大小。
- 使用 GPU 时建议挂载持久化 `MODEL_PATH`，避免重复下载。
