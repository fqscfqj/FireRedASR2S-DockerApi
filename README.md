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

## 核心环境变量
- `MODEL_PATH`：模型目录，默认 `/models`
- `VRAM_TTL`：模型空闲显存回收时间（秒），默认 `300`
- `MODEL_DOWNLOAD_MODE`：`lazy` 或 `startup`，默认 `lazy`
- `FIRERED_REPO_DIR`：FireRedASR2S 源码目录，默认 `/opt/FireRedASR2S`
- `ASR_TYPE`：`aed` 或 `llm`，默认 `aed`
- `USE_HALF`：全局半精度开关（FP16），默认 `false`
- `HALF_FALLBACK_FP32`：全局 FP16 异常自动降级到 FP32 的开关，默认 `true`
- `ASR_USE_HALF`：ASR 半精度开关，默认继承 `USE_HALF`
- `ASR_HALF_FALLBACK_FP32`：ASR 在 FP16 推理异常时，自动降级到 FP32 并重试一次，默认继承 `HALF_FALLBACK_FP32`
- `VAD_USE_HALF`：VAD 半精度开关，默认继承 `USE_HALF`
- `LID_USE_HALF`：LID 半精度开关，默认继承 `USE_HALF`
- `LID_HALF_FALLBACK_FP32`：LID 在 FP16 推理触发上游断言时，自动降级到 FP32 并重试一次，默认继承 `HALF_FALLBACK_FP32`
- `PUNC_USE_HALF`：Punc 半精度开关，默认继承 `USE_HALF`
- `PROCESS_ALL_FILTER_SCRIPT_MISMATCH`：`/v1/process_all`按LID过滤“语言脚本不匹配”字符（如英文段误识别汉字），默认 `true`
- `PROCESS_ALL_FILTER_MIN_CONFIDENCE`：启用上述过滤时的最小LID置信度，默认 `0.80`
- `ASR_REPEAT_FILTER_ENABLED`：是否启用重复 token/字符抑制与低信息文本过滤，默认 `true`
- `ASR_MAX_CONSECUTIVE_TOKEN_REPEATS`：同一 token 连续保留上限，默认 `8`
- `ASR_MAX_CONSECUTIVE_CHAR_REPEATS`：同一字符连续保留上限，默认 `6`
- `ASR_LOW_INFO_MIN_CHARS`：触发低信息过滤的最小词元数阈值，默认 `24`
- `ASR_LOW_INFO_UNIQUE_RATIO`：低信息文本最小唯一词元占比阈值（低于该值则丢弃），默认 `0.16`
- `API_KEY`：启用鉴权的密钥，默认空（空=不鉴权）
- `API_KEY_HEADER`：鉴权头名，默认 `X-API-Key`

### 显存优化建议
- 首选 `MODEL_DOWNLOAD_MODE=lazy` + `VRAM_TTL=300`（已默认开启）
- GPU 显存紧张时，设置 `USE_HALF=true`；如需细粒度控制，再单独设置 `*_USE_HALF`
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
