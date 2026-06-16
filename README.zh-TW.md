<p align="center">
  <img src="static/video-use-banner.png" alt="video-use-cht" width="100%">
</p>

# video-use-cht

[English](./README.md) · **繁體中文**

> ### 🍴 fork 自 [browser-use/video-use](https://github.com/browser-use/video-use)
> 一個針對**繁體中文 / 台灣華語 / 台語**特化的衍生版本,把雲端的
> **ElevenLabs Scribe** 轉錄後端換成**本地 [Breeze-ASR](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)(透過 WhisperX)**。
> 原始剪輯器的設計全數歸功於 **Browser Use**;這個 fork 只更動「轉錄」這一層。
> 為獨立專案,**未**與 Browser Use 或聯發科研究院(MediaTek Research)有任何隸屬或背書關係。
> 詳見 [歸屬與授權](#歸屬與授權)。

辨識**中文 / 台語 / 中英夾雜**更準、**在本機跑、不需轉錄 API key**。轉錄之後的所有流程(打包、EDL、算圖、自我檢查)都與上游相同。

把素材丟進一個資料夾,用對話跟 Claude Code 講話,拿回 `final.mp4`。適用任何題材 —— 口播、混剪、教學、旅遊、訪談 —— 不需預設範本、不需選單。

## 它能做什麼

- **剪掉贅詞**(`嗯`、`呃`、口頭禪)與鏡次之間的空白
- **自動調色**每一段(電影暖調、中性、或任何自訂 ffmpeg 鏈)
- 每個剪接點加 **30ms 音訊淡入淡出**,聽不到爆音
- **燒錄字幕**,樣式可完全自訂
- 透過 [HyperFrames](https://github.com/heygen-com/hyperframes)、[Remotion](https://www.remotion.dev/)、[Manim](https://www.manim.community/)、PIL **產生動畫疊層**,由平行子代理各做一個
- 在給你看之前**自我檢查每個剪接點**的算圖結果
- 把工作階段記憶存進 `project.md`

## 跟上游 video-use 差在哪

| | 上游 video-use | **video-use-cht** |
|---|---|---|
| 轉錄 | ElevenLabs Scribe(雲端、付費) | **Breeze-ASR-25/26 透過 WhisperX(本地、免費)** |
| 逐字時間戳 | Scribe 原生 | **wav2vec2 強制對齊** |
| 講者分離 | Scribe 內建 | **pyannote**(選用,需 HF token) |
| 最擅長 | 英文 | **台灣華語 / 台語 / 中英夾雜** |
| API key | `ELEVENLABS_API_KEY`(必填) | `HF_TOKEN`(選填,僅供分人) |

### 該用哪個模型?(中 / 英 / 台)

用 `--model` 依素材選擇:

- **`breeze25`**(預設)— 台灣華語 + 英文夾雜(**中 + 英**)。使用預先建好的 CTranslate2 權重 [`SoybeanMilk/faster-whisper-Breeze-ASR-25`](https://huggingface.co/SoybeanMilk/faster-whisper-Breeze-ASR-25)。
- **`breeze26`** / **`taigi`** — [BreezeASR-Taigi](https://huggingface.co/MediaTek-Research/Breeze-ASR-26):台語 + 華語(**台 + 中**),輸出成漢字。目前無公開 CT2 版,需自行轉檔,見[英文版的 Breeze-ASR-26 setup](./README.md#breeze-asr-26-taigi-setup)。

沒有單一模型同時最擅長中+英+台,而且不能在同一段音訊上同時跑兩個。預設用 `breeze25`,台語很重的片段才切到 `breeze26`。

## 安裝

```bash
# 1. clone 並 symlink 到 agent 的 skills 目錄
git clone https://github.com/JasChiang/video-use-cht ~/Developer/video-use-cht
ln -sfn ~/Developer/video-use-cht ~/.claude/skills/video-use-cht     # Claude Code

# 2. 安裝相依(Python 3.10–3.12;torch/whisperx 還沒有 3.13 的 wheel)
cd ~/Developer/video-use-cht
uv venv --python 3.12
uv pip install -e .             # whisperx + 對齊 + 分人 + helpers
brew install ffmpeg             # 必須

# 3.(選用)多人素材的講者分離
cp .env.example .env
$EDITOR .env                    # HF_TOKEN=...(token 步驟見 .env.example)
```

分人用的是 gated 的 pyannote 模型。要啟用:到 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 建一個 token,並到 [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) 按同意(pyannote.audio 4.x 旗艦 pipeline)。**沒有 token 也能轉錄**,只是全部標成 `speaker_0`。

## 使用

```bash
cd /path/to/your/videos
claude    # 或 codex、hermes 等
```

> 把這些剪成一支發表影片

它會盤點素材、提出策略、等你確認,然後產出 `edit/final.mp4`。

手動 / 批次轉錄:

```bash
python helpers/transcribe.py clip.mp4 --num-speakers 2          # 單檔,含分人
python helpers/transcribe.py clip.mp4 --model taigi --no-diarize
python helpers/transcribe_batch.py ./takes --num-speakers 2     # 整個資料夾
```

## 設計決策(與 Claude Code 協作)

這個 fork 是與 [Claude Code](https://www.anthropic.com/claude-code) 互動式設計、實作出來的。關鍵決策與理由:

- **為什麼要換 ASR —— Breeze-ASR。** 上游的 ElevenLabs Scribe 是雲端、付費、針對英文。Breeze-ASR-25 是聯發科研究院把 Whisper-large-v2 針對**台灣華語 + 中英夾雜**微調的版本(CER 比原生 Whisper 好約 10%、code-switch +56%),Breeze-ASR-26 再補上台語。在本機跑也讓轉錄免費又私密。

- **為什麼用 WhisperX,而不是直接用 Breeze。** video-use 有條硬規則:**只用逐字時間戳** —— 每個剪接點都 snap 到字邊界。但 Whisper 家族(含 Breeze)對中文的**原生逐字時間戳會抖**,因為中文沒有詞邊界、時間是用 cross-attention 推出來的。WhisperX 把 Breeze 的轉錄再經過 **wav2vec2 強制對齊**,得到穩定得多的逐字時間;而且它把 **pyannote 分人**包在同一道流程裡,連講者標籤一起搞定。所以 WhisperX = Breeze 的準度 + 可靠時間戳 + 分人,一條龍。

- **為什麼用「與 Scribe 相容的 `words[]`」接縫。** 這個 fork **只**改轉錄後端。下游全部(`pack_transcripts.py`、EDL、算圖、自我檢查)都只讀一個 JSON `words[]` 陣列。所以 adapter([`helpers/transcribe_breeze.py`](./helpers/transcribe_breeze.py))把 WhisperX 的輸出映射成 ElevenLabs Scribe 完全一樣的 schema(`type`/`start`/`end`/`text`/`speaker_id`,合成 `spacing` 條目、把 `SPEAKER_00 → speaker_0`)。這讓 diff 極小,完全沒動到上游的剪輯邏輯。

- **為什麼用 pyannote `speaker-diarization-community-1`。** 一開始鎖定 `3.1`,但 pyannote.audio 4.x 即使你要 3.1 也會去抓 `community-1` 的檔案,所以直接 pin `community-1`(你要按同意的就是它)。

- **為什麼在 Apple Silicon 跑 CPU。** CTranslate2(faster-whisper 的引擎)沒有 Metal/MPS,所以 Breeze 跑 CPU。對**離線**剪輯來說,這是換取「本地 + 免費」的合理代價;對齊與分人仍可用 MPS。

- **CJK 打包修正。** WhisperX 對中文逐字對齊(連英文也拆成單字母),導致打包後的逐字稿讀起來像 `大 家 好` / `v i d e o`。`pack_transcripts.py` 現在改成 CJK 感知的接合:任何 CJK 邊界或兩個單一 ASCII 字之間都不空格,只有真正的多字母英文詞之間才空格。

這是個反覆迭代的過程 —— 包含只有實際端到端跑才會抓到的真 bug(被改名的 `token` 參數、`community-1` 門禁、無害的 `torchcodec` 警告)。[疑難排解](#疑難排解)表就是這些經驗的精華。

## 不用素材也能驗證

可以用 macOS `say` 合成一段雙人音檔來測整條管線:

```bash
cd /tmp && mkdir -p vuc-test && cd vuc-test
say -v Meijia   -o s1.aiff "大家好，歡迎收看今天的節目，今天要來聊聊 AI 影片剪輯。"
say -v Tingting -o s2.aiff "對啊，我覺得這個 video 工具真的很方便。"
for f in s1 s2; do ffmpeg -y -i $f.aiff -ac 1 -ar 16000 $f.wav; done
ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 0.7 sil.wav
printf "file 's1.wav'\nfile 'sil.wav'\nfile 's2.wav'\n" > l.txt
ffmpeg -y -f concat -safe 0 -i l.txt -c copy conversation.wav

python ~/Developer/video-use-cht/helpers/transcribe.py conversation.wav --num-speakers 2
python ~/Developer/video-use-cht/helpers/pack_transcripts.py --edit-dir edit
cat edit/takes_packed.md
```

你會看到繁體中文逐字稿、`S0`/`S1` 講者標籤,`video` 內嵌保留。(兩個合成 TTS 聲線對分人是地獄級難度,真人聲音有差異會分得好很多。)

## 疑難排解

| 症狀 | 原因 / 解法 |
|---|---|
| 安裝時 `No matching distribution` / 編譯錯誤 | Python 3.13+ 還沒有 torch/whisperx 的 wheel。用 `uv venv --python 3.12`。 |
| `GatedRepoError: ... speaker-diarization-community-1 ... not in the authorized list` | 用**跟 `HF_TOKEN` 同一個帳號**到[模型頁](https://huggingface.co/pyannote/speaker-diarization-community-1)按同意,再重跑。pyannote.audio 4.x 即使你指定舊的 `3.1` 也會用 `community-1`。 |
| `torchcodec ... Could not load libtorchcodec` 警告 | **無害**。whisperx 用 in-memory 音訊餵 pyannote,根本不會呼叫 torchcodec(警告是因為 Homebrew 的 ffmpeg 是 8,而 torchcodec 要 4–7)。 |
| 轉錄很慢 | 正常。CTranslate2 在 Apple Silicon 沒有 Metal/MPS,Breeze 跑 CPU(約 1.2× 實時),離線剪輯沒問題。第一次還會下載約 3–4 GB 權重(一次性)。 |
| 贅詞(`嗯`/`呃`)沒被剪掉 | Whisper 家族(含 Breeze)傾向省略贅詞,所以靠逐字稿剪贅詞會比 Scribe 的 verbatim 模式弱。靜音裁切不受影響。 |
| 所有語音都標 `speaker_0` | 沒有 `HF_TOKEN`,或你加了 `--no-diarize`。單人素材本來就會維持 `speaker_0`。 |

## 運作原理

LLM 從不「看」影片,而是「讀」它 —— 透過兩層,給它逐字邊界等級的精度。

**第一層 — 音訊逐字稿(永遠載入)。** 每個來源跑一次 Breeze-ASR(WhisperX:faster-whisper 轉錄 → wav2vec2 對齊 → pyannote 分人),給出逐字時間戳與講者標籤,以與 Scribe 相容的 `words[]` schema 輸出,所有鏡次打包成單一 `takes_packed.md`。

**第二層 — 視覺合成(按需)。** `timeline_view` 對任意時間範圍產出膠片條 + 波形 + 字標 PNG,只在決策點呼叫。

## 歸屬與授權

- fork 自 **[browser-use/video-use](https://github.com/browser-use/video-use)** — MIT 授權,原始 `LICENSE` 完整保留。
- 轉錄使用 **[MediaTek-Research/Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)** 與 **[Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26)** — Apache 2.0。
- ASR-25 的 CT2 權重來自 **[SoybeanMilk/faster-whisper-Breeze-ASR-25](https://huggingface.co/SoybeanMilk/faster-whisper-Breeze-ASR-25)**。
- 管線使用 **[WhisperX](https://github.com/m-bain/whisperX)**(BSD-2)與 **[pyannote.audio](https://github.com/pyannote/pyannote-audio)**(MIT,gated 權重)。

**相對於上游的更動:** 把 ElevenLabs Scribe 後端(`helpers/transcribe.py`)換成本地 Breeze-ASR/WhisperX 後端(`helpers/transcribe_breeze.py`);相應更新 `pyproject.toml`、`.env.example` 與本 README。其餘剪輯邏輯皆為上游所有。

**以 [Claude Code](https://www.anthropic.com/claude-code) 協助打造。** 這個 fork —— 後端替換、CJK 打包修正、文件、端到端測試 —— 都是在 Claude Code 協助下完成。
