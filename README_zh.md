# nanochat

![nanochat logo](dev/nanochat.png)

> 100 美元能买到的最好的 ChatGPT。

[English](README.md) | **简体中文**

这是一个完整的 LLM（如 ChatGPT）的全栈实现，采用单一、简洁、极简、可定制、依赖少的代码库。nanochat 设计用于在单个 8XH100 节点上运行，通过像 [speedrun.sh](speedrun.sh) 这样的脚本，从开始到结束运行整个流程。这包括分词、预训练、微调、评估、推理，以及通过简单的 UI 进行 Web 服务，这样你就可以像 ChatGPT 一样与自己的 LLM 对话。nanochat 将成为 Eureka Labs 正在开发的 LLM101n 课程的高潮项目。

## 与它对话

要感受这个仓库的终点，你可以找到托管在 [nanochat.karpathy.ai](https://nanochat.karpathy.ai/) 上的 [nanochat d34](https://github.com/karpathy/nanochat/discussions/314)。"d34" 意味着这个模型在 Transformer 神经网络中有 34 层。这个模型有 22 亿个参数，通过简单地运行训练脚本 [run1000.sh](run1000.sh) 并使用 `--target_param_data_ratio=40`（比 Chinchilla-optimal 长 2 倍）在 880 亿个 token 上训练，训练总成本约为 2,500 美元（在 8XH100 GPU 节点上约 100 小时的训练时间）。虽然现在这足以超越 2019 年的 GPT-2，但它远远落后于 GPT-5 等现代大型语言模型。在与这些微模型对话时，你会发现它们犯很多错误，它们有点天真和愚蠢，而且会产生大量幻觉，有点像孩子。这有点有趣。但 nanochat 的独特之处在于它完全是你的——完全可配置、可调整、可定制，并且从头到尾由你训练。要训练并与你自己的模型对话，我们来看...

## 快速开始

感受这种魔力的最快方法是运行 speedrun 脚本 [speedrun.sh](speedrun.sh)，它训练并推理 100 美元级别的 nanochat。在每小时 24 美元的 8XH100 节点上，总运行时间约为 4 小时。从你喜欢的提供商启动一个新的 8XH100 GPU 机器（例如，我使用并喜欢 [Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash speedrun.sh
```

或者，由于脚本运行 4 小时，我喜欢在名为 `speedrun` 的新 screen 会话中启动它（并将输出记录到 `speedrun.log`）：

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

如果你不太熟悉，请参阅 [screen 备忘单](https://gist.github.com/jctosta/af918e1618682638aa82)。你可以在 screen 会话中观看它运行，或者使用 `Ctrl-a d` 分离并使用 `tail speedrun.log` 查看进度。现在等待 4 小时。完成后，你可以通过类似 ChatGPT 的 Web UI 与你的 LLM 对话。再次确保你的本地 uv 虚拟环境处于活动状态（运行 `source .venv/bin/activate`），然后提供服务：

```bash
python -m scripts.chat_web
```

然后访问显示的 URL。确保正确访问它，例如在 Lambda 上使用你所在节点的公共 IP，后跟端口，例如 [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/) 等。然后像通常与 ChatGPT 对话一样与你的 LLM 对话！让它写故事或诗歌。问它你是谁，看看幻觉。问它天空为什么是蓝色的。或者为什么是绿色的。speedrun 是一个 4e19 FLOPs 能力的模型，所以有点像和幼儿园小朋友对话 :)。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

你也可以 `cat report.md` 文件，它出现在项目目录中，包含运行的"成绩单"，即一堆评估和指标。最后，你会看到一个汇总表，例如：

---

- 字符数: 333,989
- 行数: 8,304
- 文件数: 44
- Token 数（约）: 83,497
- 依赖项（uv.lock 行数）: 2,004

| 指标          | BASE     | MID      | SFT      | RL       |
|---------------|----------|----------|----------|----------|
| CORE          | 0.2219   | -        | -        | -        |
| ARC-Challenge | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy      | -        | 0.3561   | 0.3876   | -        |
| GSM8K         | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval     | -        | 0.0671   | 0.0854   | -        |
| MMLU          | -        | 0.3111   | 0.3151   | -        |
| ChatCORE      | -        | 0.0730   | 0.0884   | -        |

总运行时间: 3小时51分钟

---

（默认情况下，你的表格可能缺少 RL 数字）。有关 speedrun 脚本以及要查找和期望的更多信息，请参考我在仓库 Discussions 中发布的演练：["介绍 nanochat：100 美元能买到的最好的 ChatGPT"](https://github.com/karpathy/nanochat/discussions/1)。

## 更大的模型

毫不奇怪，100 美元不足以训练高性能的 ChatGPT 克隆。事实上，LLM 以其数百万美元的资本支出而闻名。对于我们的目的，我认为还有两个更感兴趣的规模。首先是 ~$300 级别的 d26 模型（即 depth=26），训练时间约 12 小时，在 CORE 分数上略微超过 GPT-2。第二个是 1000 美元级别（~41.6 小时），因为它是一个不错的整数。但这两个都还没有完全支持，因此还没有在这里的 master 分支中。

也就是说，为了给出一个概念，[speedrun.sh](speedrun.sh) 文件训练 GPT-2 级别模型 d26 所需的示例更改仅涉及三个更改：

```bash
...
# 你需要下载更多用于预训练的数据分片
# 获取参数数量，乘以 20 得到 token 数，乘以 4.8 得到字符数，
# 除以 2.5 亿得到分片数。待办：需要改进这个...
python -m nanochat.dataset -n 450 &
...
# 使用 --depth 增加模型大小。为了避免 OOM，将设备批次大小减半 32 -> 16：
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# 确保在中期训练期间也使用相同的设置：
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

就是这样！最需要注意的事情是确保你有足够的数据分片用于训练（否则代码会循环并对相同的训练集进行更多轮次，略微降低学习速度），以及管理你的内存/VRAM，主要是通过减少 `device_batch_size` 直到适合（脚本通过增加梯度累积循环次数自动补偿，简单地将并行计算转换为顺序计算）。

关于运行 nanochat 的计算环境的更多信息：

- 代码在 Ampere 8XA100 GPU 节点上也能正常运行，但会稍慢一些。
- 通过省略 `torchrun`，所有代码甚至可以在单个 GPU 上正常运行，并产生 ~相同的结果（代码会自动切换到梯度累积），但你必须等待 8 倍的时间。
- 如果你的 GPU 少于 80GB，你将不得不调整一些超参数，否则你会遇到 OOM / VRAM 不足。查找脚本中的 `--device_batch_size` 并减少它直到适合。例如从 32（默认）到 16、8、4、2，甚至 1。少于这个，你必须更了解你在做什么并更有创意。
- 大多数代码都是相当普通的 PyTorch，所以它应该可以在支持 PyTorch 的任何设备上运行——xpu、mps 等，但我还没有开箱即用地实现这个，所以可能需要一些调整。

## 在 CPU / MPS 上运行

nanochat 可以在 CPU 或 MPS（如果你在 Macbook 上）上运行，并且会自动尝试检测最适合运行的设备。没有 GPU 你不会走得太远，但至少你能够运行代码路径，也许可以耐心训练一个微小的 LLM。有关如何使所有运行命令更小（随时调整！）的示例，你可以参考 [dev/runcpu.sh](dev/runcpu.sh) 文件。你会看到我基本上限制所有脚本训练更小的模型，运行更少的迭代次数等。这个功能是新的，稍微粗糙（触及了很多代码），并在 2025 年 10 月 21 日的这个 [CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88) 中合并。

## 自定义

要自定义你的 nanochat，请参阅 Discussions 中的[指南：为你的 nanochat 注入身份](https://github.com/karpathy/nanochat/discussions/139)，它描述了如何通过合成数据生成并将该数据混合到中期训练和 SFT 阶段来调整 nanochat 的个性。

此外，要为 nanochat 添加新能力，请参阅[指南：在草莓中数 r（以及如何一般地添加能力）](https://github.com/karpathy/nanochat/discussions/164)。

## 问题

nanochat 设计得简洁明了。这样做的一个大优势是，我们可以将所有文件打包在一起，并将其复制粘贴到你喜欢的 LLM 以提出任意问题。例如，我喜欢使用 [files-to-prompt](https://github.com/simonw/files-to-prompt) 工具打包仓库，如下所示：

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

这包括所有 py、rs、html、toml、sh 文件，排除 `rustbpe/target` 文件夹，并选择 cxml 输出格式。所有内容都写入 `packaged.txt` 文件，目前大小约为 330KB（即远低于最先进 LLM 的 ~100K token），以及 45 个文件中的 ~8K 行代码。

或者，我建议使用 Devin/Cognition 的 [DeepWiki](https://deepwiki.com/karpathy/nanochat) 来询问这个仓库的问题。在仓库的 URL 中，只需将 github.com 更改为 deepwiki.com，你就可以开始了。

## 测试

我在这里投入不多，但有一些测试，特别是针对分词器的测试。运行例如：

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## 文件结构

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # 身份示例合成数据
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # 预训练数据分片生成
│   └── runcpu.sh                   # 如何在 CPU/MPS 上运行的小示例
├── nanochat
│   ├── __init__.py                 # 空文件
│   ├── adamw.py                    # 分布式 AdamW 优化器
│   ├── checkpoint_manager.py       # 保存/加载模型检查点
│   ├── common.py                   # 杂项小工具，生活质量
│   ├── configurator.py             # argparse 的更好替代方案
│   ├── core_eval.py                # 评估基础模型 CORE 分数（DCLM 论文）
│   ├── dataloader.py               # 分词分布式数据加载器
│   ├── dataset.py                  # 预训练数据的下载/读取工具
│   ├── engine.py                   # 使用 KV 缓存的高效模型推理
│   ├── execution.py                # 允许 LLM 将 Python 代码作为工具执行
│   ├── gpt.py                      # GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # 评估每字节位数（而不是损失）
│   ├── muon.py                     # 分布式 Muon 优化器
│   ├── report.py                   # 编写 nanochat 报告的工具
│   ├── tokenizer.py                # 类似 GPT-4 的 BPE 分词器包装器
│   └── ui.html                     # nanochat 前端的 HTML/CSS/JS
├── pyproject.toml
├── run1000.sh                      # 训练 ~$800 nanochat d32
├── rustbpe                         # 自定义 Rust BPE 分词器训练器
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md                   # 查看为什么这个存在
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py                # 基础模型：计算 CORE 分数
│   ├── base_loss.py                # 基础模型：计算每字节位数，采样
│   ├── base_train.py               # 基础模型：训练
│   ├── chat_cli.py                 # 聊天模型（SFT/Mid）：通过 CLI 对话
│   ├── chat_eval.py                # 聊天模型（SFT/Mid）：评估任务
│   ├── chat_rl.py                  # 聊天模型（SFT/Mid）：强化学习
│   ├── chat_sft.py                 # 聊天模型：训练 SFT
│   ├── chat_web.py                 # 聊天模型（SFT/Mid）：通过 WebUI 对话
│   ├── mid_train.py                # 聊天模型：中期训练
│   ├── tok_eval.py                 # 分词器：评估压缩率
│   └── tok_train.py                # 分词器：训练它
├── speedrun.sh                     # 训练 ~$100 nanochat d20
├── tasks
│   ├── arc.py                      # 多项选择科学问题
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 从任意 jsonl 对话创建任务
│   ├── gsm8k.py                    # 8K 小学数学问题
│   ├── humaneval.py                # 误称；简单的 Python 编码任务
│   ├── mmlu.py                     # 多项选择题，广泛主题
│   ├── smoltalk.py                 # HuggingFace 的 SmolTalk 聚合数据集
│   └── spellingbee.py              # 教模型拼写/计数字母的任务
├── tests
│   └── test_engine.py
│   └── test_rustbpe.py
└── uv.lock
```

## 贡献

nanochat 还远未完成。目标是改进在 < $1000 美元的预算下从头到尾可访问的微模型的最新技术。可访问性既关乎总体成本，也关乎认知复杂性——nanochat 不是一个详尽可配置的 LLM"框架"；代码库中不会有巨大的配置对象、模型工厂或 if-then-else 怪物。它是一个单一的、连贯的、极简的、可读的、可定制的、最大可分叉的"强基线"代码库，设计为从头到尾运行并产生一个具体的 ChatGPT 克隆及其成绩单。

当前的 LLM 政策：披露。提交 PR 时，请声明任何有大量 LLM 贡献且你未编写或未完全理解的部分。

## 致谢

- 名称（nanochat）源自我的早期项目 [nanoGPT](https://github.com/karpathy/nanoGPT)，它仅涵盖预训练。
- nanochat 也受到 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 的启发，它通过清晰的指标和排行榜使 nanoGPT 仓库游戏化，并借用了它的许多想法和一些预训练实现。
- 感谢 [HuggingFace](https://huggingface.co/) 提供 fineweb 和 smoltalk。
- 感谢 [Lambda](https://lambda.ai/service/gpu-cloud) 提供用于开发此项目的计算资源。
- 感谢首席 LLM 专家 🧙‍♂️ Alec Radford 的建议/指导。
- 感谢仓库管理员 Sofie [@svlandeg](https://github.com/svlandeg) 帮助管理 nanochat 的问题、拉取请求和讨论。

## 引用

如果你发现 nanochat 对你的研究有帮助，请简单地引用为：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 许可证

MIT