<h1 align="center"> Llama Cookbook </h1>
<p align="center">
	<a href="https://llama.developer.meta.com/join_waitlist?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img src="https://img.shields.io/badge/Llama_API-Join_Waitlist-brightgreen?logo=meta" /></a>
	<a href="https://llama.developer.meta.com/docs?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img src="https://img.shields.io/badge/Llama_API-Documentation-4BA9FE?logo=meta" /></a>

</p>
<p align="center">
	<a href="https://github.com/meta-llama/llama-models/blob/main/models/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img alt="Llama Model cards" src="https://img.shields.io/badge/Llama_OSS-Model_cards-green?logo=meta" /></a>
	<a href="https://www.llama.com/docs/overview/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img alt="Llama Documentation" src="https://img.shields.io/badge/Llama_OSS-Documentation-4BA9FE?logo=meta" /></a>
	<a href="https://huggingface.co/meta-llama"><img alt="Hugging Face meta-llama" src="https://img.shields.io/badge/Hugging_Face-meta--llama-yellow?logo=huggingface" /></a>

</p>
<p align="center">
	<a href="https://github.com/meta-llama/synthetic-data-kit"><img alt="Llama Tools Syntethic Data Kit" src="https://img.shields.io/badge/Llama_Tools-synthetic--data--kit-orange?logo=meta" /></a>
	<a href="https://github.com/meta-llama/llama-prompt-ops"><img alt="Llama Tools Syntethic Data Kit" src="https://img.shields.io/badge/Llama_Tools-llama--prompt--ops-orange?logo=meta" /></a>
</p>
<h2> 基于 Llama 构建应用的官方指南 </h2>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

欢迎来到官方指南代码库，帮助您快速上手使用 Llama 模型家族进行[推理（Inference）](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/inference/)、[微调（Fine-tuning）](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning)以及[端到端场景应用（End-to-end use-cases）](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases)开发。

本仓库涵盖了社区最热门的实现方案、应用场景以及针对 Llama 文本与视觉多模态模型的最新实用指南（Recipes）。

## 最新 Llama 4 实战方案

* [快速入门](./getting-started/build_with_llama_api.ipynb)：使用 [Llama API](https://bit.ly/llama-api-main)
* 将 [Llama API](https://bit.ly/llama-api-main) 与 [WhatsApp](./end-to-end-use-cases/whatsapp_llama_4_bot/README.md) 进行集成
* 使用 [Llama 4 Scout](./getting-started/build_with_llama_4.ipynb) 处理 5M 超长上下文
* 使用 [Llama 4 Maverick](./end-to-end-use-cases/research_paper_analyzer/README.md) 深度分析学术研究论文
* 使用 [Llama 4 Maverick](./end-to-end-use-cases/book-character-mindmap/README.md) 基于图书内容生成角色思维导图

## 代码库结构

- [第三方服务集成 (3P Integrations)](https://github.com/meta-llama/llama-cookbook/tree/main/3p-integrations)：来自各类 Llama 云服务商与合作伙伴的入门指南与端到端用例
- [端到端实际用例 (End to End Use Cases)](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases)：顾名思义，涵盖各种垂直领域与实际业务场景
- [入门指南 (Getting Started)](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/)：推理、微调与 RAG 示例的权威参考
- [源码目录 (src)](https://github.com/meta-llama/llama-cookbook/tree/main/src/)：包含原 `llama-recipes` 核心库源码以及关于微调的常见问答

> 注：我们近期对代码库进行了架构重构，[archive-main](https://github.com/meta-llama/llama-cookbook/tree/archive-main) 是重构前的快照分支。

## 常见问答 (FAQ)

- **问：** `llama-recipes` 去哪了？
  **答：** 我们近期已将项目从 `llama-recipes` 更名为 `llama-cookbook`。

- **问：** 我有一些关于微调（Fine-Tuning）的疑问，是否有专门的解答板块？
  **答：** 请在[此处](https://github.com/meta-llama/llama-cookbook/tree/main/src/docs/)查看微调常见问答（FAQ）。

- **问：** 某些链接失效或某些文件夹缺失了：
  **答：** 我们近期对仓库进行了重构，[archive-main](https://github.com/meta-llama/llama-cookbook/tree/archive-main) 是重构前保存的快照分支。

- **问：** 在哪里可以了解最新模型的详细信息？
  **答：** 请访问官方 [Llama 模型网站](https://www.llama.com)。

## 参与贡献 (Contributing)

有关行为准则以及向我们提交拉取请求（PR）的流程规范，请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证 (License)
<!-- markdown-link-check-disable -->
关于 Meta Llama 4，请在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE)查看许可证文件，并在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama4/USE_POLICY.md)查看可接受使用政策（AUP）。

关于 Meta Llama 3.3，请在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE)查看许可证文件，并在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/USE_POLICY.md)查看可接受使用政策（AUP）。

关于 Meta Llama 3.2，请在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)查看许可证文件，并在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md)查看可接受使用政策（AUP）。

关于 Meta Llama 3.1，请在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)查看许可证文件，并在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md)查看可接受使用政策（AUP）。

关于 Meta Llama 3，请在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE)查看许可证文件，并在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama3/USE_POLICY.md)查看可接受使用政策（AUP）。

关于 Meta Llama 2，请在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE)查看许可证文件，并在[此处](https://github.com/meta-llama/llama-models/blob/main/models/llama2/USE_POLICY.md)查看可接受使用政策（AUP）。
<!-- markdown-link-check-enable -->

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月18日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
