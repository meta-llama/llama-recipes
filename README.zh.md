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

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<h2> 使用 Llama 构建应用的官方指南 </h2>

欢迎来到帮助您快速上手使用 Llama 模型家族进行[推理 (Inference)](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/inference/)、[微调 (Fine-tuning)](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning) 以及实现[端到端应用场景 (End-to-End Use-Cases)](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases)的官方代码库。

本代码库涵盖了社区最热门的实现方案、应用案例以及针对 Llama 文本与视觉模型的最新配方实践（Recipes）。

## 最新 Llama 4 配方

* 通过 [Llama API](https://bit.ly/llama-api-main) [快速上手](./getting-started/build_with_llama_api.ipynb)
* 将 [Llama API](https://bit.ly/llama-api-main) 集成至 [WhatsApp](./end-to-end-use-cases/whatsapp_llama_4_bot/README.md)
* 使用 [Llama 4 Scout](./getting-started/build_with_llama_4.ipynb) 处理 5M 超长上下文
* 使用 [Llama 4 Maverick](./end-to-end-use-cases/research_paper_analyzer/README.md) 分析科研论文
* 使用 [Llama 4 Maverick](./end-to-end-use-cases/book-character-mindmap/README.md) 生成书籍角色思维导图

## 代码库结构

- [第三方集成 (3P Integrations)](https://github.com/meta-llama/llama-cookbook/tree/main/3p-integrations)：来自各主流 Llama 云服务商与生态平台的快速上手配方与端到端案例
- [端到端用例 (End to End Use Cases)](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases)：涵盖各垂直领域与落地场景的完整项目方案
- [快速入门 (Getting Started)](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/)：模型推理、微调与 RAG（检索增强生成）示例参考
- [src](https://github.com/meta-llama/llama-cookbook/tree/main/src/)：包含原始 `llama-recipes` 库的核心源码及微调常见问题解答

> **说明**：本仓库近期进行了架构重构，[archive-main](https://github.com/meta-llama/llama-cookbook/tree/archive-main) 为重构前历史版本的快照分支。

## 常见问题解答 (FAQ)

- **问：llama-recipes 去哪儿了？**
  **答：** 我们近期已将项目正式更名为 `llama-cookbook`。

- **问：关于微调（Fine-Tuning）遇到问题，是否有对应的解答板块？**
  **答：** 请查看微调常见问题解答 [FAQ](https://github.com/meta-llama/llama-cookbook/tree/main/src/docs/)。

- **问：部分链接失效或目录缺失：**
  **答：** 我们近期对仓库进行了重构，[archive-main](https://github.com/meta-llama/llama-cookbook/tree/archive-main) 为重构前的快照分支。

- **问：在哪里可以查阅最新模型的详细信息？**
  **答：** 请访问官方 [Llama 模型网站](https://www.llama.com)。

## 参与贡献

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解我们的行为准则（Code of Conduct）以及提交 Pull Request 的流程规范。

## 开源许可证

<!-- markdown-link-check-disable -->
查看 Meta Llama 4 的开源许可证 [LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE) 与可接受使用政策 [USE_POLICY.md](https://github.com/meta-llama/llama-models/blob/main/models/llama4/USE_POLICY.md)

查看 Meta Llama 3.3 的开源许可证 [LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE) 与可接受使用政策 [USE_POLICY.md](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/USE_POLICY.md)

查看 Meta Llama 3.2 的开源许可证 [LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) 与可接受使用政策 [USE_POLICY.md](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md)

查看 Meta Llama 3.1 的开源许可证 [LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) 与可接受使用政策 [USE_POLICY.md](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md)

查看 Meta Llama 3 的开源许可证 [LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE) 与可接受使用政策 [USE_POLICY.md](https://github.com/meta-llama/llama-models/blob/main/models/llama3/USE_POLICY.md)

查看 Meta Llama 2 的开源许可证 [LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE) 与可接受使用政策 [USE_POLICY.md](https://github.com/meta-llama/llama-models/blob/main/models/llama2/USE_POLICY.md)
<!-- markdown-link-check-enable -->

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月14日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
