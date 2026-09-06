<h1 align="center"> Llama Cookbook 实战指南 </h1>
<p align="center">
	<a href="https://llama.developer.meta.com/join_waitlist?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img src="https://img.shields.io/badge/Llama_API-Join_Waitlist-brightgreen?logo=meta" /></a>
	<a href="https://llama.developer.meta.com/docs?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img src="https://img.shields.io/badge/Llama_API-Documentation-4BA9FE?logo=meta" /></a>
</p>
<p align="center">
	<a href="https://github.com/meta-llama/llama-models/blob/main/models/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img alt="Llama 模型卡片" src="https://img.shields.io/badge/Llama_OSS-Model_cards-green?logo=meta" /></a>
	<a href="https://www.llama.com/docs/overview/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img alt="Llama 官方文档" src="https://img.shields.io/badge/Llama_OSS-Documentation-4BA9FE?logo=meta" /></a>
	<a href="https://huggingface.co/meta-llama"><img alt="Hugging Face meta-llama" src="https://img.shields.io/badge/Hugging_Face-meta--llama-yellow?logo=huggingface" /></a>
</p>
<p align="center">
	<a href="https://github.com/meta-llama/synthetic-data-kit"><img alt="Llama Tools 合成数据套件" src="https://img.shields.io/badge/Llama_Tools-synthetic--data--kit-orange?logo=meta" /></a>
	<a href="https://github.com/meta-llama/llama-prompt-ops"><img alt="Llama Tools Prompt 运维套件" src="https://img.shields.io/badge/Llama_Tools-llama--prompt--ops-orange?logo=meta" /></a>
</p>

<p align="center">
    <a href="README.md">English</a> · <b>简体中文</b>
</p>

<h2> 基于 Llama 构建 AI 应用的官方实战指南 </h2>

欢迎来到使用 Llama 模型家族构建 AI 应用的官方代码与实战指南库！本仓库旨在帮助你快速上手[模型推理 (Inference)](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/inference/)、[微调优化 (Fine-Tuning)](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning) 以及丰富的[端到端行业实战用例 (End-to-End Use-Cases)](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases)。

本仓库涵盖社区最流行的开发方案、经典落地场景，以及针对 Llama 文本与视觉多模态模型的最新实战配方（Recipes）。

## 最新 Llama 4 实战配方 (Latest Llama 4 recipes)

* [快速入门 Llama API](./getting-started/build_with_llama_api.ipynb)（通过 [Llama API](https://bit.ly/llama-api-main)）
* 将 [Llama API](https://bit.ly/llama-api-main) 集成至 [WhatsApp 聊天机器人](./end-to-end-use-cases/whatsapp_llama_4_bot/README.md)
* 基于 [Llama 4 Scout](./getting-started/build_with_llama_4.ipynb) 实现 500 万超长上下文（5M Long Context）处理
* 使用 [Llama 4 Maverick](./end-to-end-use-cases/research_paper_analyzer/README.md) 深度分析学术研究论文
* 使用 [Llama 4 Maverick](./end-to-end-use-cases/book-character-mindmap/README.md) 根据整本书籍生成人物关系思维导图

## 仓库目录架构 (Repository Structure)

- [第三方生态集成 (3P Integrations)](https://github.com/meta-llama/llama-cookbook/tree/main/3p-integrations)：来自各类主流 Llama 云服务商与提供商的快速上手指南与端到端实战
- [端到端落地实战 (End to End Use Cases)](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases)：涵盖跨多个行业领域与业务场景的生产级应用示例
- [快速入门 (Getting Started)](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/)：模型推理、微调训练与检索增强生成（RAG）的参考代码与教程
- [核心源码与文档 (src)](https://github.com/meta-llama/llama-cookbook/tree/main/src/)：包含原 `llama-recipes` 核心库源码以及针对微调的常见问题解答

> 注意：我们近期对仓库结构进行了重构，[archive-main](https://github.com/meta-llama/llama-cookbook/tree/archive-main) 是重构前的快照归档分支。

## 常见问题解答 (FAQ)

- **问：之前的 `llama-recipes` 发生了什么？**  
  **答：** 我们近期将 `llama-recipes` 正式重命名为 `llama-cookbook`，以更清晰地承载官方实战指南的定位。

- **问：我对模型微调（Fine-Tuning）有一些疑问，哪里可以找到解答？**  
  **答：** 请查阅微调常见问题解答文档 [Fine-Tuning FAQ](https://github.com/meta-llama/llama-cookbook/tree/main/src/docs/)。

- **问：部分链接失效或文件夹缺失：**  
  **答：** 我们近期对仓库进行了重构，重构前的快照历史可以在 [archive-main](https://github.com/meta-llama/llama-cookbook/tree/archive-main) 分支中找到。

- **问：在哪里可以获取最新发布的 Llama 模型详细信息？**  
  **答：** 请访问 [Llama 模型官方主页](https://www.llama.com)。

## 参与贡献 (Contributing)

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解我们的行为准则以及向项目提交 Pull Request 的流程规范。

## 开源协议与使用规范 (License)
<!-- markdown-link-check-disable -->
查看 Meta Llama 4 的[许可证文件](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE)与[可接受使用政策](https://github.com/meta-llama/llama-models/blob/main/models/llama4/USE_POLICY.md)

查看 Meta Llama 3.3 的[许可证文件](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE)与[可接受使用政策](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/USE_POLICY.md)

查看 Meta Llama 3.2 的[许可证文件](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)与[可接受使用政策](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md)

查看 Meta Llama 3.1 的[许可证文件](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)与[可接受使用政策](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md)

查看 Meta Llama 3 的[许可证文件](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE)与[可接受使用政策](https://github.com/meta-llama/llama-models/blob/main/models/llama3/USE_POLICY.md)

查看 Meta Llama 2 的[许可证文件](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE)与[可接受使用政策](https://github.com/meta-llama/llama-models/blob/main/models/llama2/USE_POLICY.md)
<!-- markdown-link-check-enable -->

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）翻译维护，最后同步更新于 2026年9月6日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
