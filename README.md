# 大模型工业化落地路径图

## 预训练
\[TBD;\]

## 继续训练
继续训练可分为三个阶段：指令泛化、价值观对齐和下游任务精调。

### 指令泛化
指令泛化旨在训练模型学会遵循各式各样的指令完成任务的能力，其核心目的在于提升模型在下游任务上的泛化能力。大量实验表明，经过指令泛化后的大模型，相比未泛化之前，在多任务上的 zero-shot 或 few-shot 能力有着较大幅度的提升；但值得注意的是，指令泛化并不能保证大模型在下游任务上的表现达到令人满意的水平，即指令泛化只能教会大模型更好地理解“要做什么”而无法真正教会大模型“要怎么做”。

涉及到指令泛化的中文知名开源大模型项目：
- [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna)
- [pandallm](https://github.com/dandelionsllm/pandallm)
- [MOSS](https://github.com/dandelionsllm/pandallm)
- [BELLE](https://github.com/LianjiaTech/BELLE)
- [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [baize-chatbot](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

#### 数据集
##### 普通指令数据集
\[TBD;\]

##### 思维链数据集
\[TBD;\]

##### 插件数据集
\[TBD;\]

#### 训练策略

##### 全参数微调
全参数微调适用于数据量较大的场景，此时也可视为进行模型的二次预训练，其优点在于可能可以达到“知识注入”和“知识更新”的效果，缺点是成本投入较大。

若要使用全参数微调，推荐的工具有：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [LOMO](https://github.com/OpenLMLab/LOMO)

##### 参数高效微调
参数高效微调是当下最流行的大模型微调方法，主流算法有 P-tuning-v2, LoRA 和 Adaption-Prompt(LLaMA-Adapter)，不少实验表明使用参数高效微调策略在少量（几万条甚至更少）高质量指令数据集上能取得比在大规模数据集上进行全参数微调更好的效果。

目前流行的参数高效微调工具包有:
- [peft](https://github.com/huggingface/peft)

相关知名项目有:
- [dolly](https://github.com/databrickslabs/dolly)
 
### 价值观对齐
价值观对齐，即通过人类反馈的强化学习机制(RLHF)来调整大模型的输出以和人类价值观保持一致。值得注意的是，价值观对齐有可能损害大模型在解决具体问题上的能力，因此需要根据企业具体的应用场景以决定是否需要进行价值观对齐。

相关知名项目：
- [deepspeed-chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)
- [colossalai-chat](https://medium.com/pytorch/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)

### 下游任务精调
在将大模型真正应用于下游具体任务上时，往往还需要在特定任务上进行最后的精调，以达到可上线标准。对于大模型，在这一阶段通常推荐采用参数高效微调策略，这是因为一来能够以低成本快速完成训练，二来能够避免全参数微调时大模型在单一任务上过拟合而造成遗忘灾难。[Modelize-AI](https://www.modelize.ai/) 提供了从数据标注到大模型微调和部署的一站式服务，能够以更低的成本拥有定制化的大模型，帮助企业无痛接入大模型，高效赋能各项业务场景。


## 推理部署

### 模型部署工具
- [text-generation-inference](https://github.com/huggingface/text-generation-inference) HuggingFace 在其生产环节使用的推理框架
- [vLLM](https://github.com/vllm-project/vllm) 提出 PageAttention, 推理吞吐相比 TGI 可提升三倍
- [FlexGen](https://github.com/FMInference/FlexGen) 专为吞吐量优化的推理框架
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) 英伟达大模型推理新框架，针对 FP8 进行专门优化
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 英伟达 transformer 模型推理框架，支持多种主流模型和文本生成方案
- [EnergonAI](https://github.com/hpcaitech/EnergonAI) ColossalAI 的姊妹项目，能够轻松兼容基于 ColossalAI 训练的模型和快速地修改模型代码以实现张量并行
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) DeepSpeed 的姊妹项目，专为推理而设计
- [petals](https://github.com/bigscience-workshop/petals) 雾计算概念项目，使用公共计算池中的边缘设备解决算力不足的问题

### 模型压缩工具
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 简单易用且高度灵活的大模型 2bit~8bit 量化工具包

## 其他话题

### 成本估算
- [训练成本估算](https://zhuanlan.zhihu.com/p/630582034)
- [推理成本估算](https://kipp.ly/blog/transformer-inference-arithmetic/)
