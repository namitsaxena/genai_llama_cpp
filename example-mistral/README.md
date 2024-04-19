# Running LLMs locally using Docker and llama-cpp

## llama-cpp-python
llama-cpp-python is a Python binding for llama.cpp. It supports inference for many LLMs models, which can be accessed on Hugging Face. This notebook goes over how to run llama-cpp-python within LangChain. Note: new versions of llama-cpp-python use GGUF model files [[python.langchain.com](https://python.langchain.com/docs/integrations/llms/llamacpp/)]

Installing llama-cpp locally
* llama-cpp-python couldn't be installed on mac directly (due to missing libraries probably)

Image 
* Downloaded from Hugging Face
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf


## Building the Image
Building and Running both worked on old mac but took several minutes (20+ minutes approximately)
```
| => docker build . -t llama-cpp-mistral:2
```

Images Built
```
| => docker images
REPOSITORY                                                              TAG             IMAGE ID       CREATED             SIZE
llama-cpp-mistral                                                       2               94c8e14a8f31   35 minutes ago      5.57GB
llama-cpp-mistral                                                       1               c7baf44d4f1d   About an hour ago   4.28GB

```

Execution
```
| => docker run -it llama-cpp-mistral:2
llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from mistral-7b-instruct-v0.2.Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = mistralai_mistral-7b-instruct-v0.2
llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                       llama.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  11:                          general.file_type u32              = 15
llama_model_loader: - kv  12:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  19:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  20:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  21:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {{ bos_token }}{% for message in mess...
llama_model_loader: - kv  23:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 7.24 B
llm_load_print_meta: model size       = 4.07 GiB (4.83 BPW) 
llm_load_print_meta: general.name     = mistralai_mistral-7b-instruct-v0.2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.11 MiB
llm_load_tensors:        CPU buffer size =  4165.37 MiB
.................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =    64.00 MiB
llama_new_context_with_model: KV self size  =   64.00 MiB, K (f16):   32.00 MiB, V (f16):   32.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.12 MiB
llama_new_context_with_model:        CPU compute buffer size =    81.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 1
AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | 
Model metadata: {'tokenizer.chat_template': "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}", 'tokenizer.ggml.add_eos_token': 'false', 'tokenizer.ggml.padding_token_id': '0', 'tokenizer.ggml.unknown_token_id': '0', 'tokenizer.ggml.eos_token_id': '2', 'general.architecture': 'llama', 'llama.rope.freq_base': '1000000.000000', 'llama.context_length': '32768', 'general.name': 'mistralai_mistral-7b-instruct-v0.2', 'tokenizer.ggml.add_bos_token': 'true', 'llama.embedding_length': '4096', 'llama.feed_forward_length': '14336', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'llama.rope.dimension_count': '128', 'tokenizer.ggml.bos_token_id': '1', 'llama.attention.head_count': '32', 'llama.block_count': '32', 'llama.attention.head_count_kv': '8', 'general.quantization_version': '2', 'tokenizer.ggml.model': 'llama', 'general.file_type': '15'}
Guessed chat format: mistral-instruct

llama_print_timings:        load time =   15549.14 ms
llama_print_timings:      sample time =     405.43 ms /   215 runs   (    1.89 ms per token,   530.31 tokens per second)
llama_print_timings: prompt eval time =   15546.32 ms /    45 tokens (  345.47 ms per token,     2.89 tokens per second)
llama_print_timings:        eval time = 1894865.28 ms /   214 runs   ( 8854.51 ms per token,     0.11 tokens per second)
llama_print_timings:       total time = 1915566.08 ms /   259 tokens
{'id': 'cmpl-8744a3b0-dd3c-49e1-b532-41111de801ad', 'object': 'text_completion', 'created': 1713499720, 'model': 'mistral-7b-instruct-v0.2.Q4_K_M.gguf', 'choices': [{'text': "<s>[INST] <<SYS>>\nYou are a helpful software developer\n<</SYS>>\nWhat do you know about BPMN 2.0 and Imixs-Workflow? [/INST] BPMN 2.0, or Business Process Model and Notation Version 2.0, is an standardized graphical notation for modeling business processes. It provides a common language for business analysts and technical developers to design, document, and execute business processes. With BPMN 2.0, it's possible to create complex process models with various elements like activities, gateways, events, pools, lanes, and connectors.\n\nImixs-Workflow is an open-source workflow engine based on the BPMN 2.0 standard. It offers a modular design for managing business processes, forms, document handling, and human interaction in Java EE applications. Imixs-Workflow enables you to model, execute, and monitor your business processes using BPMN diagrams, making it easier for organizations to automate their workflows and streamline their operations. Additionally, it supports integration with various databases, document management systems, and other external systems through APIs or plugins.", 'index': 0, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 45, 'completion_tokens': 214, 'total_tokens': 259}}
```

## References
* How to Run LLMs in a Docker Container - Ralph's Open Source Blog[[ralph.blog.imixs.com](https://ralph.blog.imixs.com/2024/03/19/how-to-run-llms-in-a-docker-container/)]
* How to run Llama 2 locally on CPU + serving it as a Docker container | by Nikolay Penkov | Medium[[medium.com](https://medium.com/@penkow/how-to-run-llama-2-locally-on-cpu-docker-image-731eae6398d1)]
* https://github.com/abetlen/llama-cpp-python
  - Packages & Image: https://github.com/abetlen/llama-cpp-python/pkgs/container/llama-cpp-python
* https://python.langchain.com/docs/integrations/llms/llamacpp/  
* https://github.com/ggerganov/llama.cpp