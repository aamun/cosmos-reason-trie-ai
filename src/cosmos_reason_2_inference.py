# Inference Class for Cosmos Reason 2 (improved)
class CosmosReason2Inference:
    """
    Inference wrapper for Cosmos Reason 2 model via vLLM.
    """

    def __init__(
        self,
        model_path,
        nframes=8,
        max_tokens=256,
        # perf/stability knobs
        max_model_len=2048,
        dtype="half",              # T4 -> half
        enforce_eager=True,        # fast startup, avoids cudagraph/compile surprises
        gpu_memory_utilization=0.90,
        # sampling knobs
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.05,
        seed=0,
    ):
        self.model_path = model_path
        self.nframes = nframes

        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.gpu_memory_utilization = gpu_memory_utilization

        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed

        self.llm = None
        self.processor = None
        self.sampling_params = None

    def load_model(self):
        """Load the model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor
            import torch, gc

            # Reduce fragmentation between notebook cells
            torch.cuda.empty_cache()
            gc.collect()

            print(f"🔄 Loading model from: {self.model_path}")
            print(
                f"   max_model_len={self.max_model_len} | dtype={self.dtype} | "
                f"enforce_eager={self.enforce_eager}"
            )

            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=1,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                enforce_eager=self.enforce_eager,
                trust_remote_code=True,
                limit_mm_per_prompt={"video": 1, "image": 0},
                gpu_memory_utilization=self.gpu_memory_utilization,
                seed=self.seed,
            )

            # Processor for chat template (Qwen3-VL style)
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )

            print("✅ Model loaded successfully!")
            return True

        except ImportError as ie:
            print(f"⚠️ Import error: {ie}")
            print("   Try: pip install -U vllm transformers qwen-vl-utils")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def set_sampling(
        self,
        max_tokens=None,
        temperature=None,
        top_p=None,
        repetition_penalty=None,
    ):
        """Update sampling params without recreating the engine."""
        if self.llm is None:
            print("⚠️ Model not loaded yet.")
            return

        from vllm import SamplingParams

        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty

        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

    def query(self, video_path, question, system_prompt="You are a helpful assistant."):
        """Query the model with a video and question."""
        if self.llm is None or self.processor is None:
            print("⚠️ Model not loaded. Call load_model() first.")
            return None

        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "video", "video": str(video_path), "nframes": self.nframes},
                    {"type": "text", "text": question}
                ]}
            ]

            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            _, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )

            model_input = {
                "prompt": text_prompt,
                "multi_modal_data": {"video": video_inputs},
                "mm_processor_kwargs": video_kwargs
            }

            outputs = self.llm.generate([model_input], self.sampling_params)
            return outputs[0].outputs[0].text

        except Exception as e:
            print(f"❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def close(self):
        """Clean shutdown (prevents NCCL warning + frees memory)."""
        try:
            import torch, gc
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

            self.llm = None
            self.processor = None
            self.sampling_params = None

            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    def __del__(self):
        self.close()
