"""
Quantum-Enhanced Agent  (NumPy)
================================
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, Sequence

import numpy as np

from hybrid.interfaces.model import HybridQuantumLLM
from agents.reasoning.quantum_reasoning import QuantumReasoningModule
from agents.memory.quantum_memory import QuantumMemoryNetwork

logger = logging.getLogger(__name__)


class QuantumTool:
    name: str = "unnamed_tool"
    description: str = ""
    def run(self, input_text: str) -> str:
        raise NotImplementedError


class QuantumAgent:
    def __init__(
        self,
        model: HybridQuantumLLM,
        reasoning_module: QuantumReasoningModule,
        memory: QuantumMemoryNetwork,
        tools: Optional[Sequence[QuantumTool]] = None,
        max_steps: int = 5,
    ):
        self.model = model
        self.reasoner = reasoning_module
        self.memory = memory
        self.tools = list(tools or [])
        self.max_steps = max_steps

    def run(self, query: str, tokenizer=None) -> dict[str, Any]:
        steps: list[dict[str, str]] = []
        context_embedding = self._encode_text(query, tokenizer)

        for step_idx in range(self.max_steps):
            thought = self._think(query, steps, tokenizer)

            if self.tools:
                tool_embeddings = self._tool_embeddings(tokenizer)
                memory_state = self._latest_memory()

                reasoning_result = self.reasoner(
                    context=context_embedding,
                    options=tool_embeddings,
                    memory_state=memory_state,
                )
                decision_probs = reasoning_result["decision_probs"]
                selected_idx = int(np.argmax(decision_probs)) % len(self.tools)
                selected_tool = self.tools[selected_idx]

                observation = selected_tool.run(thought)
                obs_embedding = self._encode_text(observation, tokenizer)
                self.memory.store(obs_embedding)

                steps.append({
                    "thought": thought,
                    "tool": selected_tool.name,
                    "observation": observation,
                })
                context_embedding = obs_embedding
            else:
                steps.append({"thought": thought, "tool": None, "observation": None})
                break

            if "FINAL ANSWER" in thought.upper():
                break

        return {
            "answer": steps[-1]["thought"] if steps else "",
            "steps": steps,
            "memory_size": self.memory.size,
        }

    def _encode_text(self, text: str, tokenizer=None) -> np.ndarray:
        if tokenizer is not None:
            ids = np.array(tokenizer.encode(text))[np.newaxis, :]
        else:
            warnings.warn(
                "No tokenizer provided; using lossy ord() fallback.",
                stacklevel=2,
            )
            ids = np.array([[ord(c) % self.model.vocab_size for c in text[:64]]])
        emb = self.model.embedding(ids)
        return emb.mean(axis=1).squeeze(0)

    def _think(self, query: str, steps: list, tokenizer=None) -> str:
        prompt = query
        if steps and steps[-1].get("observation"):
            prompt = f"{query} | Observation: {steps[-1]['observation']}"
        if tokenizer is not None:
            ids = np.array(tokenizer.encode(prompt))[np.newaxis, :]
            gen_ids = self.model.generate(ids, max_new_tokens=16, temperature=0.7)
            return tokenizer.decode(gen_ids[0])
        else:
            ids = np.array([[ord(c) % self.model.vocab_size for c in prompt[:64]]])
            self.model.generate(ids, max_new_tokens=8, temperature=0.7)
            return f"[quantum-thought-step-{len(steps)}]"

    def _tool_embeddings(self, tokenizer=None) -> np.ndarray:
        if not self.tools:
            return np.zeros(self.model.d_model)
        embeddings = [self._encode_text(t.description or t.name, tokenizer) for t in self.tools]
        return np.mean(embeddings, axis=0)

    def _latest_memory(self) -> Optional[np.ndarray]:
        if self.memory.size == 0:
            return None
        recent = list(self.memory._buffer)[-1]
        padded = np.zeros(self.model.d_model)
        padded[:min(len(recent), self.model.d_model)] = np.array(recent[:self.model.d_model])
        return padded


class QuantumMultiAgentCoordinator:
    def __init__(self, agents: Sequence[QuantumAgent]):
        self.agents = list(agents)

    def collaborative_solve(self, problem: str, tokenizer=None) -> dict:
        sub_answers = []
        for i, agent in enumerate(self.agents):
            result = agent.run(f"[Sub-task {i+1}/{len(self.agents)}] {problem}", tokenizer)
            sub_answers.append(result)
        return {
            "sub_answers": sub_answers,
            "combined_answer": " | ".join(r["answer"] for r in sub_answers),
            "total_steps": sum(len(r["steps"]) for r in sub_answers),
        }
