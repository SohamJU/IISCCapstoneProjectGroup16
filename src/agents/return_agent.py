"""Handles return and refund queries.

This module provides a lightweight `ProductReturnAgent` that reads the
project's return policy and exposes simple helpers to summarize the policy
and respond to common return/refund questions. The implementation is kept
minimal so it can be extended later to call other services (order lookup,
RAG, or LLMs).
"""

from pathlib import Path
from typing import Optional
import csv


class ProductReturnAgent:
	"""Agent that answers return-related questions using the project's
	`data/knowledge_base/return_policy.md` file as the source of truth.

	Usage:
		agent = ProductReturnAgent()
		summary = agent.summarize()
		answer = agent.handle_query("How do I return an item?")
	"""

	def __init__(self, policy_path: Optional[Path] = None) -> None:
		self.policy_path = (
			Path(policy_path)
			if policy_path
			else Path(__file__).resolve().parents[2] / "data" / "knowledge_base" / "return_policy.md"
		)
		self.policy_text = self._load_policy()

	def _load_policy(self) -> str:
		try:
			return self.policy_path.read_text(encoding="utf-8")
		except Exception:
			return """Return policy not available. Please check the data/knowledge_base directory."""

	def summarize(self, max_chars: int = 1000) -> str:
		"""Return a short summary (top portion) of the policy text."""
		return self.policy_text[:max_chars].strip()

	def handle_query(self, query: str) -> str:
		"""Provide a simple, deterministic response to common return questions.

		This is intentionally rule-based and conservative: it extracts a small
		set of intents from the query and returns text pulled from the policy
		so the agent's answers remain consistent with the documented policy.
		"""
		q = (query or "").lower()

		if "how" in q and "return" in q:
			# return the process section if present
			if "Initiate your return" in self.policy_text:
				# return lines around the phrase to give actionable steps
				start = self.policy_text.find("Initiate your return")
				snippet = self.policy_text[start : start + 800]
				return snippet.strip()
			return "To return an item, follow the steps on your Order History page."

		if "refund" in q or "refunds" in q or "refund status" in q:
			if "Refund issued" in self.policy_text:
				start = self.policy_text.find("Refund issued")
				snippet = self.policy_text[start : start + 400]
				return snippet.strip()
			return "Refund timelines vary; check your order's return status in Order History."

		if "restock" in q or "restocking" in q or "restocking fee" in q:
			if "Restocking Fees" in self.policy_text:
				start = self.policy_text.find("Restocking Fees")
				snippet = self.policy_text[start : start + 400]
				return snippet.strip()
			return "Some returns may incur restocking fees depending on timing and condition."

		# Fallback: return a short policy summary
		return self.summarize(1200) or "Return policy is not available."

	def lookup_order(self, order_id: str) -> str:
		"""Look up an order in `data/synthetic/orders.csv` and return a short status.

		If the synthetic dataset is not available, return a helpful message.
		"""
		orders_csv = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "orders.csv"
		if not orders_csv.exists():
			return "Order lookup data not available. Run the synthetic data pipeline to generate test orders."

		try:
			with orders_csv.open(encoding="utf-8") as fh:
				reader = csv.DictReader(fh)
				for row in reader:
					if row.get("order_id") == order_id:
						return (
							f"Order {order_id}: status={row.get('status')}, "
							f"tracking={row.get('tracking_number')}, delivered={row.get('actual_delivery_date')}, "
							f"paid={row.get('payment_status')}, total=${row.get('total_amount')}"
						)
		except Exception:
			return "Error reading orders data."

		return f"Order {order_id} not found in test dataset."

	def rag_query(self, query: str) -> str:
		"""Retrieve the most relevant snippet from the return policy using
		a lightweight paragraph-level scoring approach.

		This is a small, dependency-free RAG replacement: it splits the
		policy into paragraphs, tokenizes the query and paragraphs, scores
		each paragraph by token overlap, and returns the highest-scoring
		paragraph (or a short summary if nothing matches).
		"""
		if not self.policy_text:
			return "Return policy not available."

		q = (query or "").lower()
		# simple tokenization: split on non-alphanumeric
		import re

		def tokens(text: str):
			return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]

		q_tokens = set(tokens(q))

		# split into paragraphs (blocks separated by blank lines)
		paragraphs = [p.strip() for p in self.policy_text.split("\n\n") if p.strip()]
		best_score = 0
		best_para = None

		for para in paragraphs:
			para_tokens = set(tokens(para))
			if not para_tokens:
				continue
			# overlap score normalized by paragraph length
			overlap = q_tokens & para_tokens
			score = len(overlap) / max(1, len(para_tokens))
			# boost exact phrase matches
			if q and q in para.lower():
				score += 0.25

			if score > best_score:
				best_score = score
				best_para = para

		# if we found a paragraph with a non-zero score, return it
		if best_para and best_score > 0:
			# return up to ~800 chars to keep output concise
			return best_para[:800].strip()

		# fallback to returning a short summary
		return self.summarize(800)


__all__ = ["ProductReturnAgent"]

