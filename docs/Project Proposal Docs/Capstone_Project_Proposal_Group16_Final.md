# GROUP-16 — Project Proposal Document

## 1. Project Title

**Enhancing Customer Service with Agent-powered Chatbots**

---

## 2. Brief Problem Statement

This project aims to develop an intelligent multi-agent customer service chatbot system for e-commerce platforms using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and agentic AI workflows. The system will provide fast, personalized, and context-aware customer support across multiple customer touchpoints such as chat interfaces, customer support portals, and web applications. E-commerce platforms often struggle to handle large volumes of customer queries efficiently. Traditional chatbots frequently fail to provide accurate, contextual, and personalized responses, resulting in poor customer satisfaction, delayed issue resolution, and increased support costs. Businesses also face challenges in managing complex customer scenarios such as order tracking, refunds, returns, product recommendations, and escalation handling. The proposed system will leverage multi-agent architecture, intelligent reasoning strategies, and dynamic knowledge retrieval to automate customer support workflows and improve overall customer experience.

---

## 3. Background Information

**Domain.** E-commerce customer support is a high-volume, high-cost function. Buyers expect instant, accurate, and personalized help across the full purchase lifecycle — discovery, ordering, delivery, returns, and post-sale care. Support teams must handle repetitive transactional queries (order status, refunds) alongside nuanced, emotionally charged interactions (complaints, escalations) at scale.

**Problem Description & Analysis.** Conventional rule-based and FAQ chatbots rely on rigid decision trees and keyword matching, so they break on natural-language variation, multi-intent requests, and follow-up questions. A single monolithic LLM chatbot improves fluency but introduces new failure modes: it hallucinates facts, lacks access to live order/transaction data, cannot reliably decompose complex tasks, and loses conversational context. These gaps directly cause poor resolution rates, repeat contacts, and higher operational cost.

A multi-agent, tool-augmented, retrieval-grounded design addresses these gaps by (a) routing each query to a specialized agent, (b) grounding answers in a product/policy knowledge base via RAG, (c) calling backend tools/APIs for transactional truth, (d) retaining session context through a memory layer, and (e) escalating safely when confidence is low or risk is high.

**Possible Applications.** Automated order tracking and delivery-exception handling; self-service returns, refunds, and exchanges; grounded product Q&A and comparisons; personalized cross-sell/up-sell recommendations; sentiment-driven escalation to human agents; and reusable deployment across web chat widgets, customer support portals, and mobile/web applications.

---

## 4. Motivation for Selection of the Project

- **Cutting-edge and industry-relevant.** The project centers on agentic AI systems, a current and high-impact industry direction, with strong alignment to LLMs, RAG, and multi-agent architectures.
- **Direct application of course learning.** It provides an end-to-end opportunity to implement the program's core concepts — retrieval, orchestration, tool use, and evaluation — in a single integrated system.
- **Company alignment and high applicability.** The customer-support domain has broad real-world relevance and clear business value, making the outcomes directly transferable.
- **Aligned to team strengths.** The problem sits within the team's existing competencies, supporting reliable delivery, validation, and iteration.

---

## 5. Detailed Dataset Description and Dataset Source

The project will utilize a combination of publicly available and synthetic datasets.

**1. Customer Service Conversations (real).**
- *Source:* Kaggle — *Customer Support on Twitter* dataset.
- *Content:* Real customer service interactions from various companies. Used (after preprocessing) to reconstruct ordered customer↔support conversations for RAG and memory.

**2. E-commerce Product Catalog & Reviews (real).**
- *Source:* Amazon product datasets — e.g., Stanford SNAP Amazon product graph (`snap.stanford.edu/data/amazon/productGraph`), the Amazon Reviews 2023 collection (`amazon-reviews-2023.github.io`), and the Kaggle Amazon Products 2023 (1.4M products) dataset. A subset focused on electronics and home appliances will be used for the POC.
- *Content:* Product titles, categories, IDs, ratings, current/original prices, best-seller flags, specifications, and customer reviews (including "bought together" / "viewed together" signals where available).

**3. Synthetic E-commerce Data (generated).** Generated using GPT-class language models to simulate the transactional layer needed by the agents:
- *Synthetic customer queries* — single- and multi-intent queries with persona/sentiment variation and intent labels (already implemented via an intent-driven generator).
- *Orders / Transactions* — `order_id, customer_id, order_date, status, tracking_number, est_delivery_date, actual_delivery_date, payment_status, shipping_address, total_amount`.
- *Order Items* — `order_id, product_id, quantity, unit_price, item_status`.
- *Customer Profiles* — `customer_id, name, email, loyalty_tier, location, signup_date, preferred_categories, language`.
- *Returns* — `return_id, order_id, product_id, reason, status, refund_amount, request_date`.
- *Policy / Knowledge Base documents* — return/refund, shipping, warranty, and payment policies (core RAG corpus for the Return Agent).
- *Evaluation / gold dataset* — query → expected intent, expected agent route, ideal answer, and retrieved-doc relevance labels (ground truth for evaluation).
- *Escalation trigger examples* — labeled abusive/frustrated/urgent messages for testing the Escalation Agent.

*Optional / later passes:* inventory and live pricing layer, and long-term customer memory/profile summaries.

---

## 6. Current Benchmark

*To be determined.* A specific quantitative benchmark for this task has not yet been identified. This section will be updated once a suitable baseline or comparable published result is selected. (Key methodological references are listed in Section 7.)

---

## 7. Proposed Plan

### 7.1 Methodology

**a. Approaches**
- **Multi-agent orchestration.** A Router/Orchestrator classifies intent(s), decomposes complex requests into tasks, dispatches them to specialized agents, coordinates hand-offs, and consolidates the final response.
- **Specialized agents.** Product (RAG-grounded Q&A/comparison), Order (transactional lookup), Return/Refund (policy + workflow), Recommendation (personalized suggestions), and Escalation (risk/sentiment-driven hand-off), plus a fallback agent for out-of-scope queries.
- **Retrieval-Augmented Generation.** Product and policy knowledge is chunked, embedded, and stored in a vector database; semantic retrieval grounds responses to reduce hallucination.
- **Tool calling.** Mock order-lookup, tracking, and refund-status APIs (backed by the synthetic database) provide transactional truth.
- **Conversation memory.** A session memory layer retains short-term context (order IDs, prior turns, follow-ups) for coherent multi-turn dialogue.

| Agent | Responsibility |
|---|---|
| Router / Orchestrator | Classify intent, decompose tasks, route, consolidate |
| Product Agent | Product Q&A, specs, comparisons (RAG) |
| Order Agent | Order tracking and status (tool lookup) |
| Return / Refund Agent | Return policy and refund/exchange workflows |
| Recommendation Agent | Personalized suggestions and cross-sell |
| Escalation Agent | Sentiment/risk detection and human hand-off |

**b. Packages and Tools**
- **Language & data:** Python, pandas.
- **LLM & embeddings:** an LLM API (e.g., OpenAI/GPT-class) for generation and routing; sentence-transformer or OpenAI embeddings for retrieval (provider-agnostic, finalized during implementation).
- **Vector store:** ChromaDB.
- **Structured data:** PostgreSQL (Aiven-hosted) via SQLAlchemy for orders/profiles/returns; mock APIs stubbed over the database.
- **Interface & demo:** Gradio.
- **Evaluation & ops:** RAGAS for RAG evaluation; logging and monitoring utilities; reproducible preprocessing scripts.

**c. Algorithms**
- Intent classification / query routing.
- Text chunking, embedding generation, and semantic (vector) similarity retrieval.
- Task decomposition and multi-step agent coordination (sequential/parallel execution with reconciliation).
- Candidate generation and ranking for product recommendations.
- Sentiment / risk scoring for escalation triggers.

**d. Metrics**
- Retrieval relevance / context precision and recall.
- Response accuracy and groundedness/faithfulness (RAGAS).
- Intent-routing accuracy.
- End-to-end response latency.
- Escalation precision/recall and recommendation quality.

### 7.2 Stages and Deliverables

| Phase | Module | Key Components / Deliverable |
|---|---|---|
| **Foundation** | Setup | Repo, configuration, logging |
| | Data Prep | Load, clean, normalize structured & unstructured text |
| | Synthetic Data | Prompting and query/transaction generation |
| | Embeddings | Chunking + ChromaDB storage |
| **Core Build** | RAG | Retrieval + grounded responses |
| | Router | Intent classification (product/order/return/recommend) |
| | Agents | Product (RAG), Order (lookup), Return (policy), Recommend (filters) |
| | Orchestrator | Routing, multi-step handling, fallback |
| | Memory | Session + chat history |
| **Integration** | Tools | Mock order/return APIs |
| | Demo | Notebook testing |
| | UI | Gradio chatbot |
| | Monitoring | Logging + error handling |
| **Testing** | Evaluation | Metrics + RAGAS |
| | Testing | Agent + pipeline validation |
| | Optimization | Prompt tuning + hallucination reduction |
| **Delivery** | Deployment | Local / Hugging Face Spaces |
| | Docs & Demo | README, diagrams, walkthrough |

### 7.3 Agentic E-Commerce Architecture (Representative Workflows)

**Order Tracking Workflow**
- Customer submits order status inquiry → Identify customer and order → Retrieve order and shipment details → Fetch carrier tracking information → Analyze shipment status and ETA → Provide status update and recommendations → Escalate delivery exceptions if required.

**Returns & Refunds Workflow**
- Customer requests return/refund/exchange/replacement → Validate order and purchase information → Assess return eligibility and policy compliance → Determine resolution path → Initiate return request and logistics → Process refund or replacement → Monitor completion and notify customer → Capture feedback and close case.

**Escalation Handling Workflow**
- Analyze customer interaction → Detect dissatisfaction/risk/exception → Assess severity and business impact → Generate case summary and supporting evidence → Route to appropriate support team → Monitor SLA and resolution progress → Close escalation upon resolution.

**Product Recommendation Workflow**
- Capture customer intent and context → Retrieve customer profile and behavioral data → Analyze preferences and purchase history → Generate recommendation candidates → Rank products using recommendation models → Present personalized recommendations → Capture engagement and feedback signals.

**Agent Orchestrator Workflow**
- Receive customer request → Classify intent(s) → Decompose request into executable tasks → Assign tasks to specialized agents → Coordinate agent collaboration and hand-offs → Collect outputs from participating agents → Generate consolidated response → Update context memory and workflow logs.

**Delayed Order Refund Workflow (multi-agent)**
- Customer reports delayed order and requests refund → Order Tracking Agent verifies shipment status → Returns & Refunds Agent validates refund eligibility → Escalation Agent assesses risk if thresholds exceeded → Orchestrator consolidates outcomes → Refund approved/denied/escalated → Customer notified and workflow closed.

**Task Decomposition Workflow**
- Receive complex customer request → Break request into smaller tasks → Assign each task to relevant agent → Execute tasks in sequence or parallel → Validate task outcomes → Reconcile results and prepare final response → Persist workflow history and reasoning trace.

### 7.4 Deployment Plan
The chatbot will be deployed first as a local **Gradio** application for development and demonstration, with a path to public hosting on **Hugging Face Spaces**.

### 7.5 MLOps and Automation
- Reproducible, scripted preprocessing pipelines for structured and unstructured data.
- Centralized logging and runtime monitoring with error handling.
- An automated evaluation harness (RAGAS + custom metrics) against the synthetic gold dataset.
- Versioned datasets and vector-store artifacts for repeatable experiments.

---

## 8. Preliminary Exploratory Data Analysis

### 8.1 Structured Data — Amazon Dataset

The structured Amazon dataset consists of User IDs, Product IDs, user ratings, and timestamps. Initial preprocessing was performed to ensure data quality and usability, including:
- Converting timestamps into a standardized datetime format to support temporal analysis.
- Identifying and handling invalid or inconsistent rating values.

### 8.2 Planned EDA Across Data Sources

The following EDA is planned across the three data sources (no final figures yet):
- **Twitter Customer Support (unstructured):** number of conversations and messages, turns per conversation, customer vs. support message split, inbound/outbound distribution, and message-length characteristics — informing chunking and memory design.
- **Amazon product catalog/reviews (structured):** category coverage, price distributions, rating distributions, best-seller proportions, and review-length/volume per product — guiding the POC category subset and recommendation features.
- **Synthetic queries (generated):** intent distribution (single vs. multi-intent), persona/sentiment mix, and noise characteristics — validating coverage of the target query space and balancing the evaluation set.

---

## 9. Expected Outcomes

- A working multi-agent customer-support chatbot that routes queries to specialized agents, grounds answers in RAG, calls mock transactional tools, and maintains session memory.
- Safe handling of complex, multi-intent, follow-up, escalation, and out-of-scope queries.
- A measurable evaluation report covering retrieval relevance, response groundedness/accuracy, routing accuracy, and latency.
- A reproducible codebase with documentation and a deployable Gradio demo.

---

## 10. Project Demonstration Strategy (Tentative)

A live **Gradio chatbot** walkthrough exercising representative queries across every agent — e.g., product specs/comparisons, order tracking, returns/refunds, recommendations, multi-intent requests ("return my last order and buy a replacement"), memory-dependent follow-ups, escalation triggers, and polite out-of-scope deflection — with the routing and agent decisions made visible.

---

## 11. Proposed Timeline of Project Stage Executions

*Gantt chart and weekly progress goals for the 4 Capstone Project Mentored Sessions to be inserted here.*

> Placeholder — the team's timeline chart will be added in this section.

---

## 12. Team Members' Names

- Soham De
- Gaurav Khamesra
- Prashant Singh
- Shourya Gupta
- Vikash Singh
- Richa Katare
- Sai Trinath
- Harshil Patel
- Chacha Srihari Kandimalla
- Rahul Raman
- Rakesh Ravikumar
- Tushar Pratim Sarma

---

## 13. Designated Team Coordinator's Name(s)

**Soham De, Harshil Patel**
