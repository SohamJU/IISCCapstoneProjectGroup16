# Test Queries for the Customer Support Agent

This document provides a set of example user queries for testing the chatbot across the main support scenarios supported by this project.

## How to use this file
- Use these prompts to validate routing, fallback behavior, and response quality.
- Cover both simple and multi-intent requests.
- Include a mix of polite, frustrated, and ambiguous customer messages.

## 1. Greeting and conversational fallback
- Hi
- Hello, can you help me?
- Hey there
- Thanks for your help

## 2. Product-related queries
- What are the features of this product?
- Can you compare two products for me?
- Is this product waterproof?
- What is the warranty on this item?
- Do you have reviews for this product?
- Which product is better for gaming?
- I need a recommendation for a laptop under $500

## 3. Order-related queries
- Where is my order ORD-12345?
- Can you track my shipment?
- My order has not arrived yet
- I want to cancel my order
- What is the status of my latest purchase?
- I placed an order yesterday and need an update

## 4. Return and refund queries
- I want to return my order
- Can I get a refund for a damaged item?
- My item arrived late and I need a replacement
- I received the wrong product and want to return it
- What is your return policy for opened items?
- I need to exchange this product

## 5. Recommendation queries
- Suggest a similar product that is cheaper
- What should I buy instead of this one?
- Recommend something good for home office use
- Give me the best option for a gift under $50
- I need an alternative with better battery life

## 6. Escalation and human support requests
- I want to speak to a human agent
- This is urgent, please escalate this case
- I need a manager
- I am angry and want someone to help me directly
- This is a legal or fraud issue

## 7. Mixed-intent and complex queries
- I want to return my order and also find a cheaper replacement
- My package is delayed and I need to know the status of my shipment
- I received a defective item and want a refund and a recommendation
- Can you compare this product and tell me if it is eligible for return?
- I need help with my order and also want to speak to a human agent

## 8. Ambiguous or unsupported queries
- Tell me a joke
- What is the weather today?
- Can you write code for me?
- Who won the election?
- Help me with my homework

## 9. Persona-based test cases
### Polite
- Hi, I am looking for help with my order.
- Could you help me compare these products?

### Frustrated
- This is taking too long. I need my order status now.
- I received the wrong item and I want this fixed immediately.

### Angry
- I want this resolved now. This is unacceptable.
- I need a human manager right away.

### Confused
- I am not sure what happened to my delivery.
- I bought this item but I do not know if I can return it.

## 10. Schema-aligned example intents
These queries align with the project’s synthetic data themes:
- Customer profile: loyalty tier, city, state, preferred categories
- Orders: status, tracking, delivery dates, payment status
- Returns: reason, refund, approval status
- Products: price, warranty, features, reviews, category
- Customer queries: product comparison, returns, payment issues, account updates

## Suggested evaluation checklist
For each test query, verify that the chatbot:
- routes to the correct agent or fallback path
- responds with a helpful, human-like tone
- stays within the support domain when appropriate
- handles mixed intents in a clear step-by-step way
- escalates politely when the user asks for a human
