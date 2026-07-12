"""Generate e-commerce policy / knowledge base Markdown documents.

Creates 4 detailed policy documents for the RAG corpus:
  - return_policy.md
  - shipping_policy.md
  - warranty_policy.md
  - payment_policy.md

Supports three modes via --provider:
  - "static"  (default) — high-quality hardcoded policy text
  - "gemini"  — generate via Google Gemini API (requires GOOGLE_API_KEY)
  - "openai"  — generate via OpenAI API (requires OPENAI_API_KEY)

Usage:
    python -m src.data.pipeline.generate_policies
    python -m src.data.pipeline.generate_policies --provider gemini
    python -m src.data.pipeline.generate_policies --provider openai
"""

from __future__ import annotations

from typing import Any, cast
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.data import KNOWLEDGE_BASE_DIR

# ── Company context for LLM prompts ──────────────────────────────────────
COMPANY_CONTEXT = """
You are writing official customer-facing policy documents for "ShopSmart
Electronics", an online retailer specializing in consumer electronics and home
appliances. The store sells products like headphones, laptops, monitors,
smart-home devices, kitchen appliances, air purifiers, etc.

Write detailed, realistic, and comprehensive policy documents in Markdown
format. Use clear ## section headers, bullet points, and FAQ subsections.
Each document should be 800-1500 words and suitable for RAG chunking.
Use a professional but friendly tone. Include specific numbers (days,
percentages, dollar amounts) to make policies concrete.
"""

POLICY_SPECS = {
    "return_policy.md": {
        "title": "Return & Exchange Policy",
        "prompt": (
            "Write a comprehensive return and exchange policy covering:\n"
            "- Return window (30 days from delivery)\n"
            "- Conditions for returns (unused, original packaging, receipt)\n"
            "- Non-returnable items (opened software, personalized items, hygiene products)\n"
            "- Return process steps (initiate online, ship back, inspection, refund)\n"
            "- Exchange vs. return options\n"
            "- Restocking fees (15% for opened electronics after 15 days)\n"
            "- Defective product exceptions (extended to 90 days)\n"
            "- International return shipping costs\n"
            "- FAQ section with 5-7 common questions"
        ),
    },
    "shipping_policy.md": {
        "title": "Shipping & Delivery Policy",
        "prompt": (
            "Write a comprehensive shipping and delivery policy covering:\n"
            "- Shipping methods: Standard (5-7 days, free over $49), Express (2-3 days, $9.99), "
            "Next-Day ($14.99), Same-Day (select areas, $19.99)\n"
            "- Order processing time (1-2 business days)\n"
            "- Tracking information and notifications\n"
            "- Shipping restrictions (PO boxes, APO/FPO, hazardous materials)\n"
            "- International shipping (7-14 days, customs duties)\n"
            "- Delivery issues (missing, damaged, wrong address)\n"
            "- Holiday shipping cut-off dates\n"
            "- Large/heavy item shipping surcharges\n"
            "- FAQ section with 5-7 common questions"
        ),
    },
    "warranty_policy.md": {
        "title": "Warranty & Protection Policy",
        "prompt": (
            "Write a comprehensive warranty policy covering:\n"
            "- Standard manufacturer warranty (1 year for electronics, 2 years for appliances)\n"
            "- Extended protection plans (1, 2, or 3 additional years)\n"
            "- What is covered (defects, malfunctions, electrical failures)\n"
            "- What is NOT covered (accidental damage, water damage, cosmetic wear, unauthorized modifications)\n"
            "- Warranty claim process (file claim, send proof, diagnosis, repair or replace)\n"
            "- Turnaround time for warranty repairs (7-14 business days)\n"
            "- Transferability of warranty\n"
            "- Lemon policy (3 repairs in 12 months = full replacement)\n"
            "- FAQ section with 5-7 common questions"
        ),
    },
    "payment_policy.md": {
        "title": "Payment & Billing Policy",
        "prompt": (
            "Write a comprehensive payment and billing policy covering:\n"
            "- Accepted payment methods (Visa, MasterCard, Amex, Discover, PayPal, Apple Pay, Google Pay)\n"
            "- Payment security (PCI-DSS compliant, encrypted transactions)\n"
            "- Pre-authorization and charge timing\n"
            "- Refund processing times (3-5 days for card, 5-10 for PayPal, 1-2 for store credit)\n"
            "- Price matching (within 7 days, same retailer conditions)\n"
            "- Installment plans / Buy Now Pay Later options\n"
            "- Disputed charges and chargeback process\n"
            "- Billing errors and correction process\n"
            "- Gift cards and promotional credit\n"
            "- FAQ section with 5-7 common questions"
        ),
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# STATIC (HIGH-QUALITY HARDCODED) POLICY CONTENT
# ═══════════════════════════════════════════════════════════════════════════
STATIC_POLICIES = {
    "return_policy.md": """\
# Return & Exchange Policy

*Last updated: January 2025*

At ShopSmart Electronics, we want you to be completely satisfied with your purchase. If you're not happy with an item, we offer a straightforward return and exchange process.

## Return Window

- **Standard returns**: You may return most items within **30 days** of delivery for a full refund.
- **Defective products**: Items that arrive defective or develop a fault within normal use can be returned within **90 days** of delivery at no cost to you.
- **Holiday purchases**: Items purchased between November 15 and December 31 may be returned until January 31 of the following year.

## Conditions for Return

To be eligible for a return, items must meet ALL of the following conditions:

- The item must be in its **original, unused condition**
- All **original packaging, manuals, and accessories** must be included
- You must have your **order confirmation or receipt**
- The item must not show signs of **physical damage caused by the customer**

## Non-Returnable Items

The following items **cannot** be returned once opened or used:

- Software, digital downloads, and subscription cards
- Personalized or custom-configured items
- Hygiene-related products (earbuds, in-ear monitors) once the seal is broken
- Batteries and consumable parts
- Gift cards and promotional vouchers
- Clearance items marked as "Final Sale"

## Return Process

1. **Initiate your return** by logging into your account and navigating to Order History → select the order → click "Return Item"
2. **Select a reason** for the return and choose between refund or exchange
3. **Print the prepaid shipping label** (provided for defective items; return shipping is $5.99 for non-defective returns)
4. **Pack the item securely** in its original packaging and affix the shipping label
5. **Ship the package** via the designated carrier within 7 days of initiating the return
6. **Inspection & processing**: Once we receive the item, our team will inspect it within 3-5 business days
7. **Refund issued**: After approval, refunds are processed to your original payment method within 5-7 business days

## Exchange Options

- **Same item exchange**: We'll ship a replacement at no additional cost (subject to availability)
- **Different item exchange**: If the replacement costs more, you'll be charged the difference. If it costs less, the difference is refunded.
- **Exchanges for defective items** are processed with priority shipping at no extra charge

## Restocking Fees

- Returns made within **15 days** of delivery: **No restocking fee**
- Returns of opened electronics made between **16-30 days**: A **15% restocking fee** applies
- Defective items: **No restocking fee** regardless of timing
- Items returned in sealed, unopened condition: **No restocking fee** regardless of timing

## International Returns

- International customers are responsible for **return shipping costs**
- Items must be shipped with **tracking and insurance**
- Customs duties and taxes paid on the original order are **non-refundable**
- Processing time for international returns may take an additional 5-10 business days

## Frequently Asked Questions

**Q: How do I check the status of my return?**
A: Log into your account, go to Order History, and click on the returned order. The current status will be displayed (Received, Inspecting, Refund Processed).

**Q: Can I return an item purchased as a gift?**
A: Yes. Gift returns are issued as store credit to the gift recipient. The original purchaser will not be notified.

**Q: What if my item arrived damaged?**
A: Contact our support team within 48 hours of delivery with photos of the damage. We'll arrange a free return and send a replacement immediately.

**Q: Can I return a product I bought on sale?**
A: Yes, sale items follow the same return policy unless marked as "Final Sale."

**Q: How long until I get my refund?**
A: Once we approve the return, refunds take 5-7 business days for credit cards, 7-10 days for PayPal, and 1-2 days for store credit.

**Q: What if I lost the original packaging?**
A: We strongly recommend keeping original packaging for 30 days. Returns without original packaging may be subject to a higher restocking fee (up to 25%) or may be declined for non-defective items.

**Q: Can I return part of a bundle?**
A: Bundled items must be returned as a complete set. Individual items from a bundle cannot be returned separately.
""",
    "shipping_policy.md": """\
# Shipping & Delivery Policy

*Last updated: January 2025*

ShopSmart Electronics offers multiple shipping options to get your order to you quickly and affordably. All orders are processed from our warehouses located across the continental United States.

## Shipping Methods & Costs

| Method | Delivery Time | Cost | Free Threshold |
|---|---|---|---|
| **Standard Shipping** | 5-7 business days | $5.99 | Free on orders over $49 |
| **Express Shipping** | 2-3 business days | $9.99 | Free on orders over $149 |
| **Next-Day Shipping** | 1 business day | $14.99 | N/A |
| **Same-Day Delivery** | Same day (order by 12 PM) | $19.99 | Select metro areas only |

*Business days are Monday through Friday, excluding federal holidays.*

## Order Processing

- Orders placed **before 2:00 PM EST** on a business day begin processing the same day
- Orders placed **after 2:00 PM EST** or on weekends/holidays begin processing the next business day
- Processing typically takes **1-2 business days** before handoff to the carrier
- You'll receive an email with your **tracking number** once your order ships

## Tracking Your Order

- A tracking number is emailed within **24 hours** of shipment
- Track your package via the link in your email or in your account under Order History
- Real-time notifications are available via SMS (opt-in during checkout)
- If tracking shows no movement for 5+ business days, contact our support team

## Shipping Restrictions

- **PO Boxes**: Standard and Express shipping only. Next-Day and Same-Day are not available to PO Boxes.
- **APO/FPO/DPO**: We ship to military addresses via Standard shipping. Allow 10-14 business days.
- **Alaska & Hawaii**: Standard and Express only. Add 2-3 business days to estimated delivery.
- **Hazardous materials**: Items containing lithium batteries over 100Wh or other regulated materials may have carrier restrictions and cannot be shipped via air.

## International Shipping

- Available to **40+ countries** across North America, Europe, and Asia-Pacific
- Estimated delivery: **7-14 business days** depending on destination
- Shipping cost calculated at checkout based on weight, dimensions, and destination
- **Customs duties, taxes, and import fees** are the responsibility of the recipient and are not included in the order total
- Orders over $800 may require additional customs documentation

## Large & Heavy Items

- Items over **70 lbs** or with dimensions exceeding **108 inches** (length + girth) are subject to a **freight surcharge of $29.99-$79.99**
- Large appliances include **curbside delivery** by default; white-glove delivery (room of choice + unboxing) is available for $49.99
- Freight shipments are delivered Monday-Friday during business hours; a signature is required

## Delivery Issues

- **Missing package**: If tracking shows "delivered" but you haven't received it, check with neighbors and your building's mail room. If unfound after 24 hours, contact us for a claim.
- **Damaged in transit**: Photograph the damage and contact us within 48 hours. We'll send a replacement or issue a refund.
- **Wrong address**: If the package hasn't shipped yet, we can update the address. Once shipped, rerouting fees of $10-$15 may apply depending on the carrier.
- **Failed delivery attempts**: After 3 failed attempts, the package is returned to our warehouse. We'll notify you and offer re-shipment or a refund.

## Holiday Shipping Cut-offs (2025)

| Holiday | Standard | Express | Next-Day |
|---|---|---|---|
| Christmas | Dec 14 | Dec 19 | Dec 22 |
| Valentine's Day | Feb 7 | Feb 10 | Feb 12 |
| Mother's Day | May 2 | May 6 | May 8 |

## Frequently Asked Questions

**Q: Can I change my shipping method after placing an order?**
A: Yes, if the order hasn't been processed yet (within ~2 hours of placing the order). Contact support or edit the order in your account.

**Q: Do you offer free shipping?**
A: Yes! Standard shipping is free on orders over $49. Express shipping is free on orders over $149.

**Q: What carriers do you use?**
A: We ship via UPS, FedEx, and USPS depending on the shipping method and destination. Same-Day delivery uses local courier partners.

**Q: Can I pick up my order in-store?**
A: We are an online-only retailer and do not offer in-store pickup at this time.

**Q: What happens if my package is lost?**
A: Contact our support team with your order number. We'll file a carrier claim and either reship your order or issue a full refund within 5 business days.

**Q: Do you ship to P.O. Boxes?**
A: Yes, via Standard and Express shipping only. Next-Day and Same-Day options require a physical street address.
""",
    "warranty_policy.md": """\
# Warranty & Protection Policy

*Last updated: January 2025*

ShopSmart Electronics stands behind the quality of every product we sell. All items come with manufacturer warranty coverage, and we offer extended protection plans for additional peace of mind.

## Standard Manufacturer Warranty

| Product Category | Warranty Duration |
|---|---|
| Consumer electronics (headphones, speakers, etc.) | **1 year** |
| Computers & laptops | **1 year** |
| Monitors & displays | **1 year** |
| Home appliances (kitchen, laundry, etc.) | **2 years** |
| Smart home devices | **1 year** |
| Networking equipment (routers, switches) | **1 year** |
| Accessories (cables, cases, chargers) | **6 months** |

Warranty coverage begins on the **date of delivery**, not the purchase date.

## Extended Protection Plans

For additional coverage beyond the manufacturer warranty, we offer ShopSmart Shield protection plans:

| Plan Tier | Duration | Cost | Coverage |
|---|---|---|---|
| **Shield Basic** | +1 year | 8% of item price | Defects & malfunctions |
| **Shield Plus** | +2 years | 14% of item price | Defects + accidental damage |
| **Shield Premium** | +3 years | 19% of item price | Defects + accidental + power surge |

- Protection plans must be purchased **at the time of original order** or within **30 days** of delivery
- Plans are **non-transferable** and tied to the original purchaser
- Plans cover the **full replacement cost** — no deductibles

## What Is Covered

Standard warranty and protection plans cover:

- Manufacturing defects and material flaws
- Electrical and mechanical failures during normal use
- Component malfunctions (screens, buttons, ports, motors)
- Battery degradation below **80% of original capacity** within warranty period (Shield Plus and Premium only)
- Power surge damage (Shield Premium only)
- Accidental drops and spills (Shield Plus and Premium only)

## What Is NOT Covered

The following are **excluded** from all warranty and protection coverage:

- Intentional damage or misuse
- Cosmetic damage (scratches, dents) that does not affect functionality
- Water damage or submersion (unless the product is rated water-resistant)
- Damage from unauthorized modifications, repairs, or third-party accessories
- Software issues, viruses, or data loss
- Normal wear and tear (fading, discoloration, minor scuffs)
- Damage resulting from failure to follow the product manual
- Products used for commercial or industrial purposes (consumer warranty only)
- Theft or loss

## Warranty Claim Process

1. **File a claim** online through your account: Order History → select the item → "File Warranty Claim"
2. **Provide documentation**: Upload photos of the issue and a brief description of the malfunction
3. **Diagnosis**: Our technical team reviews your claim within **2-3 business days**
4. **Resolution**:
   - **Repair**: We'll send a prepaid shipping label. Average repair turnaround is **7-14 business days**
   - **Replace**: If repair is not feasible, a replacement unit is shipped (subject to availability)
   - **Refund**: If the product is discontinued and no suitable replacement exists, a full refund to store credit is issued
5. **Return your defective item** within **14 days** of receiving the replacement (prepaid label provided)

## Turnaround Time

| Action | Timeline |
|---|---|
| Claim review | 2-3 business days |
| Repair | 7-14 business days |
| Replacement shipment | 3-5 business days |
| Refund processing | 5-7 business days |

## Lemon Policy

If a product requires **3 or more repairs for the same issue within a 12-month period**, it qualifies under our Lemon Policy. You will receive a **full replacement with a new unit** at no charge, or a **full refund** if you prefer.

## Frequently Asked Questions

**Q: How do I check if my product is still under warranty?**
A: Log into your account and go to Order History. Each eligible item displays its warranty expiration date.

**Q: Can I transfer my warranty to someone else?**
A: Standard manufacturer warranties follow the product. Extended ShopSmart Shield plans are non-transferable and tied to the original purchaser's account.

**Q: Do I need to register my product to activate the warranty?**
A: No. Your warranty is automatically activated upon delivery. Your order confirmation serves as proof of purchase.

**Q: What if my product is out of warranty?**
A: We offer paid repair services for out-of-warranty products. Contact support for a diagnostic quote.

**Q: Can I purchase a protection plan after the 30-day window?**
A: Unfortunately, protection plans can only be added at purchase or within 30 days of delivery. After that, we recommend checking if the manufacturer offers extended coverage directly.

**Q: Will I get a new or refurbished replacement?**
A: Warranty replacements are fulfilled with **new units** when available. If the exact model is discontinued, we provide a comparable new model or a refurbished unit of equal or greater value (with your consent).
""",
    "payment_policy.md": """\
# Payment & Billing Policy

*Last updated: January 2025*

ShopSmart Electronics offers a variety of secure payment methods for your convenience. We are committed to protecting your financial information and ensuring a smooth transaction experience.

## Accepted Payment Methods

We accept the following payment methods:

- **Credit Cards**: Visa, MasterCard, American Express, Discover
- **Debit Cards**: Any card with a Visa or MasterCard logo
- **Digital Wallets**: PayPal, Apple Pay, Google Pay, Samsung Pay
- **Gift Cards**: ShopSmart gift cards and e-gift certificates
- **Store Credit**: From returns, promotional offers, or loyalty rewards
- **Buy Now, Pay Later**: Affirm (for orders $50+), Klarna, Afterpay

## Payment Security

Your payment security is our top priority:

- All transactions are protected with **256-bit SSL encryption**
- We are **PCI-DSS Level 1 compliant** — the highest level of payment card security
- We use **3D Secure authentication** (Verified by Visa, MasterCard SecureCode) for additional fraud protection
- We **never store** your full credit card number on our servers
- Suspicious transactions are flagged and may require additional verification

## Pre-Authorization & Charge Timing

- When you place an order, a **pre-authorization hold** is placed on your payment method for the order total
- Your card is **charged** only when the item ships (or is ready for delivery)
- If an item in your order is backordered, you are only charged when it ships
- Pre-authorization holds typically release within **3-5 business days** if an order is cancelled before processing

## Refund Processing Times

| Payment Method | Refund Timeline |
|---|---|
| Credit/Debit Card | **5-7 business days** after approval |
| PayPal | **3-5 business days** after approval |
| Apple Pay / Google Pay | **5-7 business days** after approval |
| Store Credit | **1-2 business days** (instant in most cases) |
| Gift Card | Refunded to the **same gift card** within 24 hours |
| Affirm/Klarna/Afterpay | Processed through the BNPL provider; **7-14 business days** |

*Note: Refund timelines begin after the return is approved, not when the item is shipped back.*

## Price Match Guarantee

We offer price matching under the following conditions:

- The identical product (same model, color, size, condition) must be available from an **authorized U.S. retailer**
- The competitor's price must be current and **publicly verifiable** (no membership-exclusive prices)
- Price match requests must be made within **7 days of your purchase**
- We match the **item price only** — shipping costs, taxes, and bundled discounts are excluded
- Price matching is **not available** on: marketplace sellers (Amazon third-party, eBay), auction sites, refurbished listings, or clearance/closeout items
- To request a price match, contact support with a link to the competitor's listing

## Buy Now, Pay Later (BNPL)

We partner with Affirm, Klarna, and Afterpay to offer flexible payment options:

- **Affirm**: Split into 3, 6, or 12 monthly payments. 0% APR available on select items. Available on orders **$50 and above**.
- **Klarna**: Pay in 4 interest-free installments over 6 weeks.
- **Afterpay**: Pay in 4 interest-free installments every 2 weeks. Available on orders **$35-$1,000**.

BNPL terms, eligibility, and approval are determined by the respective provider. Missed payments may incur late fees from the provider.

## Disputed Charges & Chargebacks

If you notice an unauthorized or incorrect charge:

1. **Contact us first** — most billing issues can be resolved directly with our support team within 24 hours
2. If we cannot resolve the issue, you may **file a dispute with your bank or card issuer**
3. Please note: filing a chargeback before contacting us may delay resolution and temporarily suspend your account
4. We respond to all chargeback inquiries within **5 business days**

## Billing Errors

If you believe you've been billed incorrectly:

- **Double charges**: Contact us with your order number; we'll verify and refund the duplicate within 24 hours
- **Incorrect amount**: We'll review the order total and adjust as needed
- **Charged for cancelled order**: Pre-authorization holds release automatically. If the charge posted, contact us for an expedited refund.

## Gift Cards & Promotional Credit

- **ShopSmart Gift Cards** are available in denominations of $25, $50, $100, and $200
- Gift cards **do not expire** and have **no maintenance fees**
- Gift cards cannot be redeemed for cash except where required by law
- **Promotional credits** (from referral programs, loyalty rewards, etc.) expire **12 months** after issuance unless otherwise stated
- Promotional credits cannot be combined with other promotions unless explicitly allowed

## Frequently Asked Questions

**Q: Why was my payment declined?**
A: Common reasons include insufficient funds, incorrect card details, expired card, or your bank's fraud protection. Try a different payment method or contact your bank.

**Q: Can I use multiple payment methods on one order?**
A: You can combine a gift card or store credit with one other payment method (credit card, PayPal, etc.). Two credit cards cannot be used on the same order.

**Q: When will I be charged for a pre-order?**
A: Pre-orders are charged when the item ships, not when the order is placed. A pre-authorization hold may appear temporarily.

**Q: Is it safe to save my card for future purchases?**
A: Yes. We use tokenization to store a secure reference to your card — your actual card number is never stored on our servers.

**Q: How do I update my billing information?**
A: Go to Account Settings → Payment Methods to add, edit, or remove saved payment methods. Changes apply to future orders only.

**Q: Can I get a receipt or invoice for my order?**
A: Yes. Order receipts are emailed automatically. You can also download invoices from Order History in your account at any time.

**Q: What happens to my BNPL plan if I return an item?**
A: The refund is applied to your BNPL plan. Your remaining installment payments will be adjusted accordingly by the provider.
""",
}


def generate_with_llm(provider: str) -> dict[str, str]:
    """Generate policy documents using an LLM provider.

    Args:
        provider: "gemini" or "openai"

    Returns:
        Dict mapping filename → content string.
    """
    results = {}

    for filename, spec in POLICY_SPECS.items():
        prompt = f"{COMPANY_CONTEXT}\n\nDocument: {spec['title']}\n\n{spec['prompt']}"

        if provider == "gemini":
            content = _call_gemini(prompt)
        elif provider == "openai":
            content = _call_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if content:
            results[filename] = content
            print(f"  ✅ Generated {filename} via {provider} ({len(content):,} chars)")
        else:
            print(f"  ⚠ LLM failed for {filename}, falling back to static content")
            results[filename] = STATIC_POLICIES[filename]

    return results


def _call_gemini(prompt: str) -> str | None:
    """Call Google Gemini API."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠ No GOOGLE_API_KEY / GEMINI_API_KEY found in environment")
        return None

    try:
        import google.generativeai as genai  # type: ignore[import-untyped]

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response: Any = model.generate_content(prompt)
        return cast(str, response.text)
    except ImportError:
        print("  ⚠ google-generativeai package not installed. pip install google-generativeai")
        return None
    except Exception as e:
        print(f"  ⚠ Gemini API error: {e}")
        return None


def _call_openai(prompt: str) -> str | None:
    """Call OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠ No OPENAI_API_KEY found in environment")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response: Any = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a technical writer creating e-commerce policy documents."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        # response.choices[0].message.content is untyped from the OpenAI package
        return cast(str, response.choices[0].message.content)
    except ImportError:
        print("  ⚠ openai package not installed. pip install openai")
        return None
    except Exception as e:
        print(f"  ⚠ OpenAI API error: {e}")
        return None


def save_policies(policies: dict[str, str]) -> list[Path]:
    """Save policy Markdown files to the knowledge base directory."""
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, content in policies.items():
        path = KNOWLEDGE_BASE_DIR / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)
        print(f"  ✅ Saved {path.name}")
    return paths


def run(provider: str = "static", force: bool = False) -> dict[str, str]:
    """Generate and save policy documents.

    Args:
        provider: "static", "gemini", or "openai"
        force: Regenerate even if files exist.
    """
    # Check if all policy files already exist
    all_exist = all(
        (KNOWLEDGE_BASE_DIR / fname).exists() for fname in POLICY_SPECS
    )
    if all_exist and not force:
        print(f"  [skip] All policy files already exist in {KNOWLEDGE_BASE_DIR.name}/")
        return {
            fname: (KNOWLEDGE_BASE_DIR / fname).read_text(encoding="utf-8")
            for fname in POLICY_SPECS
        }

    if provider == "static":
        print("  Using static (hardcoded) high-quality policy content")
        policies = STATIC_POLICIES.copy()
    else:
        print(f"  Generating policies via {provider} LLM …")
        policies = generate_with_llm(provider)

    save_policies(policies)
    return policies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate policy documents.")
    parser.add_argument(
        "--provider",
        choices=["static", "gemini", "openai"],
        default="static",
        help="Content generation method (default: static).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(provider=args.provider, force=args.force)
