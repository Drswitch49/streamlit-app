#!/usr/bin/env python3
"""
A State-of-the-Art Business Software Recommendation System
... [Docstring truncated for brevity] ...
"""

import argparse
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from fpdf import FPDF
import io

# --------------------------------------------------------------------------------
# Knowledge Base of Software Solutions
# --------------------------------------------------------------------------------
# [Existing code for software_solutions, automation_workflows, etc.]

software_solutions = [
    # [Existing dictionary entries for software_solutions...]
    {
        "name": "HubSpot",
        "industry": ["Retail", "SaaS", "Healthcare", "Finance", "Education", "E-commerce", "Nonprofit"],
        "features": ["CRM", "Integration", "Analytics", "Reporting", "Sales Automation"],
        "scalability": "High",
        "cost": 6000,
        "ease_of_integration": 5,
        "ratings": 4.8,
        "description": "A popular CRM platform known for robust marketing and sales automation."
    },
    # ... [Other software entries here]
    {
        "name": "Monday.com",
        "industry": ["SaaS", "Consultancy", "Nonprofit", "Retail", "E-commerce"],
        "features": ["Integration", "Analytics", "Collaboration", "Automation"],
        "scalability": "High",
        "cost": 4000,
        "ease_of_integration": 5,
        "ratings": 4.7,
        "description": "A versatile work management tool with strong automation and integrations."
    }
]

automation_workflows = {
    # [Existing workflow dictionary...]
    "E-commerce": [
        {
            "name": "Order Fulfillment and Tracking Workflow",
            "trigger": "New online order placed",
            "actions": [
                "Create packing slip and shipping label",
                "Notify warehouse for order fulfillment",
                "Update inventory counts automatically",
                "Send tracking link to customer"
            ]
        }
    ]
}

weights = {
    "alignment_with_goals": 0.3,
    "impact": 0.25,
    "scalability": 0.2,
    "cost_effectiveness": 0.15,
    "ease_of_implementation": 0.1
}

all_industries = sorted({i for s in software_solutions for i in s["industry"]})
all_feature_set = {f for s in software_solutions for f in s["features"]}
all_features = sorted(all_feature_set)

def score_software(business: Dict[str, Any], software: Dict[str, Any]) -> float:
    alignment_score = 1 if business["type"] in software["industry"] else 0
    if len(business["key_features"]) > 0:
        impact_score = len(set(business["key_features"]) & set(software["features"])) / len(business["key_features"])
    else:
        impact_score = 0
    
    scalability_score = 1 if software["scalability"] == business["scalability"] else 0.5
    cost_score = 1 if software["cost"] <= business["budget"] else max(0, 1 - ((software["cost"] - business["budget"])/business["budget"])*0.5)
    if cost_score < 0:
        cost_score = 0
    
    ease_score = software["ease_of_integration"] / 5

    total_score = (
        weights["alignment_with_goals"] * alignment_score +
        weights["impact"] * impact_score +
        weights["scalability"] * scalability_score +
        weights["cost_effectiveness"] * cost_score +
        weights["ease_of_implementation"] * ease_score
    )
    return total_score

def build_feature_vector(software: Dict[str, Any], features: List[str]) -> np.ndarray:
    vec = np.array([1 if f in software["features"] else 0 for f in features])
    return vec

def machine_learning_similarity_rank(business: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    business_vector = np.array([1 if f in business["key_features"] else 0 for f in all_features])
    
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    for c in candidates:
        sw = next(s for s in software_solutions if s["name"] == c["name"])
        candidate_vec = build_feature_vector(sw, all_features)
        c["ml_similarity"] = cosine_similarity(business_vector, candidate_vec)
    
    for c in candidates:
        c["final_score"] = 0.7 * c["score"] + 0.3 * c["ml_similarity"]
    
    candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    return candidates

def recommend_software(business: Dict[str, Any]) -> List[Dict[str, Any]]:
    scored_solutions = []
    for sw in software_solutions:
        s = score_software(business, sw)
        scored_solutions.append({
            "name": sw["name"],
            "score": s,
            "industry": sw["industry"],
            "features": sw["features"],
            "scalability": sw["scalability"],
            "cost": sw["cost"],
            "ratings": sw["ratings"],
            "description": sw["description"]
        })
    refined = machine_learning_similarity_rank(business, scored_solutions)
    return refined

def summarize_recommendations(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not recommendations:
        return {}
    best_overall = recommendations[0]
    best_value = max(recommendations, key=lambda r: (r["final_score"] / r["cost"]) if r["cost"] else r["final_score"])
    top_rated = max(recommendations, key=lambda r: (r["ratings"] * r["final_score"]))
    return {
        "best_overall": best_overall,
        "best_value": best_value,
        "top_rated": top_rated
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Business Software Recommendation System")
    parser.add_argument(
        "--mode",
        choices=["cli", "streamlit"],
        default="streamlit",
        help="Run mode: 'cli' for command line interface, 'streamlit' for web interface."
    )
    parser.add_argument(
        "--type",
        default=None,
        help="Business type (e.g., Retail, SaaS, Healthcare, etc.)."
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Key features needed if running in CLI mode."
    )
    parser.add_argument(
        "--scalability",
        default=None,
        choices=["Low", "Medium", "High"],
        help="Scalability requirement if running in CLI mode."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Budget if running in CLI mode."
    )
    return parser.parse_args()

def run_cli(business_type: str, key_features: List[str], scalability: str, budget: int) -> None:
    if not business_type or not key_features or not scalability or budget is None:
        print("Please provide all parameters: --type, --features, --scalability, --budget")
        sys.exit(1)

    business_requirements = {
        "type": business_type,
        "key_features": key_features,
        "scalability": scalability,
        "budget": budget
    }
    recommendations = recommend_software(business_requirements)
    if not recommendations:
        print("No suitable recommendations found.")
        return

    print("Recommended Software Solutions (Top to Bottom):")
    for rec in recommendations:
        print(f"{rec['name']}: Final Score - {rec['final_score']:.2f}, Cost: £{rec['cost']}, Features: {', '.join(rec['features'])}")

    summary = summarize_recommendations(recommendations)
    if summary:
        print("\nQuick Highlights:")
        print(f"- Best Overall: {summary['best_overall']['name']} (Score: {summary['best_overall']['final_score']:.2f})")
        print(f"- Best Value: {summary['best_value']['name']}")
        print(f"- Top Rated: {summary['top_rated']['name']}")

    if business_type in automation_workflows:
        print("\nSuggested Automation Workflows:")
        for workflow in automation_workflows[business_type]:
            print(f"Workflow: {workflow['name']}")
            print(f"Trigger: {workflow['trigger']}")
            for i, action in enumerate(workflow['actions'], start=1):
                print(f"Action {i}: {action}")
            print("---")

def rating_to_stars(rating: float) -> str:
    # Convert rating out of 5 to a star representation
    full_stars = int(rating)
    half_star = 1 if rating - full_stars >= 0.5 else 0
    stars = "★" * full_stars + ("½" if half_star else "")
    return stars

def generate_pdf_report(business: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "YOFY Recommendations Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    # Business requirements summary
    pdf.cell(0, 10, f"Business Type: {business['type']}", ln=True)
    pdf.cell(0, 10, f"Key Features: {', '.join(business['key_features'])}", ln=True)
    pdf.cell(0, 10, f"Scalability: {business['scalability']}", ln=True)
    pdf.cell(0, 10, f"Budget: £{business['budget']}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Recommended Software Solutions:", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    for rec in recommendations:
        pdf.cell(0, 8, f"Name: {rec['name']}", ln=True)
        pdf.cell(0, 8, f"Final Score: {rec['final_score']:.2f}", ln=True)
        pdf.cell(0, 8, f"Cost: £{rec['cost']}", ln=True)
        pdf.cell(0, 8, f"Features: {', '.join(rec['features'])}", ln=True)
        pdf.cell(0, 8, f"Industry Fit: {', '.join(rec['industry'])}", ln=True)
        pdf.cell(0, 8, f"Ratings: {rec['ratings']} / 5.0", ln=True)
        pdf.cell(0, 8, f"Description: {rec['description']}", ln=True)
        pdf.ln(8)

    # Output to a byte stream
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer, 'F')
    pdf_buffer.seek(0)
    return pdf_buffer.read()

def run_streamlit() -> None:
    st.title("Business Software Recommendation System")
    st.markdown("""
    Welcome to the next-generation "Rotten Tomatoes" style aggregator for business software!
    This system helps you discover top software solutions aligned with your business type, 
    key features, scalability, and budget.
    """)

    # User Input
    st.sidebar.header("Your Business Requirements")
    business_type = st.sidebar.selectbox("Your Business Type", all_industries)
    key_features = st.sidebar.multiselect("Key Features You Need", all_features)
    scalability = st.sidebar.selectbox("Scalability Needs", ["Low", "Medium", "High"])
    budget = st.sidebar.number_input("Your Yearly Budget (£)", min_value=1000, max_value=20000, step=1000)

    st.subheader("How It Works")
    st.write("""
    1. **Input Your Details:** Choose your business type, required features, scalability needs, and budget.
    2. **Our System Analyzes Options:** We match your needs against a vast, curated database of software solutions.
    3. **View Recommendations:** We'll highlight top choices, their ratings, costs, and features.
    4. **Explore Workflows:** Check industry-specific workflows to automate and streamline your operations.
    5. **Next Steps:** Guidance on implementation, trials, and vendor resources.
    """)

    if st.button("Find My Software Solutions"):
        if key_features:
            business_requirements = {
                "type": business_type,
                "key_features": key_features,
                "scalability": scalability,
                "budget": budget
            }
            recommendations = recommend_software(business_requirements)

            if recommendations:
                st.subheader("Your Top Recommendations")
                st.write("These software solutions best match your criteria:")

                # Display key highlights
                summary = summarize_recommendations(recommendations)
                if summary:
                    st.markdown("### Quick Highlights")
                    c1, c2, c3 = st.columns(3)
                    c1.metric(label="Best Overall", value=summary["best_overall"]["name"])
                    c2.metric(label="Best Value", value=summary["best_value"]["name"])
                    c3.metric(label="Top Rated", value=summary["top_rated"]["name"])

                    st.write(f"- **{summary['best_overall']['name']}**: Highest final score.")
                    st.write(f"- **{summary['best_value']['name']}**: Excellent score-to-cost ratio.")
                    st.write(f"- **{summary['top_rated']['name']}**: Exceptional rating and suitability.")

                # Prepare data for chart
                df = pd.DataFrame({
                    "Software": [r["name"] for r in recommendations],
                    "Final Score": [r["final_score"] for r in recommendations]
                })

                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Software:N', sort='-y'),
                    y='Final Score:Q',
                    tooltip=['Software', 'Final Score']
                ).properties(title="Recommended Software Rankings")

                st.altair_chart(chart, use_container_width=True)

                with st.expander("Detailed Recommendations"):
                    for rec in recommendations:
                        st.write(f"### {rec['name']}")
                        st.write(f"**Final Score:** {rec['final_score']:.2f}")
                        st.write(f"**Cost:** £{rec['cost']:,}")
                        st.write(f"**Features:** {', '.join(rec['features'])}")
                        st.write(f"**Industry Fit:** {', '.join(rec['industry'])}")
                        star_rating = rating_to_stars(rec['ratings'])
                        st.write(f"**Ratings:** {star_rating} ({rec['ratings']}/5.0)")
                        st.write(f"**Description:** {rec['description']}")
                        st.write("---")

                cheapest_high_score = min(recommendations, key=lambda r: r["cost"])
                st.info(f"For a cost-effective option, consider **{cheapest_high_score['name']}** at £{cheapest_high_score['cost']:,}.")

                # Suggested Automation Workflows
                if business_type in automation_workflows:
                    st.subheader("Suggested Automation Workflows")
                    st.write("Consider these workflows to streamline operations in your industry:")
                    for workflow in automation_workflows[business_type]:
                        st.write(f"**{workflow['name']}**")
                        st.write(f"**Trigger:** {workflow['trigger']}")
                        for i, action in enumerate(workflow['actions'], start=1):
                            st.write(f"- Action {i}: {action}")
                        st.write("---")

                # Next Steps
                st.subheader("Next Steps")
                st.write("""
                To get started:
                - Visit vendor websites and request demos or trial accounts.
                - Explore documentation, training resources, and support channels.
                - Plan integration with your existing systems.
                - Gradually onboard your team to ensure a smooth transition.
                
                Leverage these recommendations as a starting point to build a robust, scalable, 
                and efficient software ecosystem for your digital business!
                """)

                # Downloadable Report Button
                pdf_data = generate_pdf_report(business_requirements, recommendations)
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_data,
                    file_name="YOFY_Recommendations_Report.pdf",
                    mime="application/pdf"
                )

            else:
                st.info("No suitable recommendations found. Consider adjusting your requirements.")
        else:
            st.warning("Please select at least one key feature to get recommendations.")

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "cli":
        run_cli(args.type, args.features, args.scalability, args.budget)
    else:
        run_streamlit()
