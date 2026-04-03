import streamlit as st
import json
from PyPDF2 import PdfReader

from parse_uploaded_cases import parse_uploaded_case
from xai import influential_factors_dynamic, counterfactual_analysis
from outcome_pred import predict_outcome

st.title("⚖️ Legal Decision Support System")

uploaded = st.file_uploader("Upload Case PDF", type=["pdf"])


def pdf_to_text(file):
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        text += p.extract_text()
    return text


if uploaded:

    # -------- STEP 1: Extract text --------
    with st.spinner("Extracting text..."):
        text = pdf_to_text(uploaded)

    # -------- STEP 2: Parse --------
    with st.spinner("Parsing case..."):
        structured = parse_uploaded_case(text)

    if not structured:
        st.error("Parsing failed")
        st.stop()

    st.success("Case parsed!")

    # -------- STEP 3: Save temp case --------
    TEMP_PATH = "../data/processed_cases/structured/temp_case.json"

    with open(TEMP_PATH, "w", encoding="utf8") as f:
        json.dump(structured, f, indent=2)

    # -------- STEP 4: Run prediction --------
    with st.spinner("Analyzing..."):
        pred, probs, results = predict_outcome("temp_case.json")

    # -------- OUTPUT --------

    st.header("📊 Prediction")
    st.write(pred)

    st.header("📈 Probabilities")
    st.json(probs)

    # -------- HYBRID RETRIEVAL --------
    st.header("📚 Hybrid Retrieved Cases")

    for case, decision, score in results:
        st.write(f"{case} → {round(score,3)}")

    # -------- DECISIONS --------
    st.header("⚖️ Retrieved Cases Used for Prediction")

    for case, decision, score in results:
        st.write(f"{case} → {decision} (score: {round(score,3)})")

    # -------- TOP INFLUENTIAL CASES --------
    st.header("🔥 Top Influential Cases")

    for case, decision, score in results:
        st.write(f"{case} → {decision} (influence: {round(score,3)})")

    # -------- FACTORS --------
    st.header("🧠 Most Influential Factors")

    # convert to (case, score) for XAI
    simple_results = [(c, s) for c, _, s in results]

    factors = influential_factors_dynamic(structured, simple_results)

    if not factors:
        st.write("No strong overlapping factors found")

    for f, s in factors:
        st.write(f"{f} → {s}")

    # -------- COUNTERFACTUAL --------
    st.header("🔄 Counterfactual Analysis")

    cf = counterfactual_analysis(structured, predict_outcome)

    if cf:
        removed, new_pred = cf
        st.write(f"If '{removed[:80]}...' was absent → New Prediction: {new_pred}")
    else:
        st.write("Not enough facts for counterfactual")
