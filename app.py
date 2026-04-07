import streamlit as st
import sys
import os

# Fix import path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import predict_ticket

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("🎫 Support Ticket Classification System")
st.markdown("Automatically classify support tickets and assign priority")

# =========================
# INPUT
# =========================
user_input = st.text_area(
    "Enter your support issue:",
    placeholder="Example: My payment failed and money was deducted..."
)

# =========================
# BUTTON
# =========================
if st.button("Analyze Ticket"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a ticket description")
    else:
        category, priority = predict_ticket(user_input)

        # =========================
        # OUTPUT
        # =========================
        st.success(f"📌 Category: {category}")

        if priority in ["High", "Critical"]:
            st.error(f"🚨 Priority: {priority}")
        elif priority == "Medium":
            st.warning(f"⚠️ Priority: {priority}")
        else:
            st.info(f"ℹ️ Priority: {priority}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built using Machine Learning & NLP")