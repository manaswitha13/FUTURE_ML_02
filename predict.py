import pickle
import os
from preprocess import preprocess

# =========================
# LOAD MODELS SAFELY
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "model_cat.pkl"), "rb") as f:
    cat_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "model_pri.pkl"), "rb") as f:
    pri_model = pickle.load(f)

# =========================
# RULE-BASED CATEGORY FIX
# =========================
def rule_category(text):
    text = text.lower()

    if any(word in text for word in ["payment", "refund", "money", "charged", "deducted"]):
        return "Billing inquiry"
    elif any(word in text for word in ["error", "not working", "bug", "crash", "issue"]):
        return "Technical issue"
    elif any(word in text for word in ["cancel", "unsubscribe"]):
        return "Cancellation request"
    elif any(word in text for word in ["product", "details", "information"]):
        return "Product inquiry"
    else:
        return None

# =========================
# RULE-BASED PRIORITY
# =========================
def rule_priority(text):
    text = text.lower()

    if any(word in text for word in ["urgent", "immediately", "asap"]):
        return "High"
    elif any(word in text for word in ["not working", "failed", "error"]):
        return "High"
    elif any(word in text for word in ["delay", "slow"]):
        return "Medium"
    else:
        return "Low"

# =========================
# FINAL PREDICTION FUNCTION
# =========================
def predict_ticket(text):
    clean = preprocess(text)

    # ML predictions
    ml_category = cat_model.predict([clean])[0]
    ml_priority = pri_model.predict([clean])[0]

    # Rule-based predictions
    rule_cat = rule_category(text)
    rule_pri = rule_priority(text)

    # Final category (rule overrides ML if available)
    final_category = rule_cat if rule_cat else ml_category

    # Final priority (take highest importance)
    priority_order = ["Low", "Medium", "High", "Critical"]

    final_priority = max(
        [ml_priority, rule_pri],
        key=lambda x: priority_order.index(x) if x in priority_order else 0
    )

    return final_category, final_priority

# =========================
# TESTING
# =========================
if __name__ == "__main__":
    text = input("Enter ticket: ")
    category, priority = predict_ticket(text)

    print("\n🎯 Prediction Result:")
    print("Category :", category)
    print("Priority :", priority)