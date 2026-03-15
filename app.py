

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model and scaler directly
@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Fraud threshold
FRAUD_THRESHOLD = 0.3

st.set_page_config(
    page_title="Financial Fraud Detector",
    layout="centered"
)

# Custom CSS for fraud alert styling
st.markdown("""
<style>
.fraud-alert {
    background-color: #ff0000;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    animation: blink 1s linear infinite;
}

@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
}

.legitimate-alert {
    background-color: #00cc44;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.suspicious-alert {
    background-color: #ff8c00;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Page header
st.title("Financial Fraud Detector")
st.markdown("Powered by **Random Forest ML** + **Scikit-learn** + **Streamlit**")
st.markdown("Enter transaction details below to detect fraud in real time.")
st.divider()

# Step by step instructions
st.subheader("How to Use This App")

with st.expander("Click here to read instructions", expanded=True):
    st.markdown("""
    **Step 1 — Load a Sample Transaction or Enter Your Own**
    Click any sample button to load a known transaction into the form.
    Or click Enter My Own Data to clear the form and type your own values.
    
    **Step 2 — Review the Transaction Details**
    The form below shows all transaction features. The Amount field shows
    the transaction value in dollars. The V1 to V28 fields are security
    features of the transaction. You can change any value to test
    different scenarios.
    
    **Step 3 — Analyze the Transaction**
    Once you are happy with the transaction details scroll down and click
    the Analyze Transaction button. The AI model will process your
    transaction in milliseconds.
    
    **Step 4 — Read Your Results**
    A GREEN box means the transaction is LEGITIMATE and safe.
    An ORANGE box means the transaction is SUSPICIOUS and needs review.
    A RED flashing box means FRAUD has been detected —
    take immediate action! The probability score tells you how confident
    the model is in its prediction.
    """)

st.divider()

# Sample transactions
FRAUD_SAMPLE_1 = {
    "V1": -2.3122265423263, "V2": 1.95199201064158,
    "V3": -1.60985073229769, "V4": 3.9979055875468,
    "V5": -0.522187864667764, "V6": -1.42654531920595,
    "V7": -2.53738730624579, "V8": 1.39165724829804,
    "V9": -2.77008927719433, "V10": -2.77227214465915,
    "V11": 3.20203320709635, "V12": -2.89990738849473,
    "V13": -0.595221881324605, "V14": -4.28925378244217,
    "V15": 0.389724120274487, "V16": -1.14074717980657,
    "V17": -2.83005567450437, "V18": -0.0168224681808257,
    "V19": 0.416955705037907, "V20": 0.126910559061474,
    "V21": 0.517232370861764, "V22": -0.0350493686052974,
    "V23": -0.465211076182388, "V24": 0.320198199610325,
    "V25": 0.0445191674733077, "V26": 0.177839798284401,
    "V27": 0.261145002567677, "V28": -0.143275874698919,
    "Amount": 2500.00
}

FRAUD_SAMPLE_2 = {
    "V1": -2.3122265423263, "V2": 1.95199201064158,
    "V3": -1.60985073229769, "V4": 3.9979055875468,
    "V5": -0.522187864667764, "V6": -1.42654531920595,
    "V7": -2.53738730624579, "V8": 1.39165724829804,
    "V9": -2.77008927719433, "V10": -2.77227214465915,
    "V11": 3.20203320709635, "V12": -2.89990738849473,
    "V13": -0.595221881324605, "V14": -4.28925378244217,
    "V15": 0.389724120274487, "V16": -1.14074717980657,
    "V17": -2.83005567450437, "V18": -0.0168224681808257,
    "V19": 0.416955705037907, "V20": 0.126910559061474,
    "V21": 0.517232370861764, "V22": -0.0350493686052974,
    "V23": -0.465211076182388, "V24": 0.320198199610325,
    "V25": 0.0445191674733077, "V26": 0.177839798284401,
    "V27": 0.261145002567677, "V28": -0.143275874698919,
    "Amount": 9.99
}

LEGITIMATE_SAMPLE_1 = {
    "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443,
    "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316,
    "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478,
    "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899,
    "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215,
    "Amount": 149.62
}

LEGITIMATE_SAMPLE_2 = {
    "V1": 1.19185711131486, "V2": 0.26615071205963,
    "V3": 0.16648011335321, "V4": 0.448154078460911,
    "V5": 0.0600176492822243, "V6": -0.0823608088155687,
    "V7": -0.0788029833323113, "V8": 0.0851016549148104,
    "V9": -0.255425128109186, "V10": -0.166974414004614,
    "V11": 1.61272666105479, "V12": 1.06523531137287,
    "V13": 0.48909501589608, "V14": -0.143772296441519,
    "V15": 0.635558093258208, "V16": 0.463917041022171,
    "V17": -0.114804663102346, "V18": -0.183361270123994,
    "V19": -0.145783041325259, "V20": -0.0690831352230203,
    "V21": -0.225775248033138, "V22": -0.638671952771851,
    "V23": 0.101288021253234, "V24": -0.339846475529127,
    "V25": 0.167170404418143, "V26": 0.125894532368176,
    "V27": -0.00898309914322813, "V28": 0.0147241691924927,
    "Amount": 45.00
}

MEDIUM_RISK_SAMPLE = {
    "V1": -1.5, "V2": 0.8,
    "V3": -0.5, "V4": 2.1,
    "V5": -0.3, "V6": -0.8,
    "V7": -1.2, "V8": 0.6,
    "V9": -1.1, "V10": -1.2,
    "V11": 1.5, "V12": -1.3,
    "V13": -0.3, "V14": -2.1,
    "V15": 0.2, "V16": -0.5,
    "V17": -1.3, "V18": -0.01,
    "V19": 0.2, "V20": 0.06,
    "V21": 0.25, "V22": -0.02,
    "V23": -0.23, "V24": 0.16,
    "V25": 0.02, "V26": 0.09,
    "V27": 0.13, "V28": -0.07,
    "Amount": 890.00
}

EMPTY_SAMPLE = {f"V{i}": 0.0 for i in range(1, 29)}
EMPTY_SAMPLE["Amount"] = 0.0

# Initialize session state
if "transaction" not in st.session_state:
    st.session_state.transaction = LEGITIMATE_SAMPLE_1.copy()

# Sample transaction buttons
st.subheader("Quick Test — Select a Sample Transaction")
st.caption("Click any button to load a sample transaction or enter your own data.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Fraud Samples**")
    if st.button("High Value Fraud ($2,500)", type="primary"):
        st.session_state.transaction = FRAUD_SAMPLE_1.copy()
    if st.button("Small Amount Fraud ($9.99)", type="primary"):
        st.session_state.transaction = FRAUD_SAMPLE_2.copy()

with col2:
    st.markdown("**Legitimate Samples**")
    if st.button("Legitimate Shopping ($149.62)"):
        st.session_state.transaction = LEGITIMATE_SAMPLE_1.copy()
    if st.button("Legitimate Restaurant ($45.00)"):
        st.session_state.transaction = LEGITIMATE_SAMPLE_2.copy()

with col3:
    st.markdown("**Other**")
    if st.button("Medium Risk ($890.00)"):
        st.session_state.transaction = MEDIUM_RISK_SAMPLE.copy()
    if st.button("Enter My Own Data"):
        st.session_state.transaction = EMPTY_SAMPLE.copy()

st.caption("To enter your own data — click 'Enter My Own Data' then type your values in the form below.")
st.divider()

# Transaction input form
st.subheader("Transaction Details")

amount = st.number_input(
    "Transaction Amount ($)",
    value=float(st.session_state.transaction["Amount"]),
    min_value=0.0,
    step=0.01
)

st.markdown("**Transaction Features (V1 - V28)**")
st.caption("These are PCA transformed security features of the transaction.")

cols = st.columns(4)
v_values = {}

for i in range(1, 29):
    col_index = (i - 1) % 4
    with cols[col_index]:
        v_values[f"V{i}"] = st.number_input(
            f"V{i}",
            value=float(st.session_state.transaction[f"V{i}"]),
            format="%.6f",
            step=0.000001
        )

st.divider()

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True
    )

# Prediction results
if analyze_button:
    try:
        # Build input dataframe
        input_data = pd.DataFrame([{
            **v_values,
            'Amount': amount
        }])

        # Scale Amount
        input_data['Amount'] = scaler.transform(
            input_data[['Amount']]
        )

        # Get fraud probability
        fraud_probability = model.predict_proba(input_data)[0][1]

        # Apply threshold
        is_fraud = fraud_probability >= FRAUD_THRESHOLD

        # Confidence level
        if fraud_probability >= 0.8:
            confidence = "HIGH"
        elif fraud_probability >= 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        prediction = "FRAUD" if is_fraud else "LEGITIMATE"

        st.divider()
        st.subheader("Analysis Results")

        # Show color coded alert
        if prediction == "FRAUD":
            st.markdown(
                '<div class="fraud-alert">FRAUD DETECTED!</div>',
                unsafe_allow_html=True
            )
        elif fraud_probability >= 0.1:
            st.markdown(
                '<div class="suspicious-alert">SUSPICIOUS TRANSACTION — REVIEW REQUIRED</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="legitimate-alert">LEGITIMATE TRANSACTION</div>',
                unsafe_allow_html=True
            )

        st.markdown("")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Prediction", value=prediction)
        with col2:
            st.metric(
                label="Fraud Probability",
                value=f"{fraud_probability * 100:.1f}%"
            )
        with col3:
            st.metric(label="Confidence", value=confidence)

        st.info(f"Transaction flagged as {prediction}. Fraud probability: {fraud_probability * 100:.2f}%")
        st.markdown(f"**Transaction Amount:** ${amount:,.2f}")

        if prediction == "FRAUD":
            st.divider()
            st.subheader("Fraud Analysis")

            if fraud_probability >= 0.8:
                st.error("""
                **HIGH RISK TRANSACTION DETECTED**

                This transaction has been flagged as highly suspicious
                based on the following indicators:

                - Unusual transaction pattern detected
                - Transaction features deviate significantly from normal behavior
                - High probability of fraudulent activity

                **Recommended Actions:**
                - Block this transaction immediately
                - Contact the cardholder to verify
                - File a Suspicious Transaction Report (STR) with FINTRAC
                - Escalate to fraud investigation team
                """)
            elif fraud_probability >= 0.5:
                st.warning("""
                **MEDIUM RISK TRANSACTION DETECTED**

                This transaction shows suspicious patterns that require
                immediate attention:

                - Transaction features show moderate anomalies
                - Pattern is inconsistent with typical legitimate transactions
                - Further verification is strongly recommended

                **Recommended Actions:**
                - Place transaction on hold pending review
                - Request additional verification from cardholder
                - Monitor account for further suspicious activity
                - Document findings for compliance records
                """)
            else:
                st.warning("""
                **LOW RISK FRAUD INDICATOR**

                This transaction has been flagged but shows borderline
                indicators:

                - Some transaction features appear slightly unusual
                - May be a false positive but requires verification
                - Pattern warrants closer examination

                **Recommended Actions:**
                - Request soft verification from cardholder
                - Monitor account activity closely
                - Review recent transaction history
                """)

        elif fraud_probability >= 0.1:
            st.divider()
            st.subheader("Suspicious Transaction Analysis")
            st.warning("""
            **SUSPICIOUS TRANSACTION — PRECAUTIONARY REVIEW RECOMMENDED**

            This transaction shows some unusual patterns that warrant
            closer attention before final approval:

            - Transaction amount is higher than typical for this pattern
            - Some security features show minor anomalies
            - Does not meet threshold for fraud but requires monitoring

            **Precautionary Actions:**
            - Do not block — transaction may be legitimate
            - Send soft verification SMS to cardholder
            - Monitor this account for next 24 hours
            - Flag for manual review if similar transactions follow
            - Document in compliance log for audit purposes
            """)

        else:
            st.success("""
            **TRANSACTION CLEARED**

            This transaction appears to be legitimate based on:

            - Transaction pattern matches normal behavior
            - All security features within expected ranges
            - No suspicious indicators detected

            Transaction has been approved for processing.
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

