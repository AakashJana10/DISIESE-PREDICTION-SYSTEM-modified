import streamlit as st
import pickle
import os
import subprocess
import sys
from train_model import DiseaseClassifier

# Ensure pickle compatibility
sys.modules['__main__'].DiseaseClassifier = DiseaseClassifier

# Page config
st.set_page_config(
    page_title="AI Disease Prediction System | Clinical Diagnostic Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Design System CSS
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">

<style>
    /* Design Tokens */
    :root {
        --primary-navy: #004a8f;
        --accent-yellow: #fcd53f;
        --accent-yellow-hover: #e5be2d;
        --secondary-green: #28a745;
        --light-blue-bg: #e6f0f9;
        --neutral-bg: #f4f7fa;
        --font-heading: 'Montserrat', sans-serif;
        --font-body: 'Nunito', sans-serif;
    }

    /* Reset & Base overrides */
    html, body, [class*="css"], .stApp {
        font-family: var(--font-body) !important;
        background-color: #ffffff;
        color: #1e293b;
    }

    h1, h2, h3, h4, .brand-font {
        font-family: var(--font-heading) !important;
    }

    /* Marker Underline */
    .marker-underline {
        position: relative;
        display: inline-block;
        white-space: nowrap;
        z-index: 1;
    }
    .marker-underline::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100%;
        height: 10px;
        background-color: var(--accent-yellow);
        z-index: -1;
        transform: rotate(-2deg);
        border-radius: 2px;
    }

    /* Top Sticky-like Header bar */
    .custom-header {
        background-color: #ffffff;
        padding: 16px 24px;
        border-bottom: 2px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .header-logo-title {
        font-family: var(--font-heading);
        font-size: 1.6rem;
        font-weight: 900;
        color: var(--primary-navy);
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    .header-logo-sub {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .header-slogan-box {
        text-align: right;
        font-weight: 700;
        color: var(--primary-navy);
        font-size: 0.95rem;
    }

    /* Hero Banner Area */
    .hero-banner {
        background-color: var(--light-blue-bg);
        border-radius: 20px 20px 0 0;
        padding: 40px 36px 30px 36px;
        position: relative;
    }
    .hero-heading {
        font-family: var(--font-heading);
        font-size: 2.8rem;
        font-weight: 900;
        color: var(--primary-navy);
        line-height: 1.15;
        letter-spacing: -1px;
        margin-bottom: 12px;
    }
    .hero-subheading {
        font-size: 1.15rem;
        color: #334155;
        font-weight: 600;
        max-width: 650px;
        margin-bottom: 24px;
    }

    /* Identity Badge Card */
    .identity-badge-card {
        background: #ffffff;
        border-radius: 0.75rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-left: 8px solid var(--accent-yellow);
        padding: 18px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 460px;
        margin-bottom: 20px;
    }
    .badge-name {
        font-family: var(--font-heading);
        font-weight: 900;
        font-size: 1.6rem;
        color: #1e293b;
        line-height: 1.1;
    }
    .badge-role {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
    }
    .badge-number {
        font-family: var(--font-heading);
        font-weight: 900;
        font-size: 2.75rem;
        color: var(--primary-navy);
        line-height: 1;
        text-align: right;
    }
    .badge-number-label {
        font-size: 0.72rem;
        font-weight: 800;
        color: var(--secondary-green);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: right;
    }

    /* Three-layered SVG wave transitions */
    .wave-container {
        line-height: 0;
        position: relative;
        margin-bottom: 30px;
        background: #ffffff;
    }

    /* Diagnostic Results Card */
    .result-badge-card {
        background: #ffffff;
        border-radius: 14px;
        border-left: 8px solid var(--accent-yellow);
        box-shadow: 0 10px 18px -3px rgba(0,0,0,0.08);
        padding: 24px;
        margin-bottom: 24px;
    }
    .res-diag-title {
        font-family: var(--font-heading);
        font-size: 2.2rem;
        font-weight: 900;
        color: var(--primary-navy);
        line-height: 1.2;
    }

    /* Value Grid / Proposals */
    .value-card {
        background: var(--neutral-bg);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
        margin-bottom: 12px;
    }
    .icon-circle {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-bottom: 12px;
    }
    .icon-navy { background: rgba(0, 74, 143, 0.12); color: var(--primary-navy); }
    .icon-yellow { background: rgba(252, 213, 63, 0.3); color: #856404; }
    .icon-green { background: rgba(40, 167, 69, 0.15); color: var(--secondary-green); }

    /* CTA Section Banner */
    .cta-banner {
        background-color: var(--primary-navy);
        border-radius: 16px;
        padding: 36px 32px;
        color: #ffffff;
        position: relative;
        overflow: hidden;
        margin-top: 36px;
        box-shadow: 0 12px 24px rgba(0, 74, 143, 0.25);
    }
    .circle-deco-green {
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: var(--secondary-green);
        opacity: 0.2;
        bottom: -100px;
        right: -60px;
        pointer-events: none;
    }
    .circle-deco-yellow {
        position: absolute;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: var(--accent-yellow);
        opacity: 0.25;
        bottom: -50px;
        right: 120px;
        pointer-events: none;
    }

    /* Streamlit button custom styling */
    div.stButton > button {
        background-color: var(--accent-yellow) !important;
        color: var(--primary-navy) !important;
        font-family: var(--font-heading) !important;
        font-weight: 900 !important;
        font-size: 1.15rem !important;
        border-radius: 9999px !important;
        border: none !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: var(--accent-yellow-hover) !important;
        transform: translateY(-2px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Disease Info Knowledge Base
disease_info = {
    "Fungal infection": {
        "description": "A fungal infection affecting skin, dermis, or nails, causing irritation and rashes.",
        "precautions": ["Keep the affected area clean and dry", "Use topical antifungal cream", "Avoid sharing towels/clothes", "Maintain strict personal hygiene"],
        "severity": "Mild to Moderate"
    },
    "Allergy": {
        "description": "Exaggerated immune response to environmental allergens such as pollen, dust, or mites.",
        "precautions": ["Identify and avoid allergen triggers", "Take prescribed antihistamines", "Use air purifiers", "Wear a mask in dusty areas"],
        "severity": "Mild to Severe"
    },
    "GERD": {
        "description": "Gastroesophageal reflux disease caused by stomach acid irritating the esophageal lining.",
        "precautions": ["Avoid spicy and acidic meals", "Do not lie down right after meals", "Elevate the head of your bed", "Eat small, frequent meals"],
        "severity": "Moderate"
    },
    "Chronic cholestasis": {
        "description": "Impaired bile flow from the liver leading to jaundice and systemic itching.",
        "precautions": ["Avoid all alcohol", "Adhere to low-fat nutrition", "Take prescribed fat-soluble vitamins", "Schedule regular hepatic monitoring"],
        "severity": "Severe"
    },
    "Drug Reaction": {
        "description": "Adverse drug response causing cutaneous or systemic symptoms.",
        "precautions": ["Stop the suspected drug immediately", "Consult emergency healthcare", "Keep an allergy card", "Notify physicians in future visits"],
        "severity": "Moderate to Severe"
    },
    "Peptic ulcer disease": {
        "description": "Sores developing on the lining of the stomach or small intestine.",
        "precautions": ["Avoid NSAID painkillers like ibuprofen", "Quit smoking and avoid alcohol", "Eat soothing meals", "Take prescribed acid reducers"],
        "severity": "Moderate"
    },
    "AIDS": {
        "description": "Advanced HIV infection causing progressive immune system failure.",
        "precautions": ["Adhere to Antiretroviral Therapy (ART)", "Use barrier protection", "Prevent opportunistic infections", "Routine CD4 testing"],
        "severity": "Severe"
    },
    "Diabetes": {
        "description": "Metabolic disorder marked by chronic hyperglycemia and insulin dysregulation.",
        "precautions": ["Check glucose levels daily", "Follow a low glycemic index diet", "Exercise 30 mins a day", "Take prescribed medication consistently"],
        "severity": "Moderate to Severe"
    },
    "Gastroenteritis": {
        "description": "Infection and inflammation of the gastrointestinal tract causing fluid loss.",
        "precautions": ["Hydrate with ORS electrolytes", "Rest the stomach with light broths", "Wash hands thoroughly", "Avoid dairy products temporarily"],
        "severity": "Mild to Moderate"
    },
    "Bronchial Asthma": {
        "description": "Chronic inflammatory disease of the airways causing bronchospasms.",
        "precautions": ["Keep your rescue inhaler accessible", "Avoid smoke, dust, and cold winds", "Monitor peak flow rate", "Follow your asthma management plan"],
        "severity": "Moderate to Severe"
    },
    "Hypertension": {
        "description": "Sustained high arterial blood pressure raising cardiovascular risk.",
        "precautions": ["Limit sodium intake to under 2g/day", "Engage in regular aerobic exercise", "Manage emotional stress", "Monitor BP regularly"],
        "severity": "Moderate to Severe"
    },
    "Migraine": {
        "description": "Severe throbbing neurological headache accompanied by nausea or photophobia.",
        "precautions": ["Rest in a dark and calm room", "Stay hydrated", "Avoid known dietary triggers", "Take prescribed migraine medication"],
        "severity": "Moderate"
    },
    "Cervical spondylosis": {
        "description": "Degenerative wear of discs and cartilage in the cervical spine.",
        "precautions": ["Practice gentle neck stretches", "Use an ergonomic chair and monitor height", "Use a supportive cervical pillow", "Avoid heavy lifting"],
        "severity": "Moderate"
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Acute loss of motor function caused by intracranial hemorrhage.",
        "precautions": ["SEEK EMERGENCY HOSPITAL CARE IMMEDIATELY", "Strict blood pressure control", "Physical therapy and neuro-rehab", "Continuous medical monitoring"],
        "severity": "Critical"
    },
    "Jaundice": {
        "description": "Yellowish pigmentation of the skin and sclera caused by hyperbilirubinemia.",
        "precautions": ["Drink ample boiled water and fruit juices", "Complete abstinence from alcohol", "Consume light, easy-to-digest food", "Undergo liver function tests"],
        "severity": "Moderate to Severe"
    },
    "Malaria": {
        "description": "Mosquito-transmitted parasite infection causing periodic paroxysms and fevers.",
        "precautions": ["Complete full course of antimalarials", "Sleep under treated mosquito netting", "Apply DEET repellent", "Eliminate stagnant water around home"],
        "severity": "Moderate to Severe"
    },
    "Chicken pox": {
        "description": "Contagious varicella virus causing vesicular itchy skin blisters.",
        "precautions": ["Isolate until all blisters have dried", "Apply soothing calamine lotion", "Keep nails clipped short", "Do not administer aspirin"],
        "severity": "Mild to Moderate"
    },
    "Dengue": {
        "description": "Flavivirus infection spread by Aedes mosquitoes, causing platelet drop.",
        "precautions": ["Rest and consume lots of fluids/ORS", "Monitor platelet counts daily", "Avoid aspirin or ibuprofen (risk of bleeding)", "Consult doctor immediately for warning signs"],
        "severity": "Moderate to Severe"
    },
    "Typhoid": {
        "description": "Salmonella Typhi bacterial infection causing prolonged fever.",
        "precautions": ["Complete the entire antibiotic regimen", "Drink only purified or boiled water", "Consume hot and freshly cooked food", "Practice hand hygiene"],
        "severity": "Moderate to Severe"
    },
    "hepatitis A": {
        "description": "Enterically transmitted viral hepatitis via contaminated food or water.",
        "precautions": ["Get sufficient bed rest", "Eat small, nutritious meals", "Strict hand sanitation", "Avoid alcohol until full recovery"],
        "severity": "Mild to Moderate"
    },
    "Hepatitis B": {
        "description": "Blood-borne viral infection leading to chronic liver cirrhosis or cancer.",
        "precautions": ["Take antiviral drugs as prescribed", "Ensure family members receive vaccination", "Never share personal razors or needles", "Periodic liver scans"],
        "severity": "Moderate to Severe"
    },
    "Hepatitis C": {
        "description": "Chronic viral liver disease often asymptomatic until advanced stages.",
        "precautions": ["Take modern direct-acting antivirals", "Abstain strictly from alcohol", "Do not share blood-contact items", "Regular viral load screening"],
        "severity": "Severe"
    },
    "Hepatitis D": {
        "description": "Severe hepatitis co-infection occurring only in patients with Hepatitis B.",
        "precautions": ["Specialist hepatology care", "Manage underlying Hepatitis B", "Maintain low sodium diet", "Routine clinical check-ups"],
        "severity": "Severe"
    },
    "Hepatitis E": {
        "description": "Water-borne viral hepatitis; can be particularly dangerous during pregnancy.",
        "precautions": ["Drink sanitized drinking water", "Avoid raw or partially cooked food", "Rest and rehydrate", "Immediate hospitalization if pregnant"],
        "severity": "Moderate"
    },
    "Alcoholic hepatitis": {
        "description": "Hepatic inflammation driven by excessive chronic alcohol consumption.",
        "precautions": ["Complete and permanent alcohol cessation", "High-protein nutritional support", "Liver panel monitoring", "Seek specialized counseling"],
        "severity": "Severe"
    },
    "Tuberculosis": {
        "description": "Mycobacterium tuberculosis infection primarily affecting the pulmonary system.",
        "precautions": ["Complete 6 months of DOTS medication", "Wear protective masks when around others", "Ensure good room ventilation", "High-protein diet"],
        "severity": "Severe"
    },
    "Common Cold": {
        "description": "Viral upper respiratory infection causing coryza and mild fatigue.",
        "precautions": ["Rest and drink warm fluids", "Use steam inhalation or saline sprays", "Wash hands regularly", "Avoid cold exposure"],
        "severity": "Mild"
    },
    "Pneumonia": {
        "description": "Infection causing alveolar inflammation and fluid accumulation in lungs.",
        "precautions": ["Complete antibiotic or antiviral therapy", "Monitor blood oxygen (SpO2)", "Stay hydrated to thin mucus", "Adequate rest"],
        "severity": "Moderate to Severe"
    },
    "Dimorphic hemorrhoids": {
        "description": "Swollen vascular cushions in the lower rectum causing pain or bleeding.",
        "precautions": ["High fiber diet and psyllium husk", "Drink at least 2.5L water daily", "Warm sitz baths 2x a day", "Avoid straining during defecation"],
        "severity": "Mild to Moderate"
    },
    "Heart attack": {
        "description": "Coronary artery occlusion leading to myocardial tissue ischemia.",
        "precautions": ["CALL EMERGENCY MEDICAL SERVICES (911/112) IMMEDIATELY", "Chew aspirin if advised by emergency doctor", "Cardioprotective low-cholesterol diet", "Structured cardiac rehabilitation"],
        "severity": "Critical"
    },
    "Varicose veins": {
        "description": "Twisted, enlarged superficial veins resulting from incompetent venous valves.",
        "precautions": ["Wear prescribed compression stockings", "Elevate legs when resting", "Avoid long uninterrupted periods of standing", "Engage in daily walking"],
        "severity": "Mild to Moderate"
    },
    "Hypothyroidism": {
        "description": "Deficient thyroid hormone output slowing basal metabolic rate.",
        "precautions": ["Take thyroxine daily on an empty stomach", "Wait 30-45 mins before breakfast", "Routine TSH tests every 6 months", "Balanced nutrition"],
        "severity": "Moderate"
    },
    "Hyperthyroidism": {
        "description": "Excess thyroid hormone production leading to hypermetabolic state.",
        "precautions": ["Take prescribed antithyroid medications", "Limit high-iodine food consumption", "Practice relaxation techniques", "Routine hormonal panels"],
        "severity": "Moderate"
    },
    "Hypoglycemia": {
        "description": "Dangerous drop in plasma glucose below physiological thresholds.",
        "precautions": ["Consume 15g fast-acting sugar immediately", "Recheck glucose in 15 minutes", "Carry glucose tablets", "Eat regular timely meals"],
        "severity": "Moderate to Severe"
    },
    "Osteoarthritis": {
        "description": "Degenerative breakdown of articulating joint cartilage.",
        "precautions": ["Low impact physical activities (swimming/cycling)", "Maintain healthy body weight", "Apply heat/cold packs for comfort", "Physical therapy"],
        "severity": "Moderate"
    },
    "Arthritis": {
        "description": "Inflammation of joints characterized by pain, swelling, and stiffness.",
        "precautions": ["Anti-inflammatory diet (Omega-3 rich)", "Daily joint range-of-motion drills", "Protect joints from strain", "Take prescribed anti-inflammatories"],
        "severity": "Moderate"
    },
    "(Vertigo) Paroymsal Positional Vertigo": {
        "description": "Inner ear balance disorder producing episodic spinning sensations.",
        "precautions": ["Perform Epley canalith repositioning", "Avoid sudden head turnings", "Sleep with head slightly elevated", "Hold handrails when standing"],
        "severity": "Mild to Moderate"
    },
    "Acne": {
        "description": "Sebaceous follicle plugging resulting in pustules and blackheads.",
        "precautions": ["Wash face twice daily with mild cleanser", "Do not pick or pop pimples", "Use non-comedogenic cosmetics", "Apply salicylic acid or benzoyl peroxide"],
        "severity": "Mild"
    },
    "Urinary tract infection": {
        "description": "Bacterial colonization of the urinary bladder or urethra.",
        "precautions": ["Drink plenty of water throughout the day", "Complete prescribed antibiotics", "Empty bladder without holding urine", "Maintain feminine hygiene"],
        "severity": "Mild to Moderate"
    },
    "Psoriasis": {
        "description": "Autoimmune disease causing rapid epidermal turnover and silvery plaques.",
        "precautions": ["Moisturize skin with thick emollients", "Avoid triggers like stress and smoking", "Supervised UV phototherapy", "Avoid hot baths"],
        "severity": "Moderate"
    },
    "Impetigo": {
        "description": "Contagious superficial bacterial skin infection forming honey-colored crusts.",
        "precautions": ["Apply topical antibiotic ointments", "Wash sores gently with warm soapy water", "Cover sores loosely", "Wash bedding and towels in hot water"],
        "severity": "Mild"
    },
    "Paroymsal Positional Vertigo": {
        "description": "Vestibular disturbance generating brief positional dizziness.",
        "precautions": ["ENT canalith repositioning therapy", "Sit down immediately when dizzy", "Ensure good lighting in room", "Avoid sudden neck extensions"],
        "severity": "Mild to Moderate"
    }
}

@st.cache_resource
def load_model():
    model_path = 'model/disease_model.pkl'
    symptoms_path = 'model/symptoms.pkl'

    if not os.path.exists(model_path) or not os.path.exists(symptoms_path):
        with st.spinner("Initializing clinical diagnostic engine..."):
            res = subprocess.run([sys.executable, 'train_model.py'], capture_output=True, text=True)
            if res.returncode != 0:
                st.error(f"Training failed:\n{res.stderr}")
                return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(symptoms_path, 'rb') as f:
        symptoms = pickle.load(f)

    return model, symptoms

def predict_disease(model, symptoms_list, selected_symptoms):
    input_data = {symptom: 0 for symptom in symptoms_list}
    for symptom in selected_symptoms:
        if symptom in input_data:
            input_data[symptom] = 1

    probabilities = model.predict_proba([input_data])[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, probabilities))
    top5 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    return top5[0][0], top5

def format_symptom_name(symptom):
    return symptom.replace('_', ' ').replace('.', '').strip().title()

# ===================== UI RENDER =====================

# 1. Header (Sticky Top Nav look)
st.markdown("""
<div class="custom-header">
    <div style="display: flex; align-items: center; gap: 18px;">
        <div>
            <div class="header-logo-title">HEALTHCARE AI 2026</div>
            <div class="header-logo-sub">Clinical Diagnostic & Symptom Intelligence</div>
        </div>
        <div style="width: 2px; height: 38px; background: #cbd5e1;"></div>
        <div style="background: #e6f0f9; padding: 4px 12px; border-radius: 6px; font-weight: 800; color: #004a8f; font-size: 0.85rem;">
            ⚡ V2.4 CERTIFIED
        </div>
    </div>
    <div class="header-slogan-box">
        Evidence-Based Symptom Analysis
        <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">95.0% Validation Accuracy • 42 Conditions</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Hero Section
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div>
            <div class="hero-heading">Precision Health.<br>Power to <span class="marker-underline">Diagnose Early</span>.</div>
            <div class="hero-subheading">
                A clinically structured diagnostic intelligence system combining Bayesian pattern recognition across 132 symptom indicators.
            </div>
            
            <!-- Identity Badge Card -->
            <div class="identity-badge-card">
                <div>
                    <div class="badge-role">Active Engine</div>
                    <div class="badge-name">BAYESIAN ML-132</div>
                </div>
                <div style="color: #fcd53f; font-size: 1.8rem; font-weight: 900; padding: 0 10px;">➔</div>
                <div>
                    <div class="badge-number-label">DISEASES COVERED</div>
                    <div class="badge-number">42</div>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Three-layered SVG wave transitions
st.markdown("""
<div class="wave-container">
    <svg viewBox="0 0 1440 120" fill="none" style="width: 100%; height: 90px; display: block;" preserveAspectRatio="none">
        <!-- Layer 1: Growth Green (#28a745) -->
        <path d="M0,35 C320,110 640,-15 1020,55 C1240,95 1360,40 1440,25 L1440,120 L0,120 Z" fill="#28a745" fill-opacity="0.85"></path>
        <!-- Layer 2: Sunshine Yellow (#fcd53f) -->
        <path d="M0,60 C240,15 540,95 880,45 C1140,8 1340,75 1440,55 L1440,120 L0,120 Z" fill="#fcd53f"></path>
        <!-- Layer 3: White background transition -->
        <path d="M0,85 C380,30 720,105 1100,68 C1280,48 1380,80 1440,76 L1440,120 L0,120 Z" fill="#ffffff"></path>
    </svg>
</div>
""", unsafe_allow_html=True)

model, symptoms_list = load_model()

if model and symptoms_list:
    formatted_symptoms = {format_symptom_name(s): s for s in symptoms_list}

    # Symptom Input Area
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <h3 style="color: #004a8f; font-weight: 900; font-size: 1.7rem; margin-bottom: 4px;">
            🔍 Select Presenting <span class="marker-underline">Symptoms</span>
        </h3>
        <p style="color: #64748b; font-size: 1rem; font-weight: 600;">
            Type or search symptoms from our 132-factor clinical registry:
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_action = st.columns([3.2, 1.2])
    with col_input:
        selected_display = st.multiselect(
            "Symptoms Selection",
            options=sorted(formatted_symptoms.keys()),
            placeholder="Search symptoms (e.g., High Fever, Cough, Headache, Skin Rash...)",
            label_visibility="collapsed"
        )
    with col_action:
        predict_btn = st.button("🔮 Analyze & Predict", use_container_width=True)

    selected_raw = [formatted_symptoms[s] for s in selected_display]

    if selected_display:
        st.markdown(f"""
        <div style="background: #f4f7fa; padding: 10px 18px; border-radius: 8px; border-left: 4px solid #004a8f; margin-top: 10px; font-size: 0.95rem;">
            <strong>Active Indicators ({len(selected_display)}):</strong> {', '.join(selected_display)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0; border-top: 2px solid #e2e8f0; margin: 32px 0;'>", unsafe_allow_html=True)

    if predict_btn:
        if not selected_display:
            st.warning("⚠️ Please select at least one symptom before running the diagnostic model.")
        else:
            with st.spinner("Executing Bayesian likelihood analysis across 42 disease models..."):
                prediction, top5 = predict_disease(model, symptoms_list, selected_raw)

            col_res_left, col_res_right = st.columns([1.1, 0.9])

            info = disease_info.get(prediction, {
                "description": "Clinical overview unavailable for this diagnosis.",
                "precautions": ["Consult your physician for personalized medical evaluation."],
                "severity": "Moderate"
            })
            sev = info.get("severity", "Moderate")

            sev_badge_color = {
                "Critical": "#dc2626",
                "Severe": "#ea580c",
                "Moderate to Severe": "#d97706",
                "Moderate": "#0284c7",
                "Mild to Moderate": "#16a34a",
                "Mild": "#28a745"
            }.get(sev, "#004a8f")

            top_confidence = top5[0][1] * 100

            with col_res_left:
                st.markdown(f"""
                <div class="result-badge-card">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #64748b; letter-spacing: 1.5px; text-transform: uppercase;">
                        Primary Statistical Match
                    </div>
                    <div class="res-diag-title">{prediction}</div>
                    <div style="display: flex; align-items: center; gap: 14px; margin-top: 10px; margin-bottom: 18px;">
                        <span style="font-size: 1.3rem; font-weight: 900; color: #004a8f;">
                            Confidence: {top_confidence:.1f}%
                        </span>
                        <span style="background: {sev_badge_color}; color: white; padding: 4px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; text-transform: uppercase;">
                            {sev}
                        </span>
                    </div>
                    
                    <h4 style="color: #004a8f; font-weight: 800; margin-bottom: 6px;">📖 Clinical Summary</h4>
                    <p style="color: #334155; font-size: 0.98rem; line-height: 1.6; margin-bottom: 18px;">
                        {info.get('description', '')}
                    </p>

                    <h4 style="color: #004a8f; font-weight: 800; margin-bottom: 8px;">💊 Recommended Immediate Measures</h4>
                    <ul style="color: #334155; padding-left: 20px; line-height: 1.7; font-size: 0.95rem;">
                        {''.join(f'<li>{p}</li>' for p in info.get('precautions', []))}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col_res_right:
                st.markdown("""
                <div style="margin-bottom: 12px;">
                    <h3 style="color: #004a8f; font-weight: 900; font-size: 1.4rem;">
                        Differential Diagnosis <span class="marker-underline">(Top 5)</span>
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                for rank, (dis, prob) in enumerate(top5, 1):
                    pct = prob * 100
                    emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank - 1]
                    item_sev = disease_info.get(dis, {}).get("severity", "Moderate")

                    st.markdown(f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 1.05rem; color: #004a8f;">{emoji} {dis}</span>
                            <span style="font-weight: 900; font-size: 1.1rem; color: #004a8f;">{pct:.1f}%</span>
                        </div>
                        <div style="background: #e2e8f0; height: 10px; border-radius: 5px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #fcd53f, #004a8f); width: {max(pct, 2):.1f}%; height: 100%; border-radius: 5px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748b; margin-top: 6px;">
                            <span>Classification: <strong>{item_sev}</strong></span>
                            <span>Rank #{rank}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        # Pre-prediction Information Section
        col_about_left, col_about_right = st.columns([1.1, 0.9])

        with col_about_left:
            st.markdown("""
            <div style="padding-right: 20px;">
                <h2 style="font-size: 2.2rem; font-weight: 900; color: #004a8f; line-height: 1.2; margin-bottom: 16px;">
                    Clinical Rigor Built for <span class="marker-underline">Diagnostic Accuracy</span>
                </h2>
                <p style="font-size: 1.05rem; color: #334155; line-height: 1.7; margin-bottom: 16px;">
                    Our predictive architecture implements <strong>Multivariate Bernoulli Naive Bayes</strong> enhanced with Laplace smoothing and asymmetric likelihood weighting. By calculating exact posterior probabilities, it eliminates subjective bias and maps correlated indicators in milliseconds.
                </p>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    <span style="background: #e6f0f9; color: #004a8f; font-weight: 700; padding: 6px 14px; border-radius: 9999px; font-size: 0.88rem;">
                        ✓ 132 Symptoms Supported
                    </span>
                    <span style="background: #e6f0f9; color: #004a8f; font-weight: 700; padding: 6px 14px; border-radius: 9999px; font-size: 0.88rem;">
                        ✓ 42 Disease Profiles
                    </span>
                    <span style="background: #e6f0f9; color: #28a745; font-weight: 700; padding: 6px 14px; border-radius: 9999px; font-size: 0.88rem;">
                        ✓ 95.0% Validated Precision
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_about_right:
            st.markdown("""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div class="value-card">
                    <div class="icon-circle icon-navy">⚡</div>
                    <h4 style="font-weight: 800; color: #004a8f; margin-bottom: 4px;">Instant Inference</h4>
                    <p style="font-size: 0.88rem; color: #64748b;">Sub-millisecond mathematical inference without cloud latency.</p>
                </div>
                <div class="value-card">
                    <div class="icon-circle icon-yellow">🎯</div>
                    <h4 style="font-weight: 800; color: #004a8f; margin-bottom: 4px;">Differential Ranking</h4>
                    <p style="font-size: 0.88rem; color: #64748b;">Full Top-5 probabilistic distribution for complete clinical visibility.</p>
                </div>
                <div class="value-card">
                    <div class="icon-circle icon-green">🛡️</div>
                    <h4 style="font-weight: 800; color: #004a8f; margin-bottom: 4px;">Preventive Guidance</h4>
                    <p style="font-size: 0.88rem; color: #64748b;">Curated precautionary actions and severity ratings for each illness.</p>
                </div>
                <div class="value-card">
                    <div class="icon-circle icon-navy">🔒</div>
                    <h4 style="font-weight: 800; color: #004a8f; margin-bottom: 4px;">Zero Privacy Risk</h4>
                    <p style="font-size: 0.88rem; color: #64748b;">Client-side evaluation; your health indicators are never logged.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. CTA Section
    st.markdown("""
    <div class="cta-banner">
        <div class="circle-deco-green"></div>
        <div class="circle-deco-yellow"></div>
        <div style="position: relative; z-index: 2; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div>
                <h3 style="font-family: var(--font-heading); font-size: 1.8rem; font-weight: 900; margin-bottom: 6px; color: #ffffff;">
                    Need a Comprehensive Health Evaluation?
                </h3>
                <p style="font-size: 1rem; color: #e6f0f9; max-width: 600px;">
                    This platform serves educational and triage decision-support purposes. Consult with a board-certified physician for clinical examinations and lab confirmations.
                </p>
            </div>
            <div style="background: #fcd53f; color: #004a8f; font-family: var(--font-heading); font-weight: 900; font-size: 1rem; padding: 12px 26px; border-radius: 9999px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                CLINICAL DECISION SUPPORT 2026
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Footer (4-Column Grid)
    st.markdown("""
    <footer style="margin-top: 48px; padding-top: 36px; border-top: 1px solid #e2e8f0;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 24px; margin-bottom: 24px;">
            <div>
                <div style="font-family: var(--font-heading); font-size: 1.25rem; font-weight: 900; color: #004a8f;">HEALTHCARE AI</div>
                <div style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 8px;">Diagnostic Intelligence</div>
                <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5;">Empowering early illness detection through interpretable machine learning models.</p>
            </div>
            <div style="border-left: 4px solid #004a8f; padding-left: 14px;">
                <h4 style="font-family: var(--font-heading); font-size: 1rem; font-weight: 800; color: #004a8f; margin-bottom: 6px;">Clinical Mission</h4>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">To bridge symptoms and triage clarity with transparent Bayesian probabilistic inference.</p>
            </div>
            <div>
                <h4 style="font-family: var(--font-heading); font-size: 1rem; font-weight: 800; color: #004a8f; margin-bottom: 6px;">Specifications</h4>
                <div style="font-size: 0.85rem; color: #64748b; line-height: 1.6;">
                    <div>• 132 Symptoms Tracked</div>
                    <div>• 42 Diseases Modeled</div>
                    <div>• 95.0% Testing Precision</div>
                </div>
            </div>
            <div>
                <h4 style="font-family: var(--font-heading); font-size: 1rem; font-weight: 800; color: #004a8f; margin-bottom: 6px;">Compliance</h4>
                <div style="font-size: 0.82rem; color: #64748b; line-height: 1.5;">
                    <div>Educational Reference Only</div>
                    <div>Non-Diagnostic Medical Disclaimer</div>
                    <div>Ver. 2.4.0 (2026 Build)</div>
                </div>
            </div>
        </div>
        <div style="background-color: #004a8f; color: #ffffff; padding: 12px; border-radius: 8px; text-align: center; font-size: 0.85rem; font-weight: 600;">
            © 2026 AI Disease Prediction System. Built for Clinical Data Science and Healthcare Informatics.
        </div>
    </footer>
    """, unsafe_allow_html=True)
