import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# Load datasets
sym = pd.read_csv("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/symtoms_df.csv")
precautions_df = pd.read_csv("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/precautions_df.csv")
medications_df = pd.read_csv("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/medications.csv")
diet_df = pd.read_csv("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/diets.csv")
description_df = pd.read_csv("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/description.csv")
workout_df = pd.read_csv("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/workout_df.csv")

# Load model
model = joblib.load(open("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/random_forest_model.pkl", 'rb'))

# Symptom to index mapping
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
disease_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Title and Sidebar Navigation
st.set_page_config(layout="wide")
st.image("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/img1.png", width=1000)
st.title("🩺 A futuristic take on personalized care")

# Sidebar Navigation
st.sidebar.title("Navigation Pannel")
st.sidebar.image("/data1/SANCHAYANghosh01/Personalized_HealthCare_System/img.png")
page = st.sidebar.radio("Go to", ("Home", "About", "Contact", "Developer", "Blog"))
st.sidebar.subheader("**⚠️Precaution :** AI model can give wrong predictions sometimes, so please consult a doctor before taking any action based on the prediction.")

# Disease Prediction Function
def predict_disease(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for s in symptoms:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1
    prediction = model.predict([input_vector])[0]
    return prediction  # Return directly if model gives disease name


# Recommendation Function
def recommendation_func(disease):
    desc = " ".join(description_df[description_df['Disease'] == disease]['Description'].values)
    pre = precautions_df[precautions_df['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
    med = medications_df[medications_df['Disease'] == disease]['Medication'].values.tolist()
    diets = diet_df[diet_df['Disease'] == disease]['Diet'].values.tolist()
    workouts = workout_df[workout_df['disease'] == disease]['workout'].values.tolist()
    return desc, pre, med, diets, workouts

# Home Page
if page == "Home":
    st.subheader("Enter Your Symptoms")
    symptoms_list = list(symptoms_dict.keys())
    selected_symptoms = st.multiselect("**Select Symptoms (Minimum 2 symptoms needed for prediction)**", symptoms_list)

    if st.button("Predict Disease"):
        if len(selected_symptoms) < 2:
            st.warning("Please select at least two symptoms.")
        else:
            disease = predict_disease(selected_symptoms)
            desc, pre, med, diets, workouts = recommendation_func(disease)

            st.success(f"🦠 Predicted Disease: **{disease}**")
            st.markdown(f"### 📝 Description:\n{desc}")

            st.markdown("### ⚠️ Precautions:")
            st.write(pre)

            st.markdown("### 💊 Medications:")
            st.write(med)

            st.markdown("### 🍎 Diet Recommendations:")
            st.write(diets)

            st.markdown("### 🏃 Workout Recommendations:")
            st.write(workouts)

# Other Pages
elif page == "About":
    st.header("📌 About NEOCURA – Your Intelligent Health Companion")
    st.markdown("""
        **➤ What is NEOCURA?**
                
                NEOCURA is an AI-powered medical assistant designed to provide personalized healthcare insights, early disease predictions, and tailored wellness recommendations.

**➤ Mission**
                
                To revolutionize digital healthcare by offering intelligent, accessible, and secure health advice to individuals worldwide.

**➤ Key Features**
                
                A. Symptom-based disease prediction using machine learning
    B. Tailored recommendations for medications, diets, workouts, and precautions
    C. Multimodal interaction via voice, text, and visual tools
    D. Seamless user experience through a responsive Streamlit interface

**➤ Technology Behind NEOCURA**
                
                NEOCURA leverages advanced AI models (e.g., Random Forest, NLP, Reinforcement Learning) trained on curated healthcare datasets for accurate and explainable predictions.

**➤ Privacy First**
                
                NEOCURA ensures strict data privacy, never storing or sharing personal health information.

**➤ For Everyone**
                
                Designed to assist patients, caregivers, and healthcare providers alike—NEOCURA bridges the gap between professional healthcare and day-to-day health management.

**➤ Future Vision**
                
                NEOCURA aims to integrate real-time wearable health data, mental health support, and multilingual assistance to serve users more holistically.

""")
# ** ** is for bold text in Markdown
elif page == "Contact":
    st.header("📞 Contact")
    st.markdown("**For further inquiries follow the details below:**")
    st.markdown("email us on **sanchayan.ghosh2022@uem.edu.in**")
    st.markdown("call us on **+91-9378012990**")
    st.markdown("follow us on **[Linkedin](https://www.linkedin.com/in/sanchayan-ghosh-0b735024a/?originalSubdomain=in)**")
    st.markdown("follow us on **[Github](https://github.com/sanchayan7432)**")
    st.markdown("follow us on **[Youtube](https://www.youtube.com/@sanchayanghosh485)**")


elif page == "Developer":
    st.header("👨‍💻 Developer Page – Meet the Mind Behind NEOCURA")
    st.markdown("""
    **🔹 Developer Name and Designation**                           
                Sanchayan Ghosh | 
    AI Researcher Intern | Prompt Engineer | Machine Learning Engineer 
    
    **🔹 Vision Behind NEOCURA**        
                “To create an intelligent, privacy-first healthcare assistant that empowers users with accurate predictions, holistic recommendations, and seamless accessibility—making quality healthcare guidance universally available.”
                
    **🔹 Key Contributions**     
    ✅ Designed and developed NEOCURA using advanced AI/ML models (Random Forest, Reinforcement Learning, NLP).
                
    ✅ Integrated voice recognition, TTS, symptom parsing, and disease prediction using custom datas
                
    ✅ Built a secure, modular, and scalable architecture with privacy-preserving mechanisms.
                
    ✅ Streamlined data pipelines across symptoms, medications, diets, and fitness modules.
                
    ✅ Developed the front-end using Streamlit for smooth and interactive UI/UX.
                
    **🔹 Acknowledgment**     
    NEOCURA is the result of months of research, experimentation, and a strong desire to bridge the gap between technology and healthcare.
                """)

elif page == "Blog":
    st.header("📝 Current Blogs")

    st.subheader("-------------------------------------------------------------------------")
    ############## blogs----------------------------------------------------------------
    # Blog 1
    st.subheader("Blog 1: Can fiber help you lose weight? Dietitian answers 5 key questions")
    st.markdown("""
                When it comes to weight loss, many people track their macros — i.e., their proteins, fats, and carbohydrates. However, as a humble nutrient, fiber is often overlooked. But what if this is the missing ingredient to weight loss success? Can fiber supplements replace whole foods? In this podcast, a nutritionist answers readers’ questions about fiber and more.
                Nowadays, the internet is awash with articles, charts, and recipes centered around eating more protein — anything from a 30-gram-protein breakfast to high-protein drinks and more — to naturally lose weight. And although the key to achieving good weight loss results is indeed a higher protein intake, there is a nutrient that is often overlooked: fiber.
    Fiber is crucial not only for digestive functioning but overall health. Studies have shown it can lower LDL cholesterol, reduce blood pressure, and protect against heart disease. Newer research also shows that fiber may promote weight loss and enhance sensitivity to insulin.
    However, statistics show that less than 5% of Americans realistically meet their recommended daily fiber intake, which is on average up to 34 grams (g)Trusted Source for adult men and about 28 g for adult women. So, how can we eat more fiber?
    In this episode of In Conversation, we’ll be tackling burning questions such as: What is fiber, and why is it important for our bodies? How can we tell whether we are eating enough fiber? Is it right to call fiber nature’s Ozempic?
    We’ll differentiate between soluble and insoluble fiber while discussing the ideal daily intake for different people. We’ll also touch on how fiber supplements like psyllium husk compare with whole foods, weighing their benefits for our well-being. We will also look at how fiber plays a crucial role in fighting insulin resistance and its potential role in supporting weight management goals.
    To discuss this and more, we’re joined by registered dietitian Lisa Valente, MS, RD. Lisa holds a Master of Science in nutrition communications from the Friedman School of Nutrition Science and Policy at Tufts University, and she completed her dietetic internship at Massachusetts General Hospital.
    After the podcast recording, for the readers of MNT, we also asked Lisa what her top favorite high fiber foods were.
    “I will say frozen berries, fresh too, but frozen berries tend to be a little bit higher in fiber for a fruit. And when you buy them frozen, they’re just more affordable and easier to have on hand. [T]hey don’t go bad on you in a day — so, you can add those to smoothies or mix them into oatmeal or yogurt,” she said.
    Lisa said her second choice would be whole wheat pasta and shared a fun fact about fiber.
    “Brown rice only has one gram more of fiber than white rice, but whole wheat pasta has significantly more than white pasta. [It creates] like this nice fiber-rich carbohydrate base to your dinner to add some vegetables or protein to,” she said.
    Her third choice was chia seeds.
    “I sprinkle them on oatmeal, but you can also make chia seed pudding where you soak them with some milk or non-dairy milk and add a little bit of fruit. They’re very filling and they not only have fiber, but they also have omega-3 and a little bit of protein. So I feel like they’re sort of this tiny but mighty little seed where you can check off a lot of nutrition boxes at once. So if I had to pick three, that’s a good starting place for me,” she told us.
                """)

    ##############----------------------------------------------------------------------
    st.subheader("-------------------------------------------------------------------------")
    st.markdown("**Coming soon with exciting updates on healthcare and AI!**")

