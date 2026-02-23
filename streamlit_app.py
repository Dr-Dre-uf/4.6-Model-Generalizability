import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.datasets import make_classification

st.set_page_config(page_title="Model Generalizability Sandbox", layout="wide")

# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_clinical_data():
    patient_cols = ['patientunitstayid', 'hospitalid', 'gender', 'age', 'ethnicity', 'admissionheight', 
                    'admissionweight', 'dischargeweight', 'hospitaladmitsource', 'hospitaldischargelocation', 
                    'hospitaldischargestatus', 'unittype', 'uniquepid', 'unitvisitnumber',
                    'patienthealthsystemstayid', 'hospitaldischargeyear']
    
    df = pd.read_csv('https://www.dropbox.com/scl/fi/qld4pvo6vlptm41av3y2e/patient.csv?rlkey=gry21fvb3u3dytz7i5jcujcu9&dl=1',
                     usecols=patient_cols)
    
    df['age'] = df['age'].replace({'> 89': 90})
    df = df.sort_values(by=['uniquepid', 'patienthealthsystemstayid', 'unitvisitnumber'])
    df = df.groupby('patienthealthsystemstayid').first().reset_index()
    
    hospital = pd.read_csv('https://www.dropbox.com/scl/fi/5sdjsbrjxk0hlbmpb4csi/hospital.csv?rlkey=rmlrhg3m9sm3hj2s6rykrg5w2&st=329j4gwk&dl=1')
    hospital = hospital.replace({'teachingstatus': {'f': 0, 't': 1}})
    df = df.merge(hospital, on='hospitalid', how='left')

    labnames = ['BUN', 'creatinine', 'sodium', 'Hct', 'wbc', 'glucose', 'potassium', 'Hgb', 'chloride', 'platelets',
                'RBC', 'calcium', 'MCV', 'MCHC', 'bicarbonate', 'MCH', 'RDW', 'albumin']
    labs = pd.read_csv('https://www.dropbox.com/scl/fi/qaxtx330hicc5u61siehn/lab.csv?rlkey=xs9oxpl5istkbuh5s80oyxwwi&st=ydfrjxkh&dl=1')
    labs['labname'] = labs['labname'].replace({'WBC x 1000': 'wbc', 'platelets x 1000': 'platelets'})
    labs = labs[labs['labname'].isin(labnames)]
    labs = labs.pivot_table(columns=['labname'], values=['labresult'], aggfunc='mean', index='patientunitstayid').reset_index()
    labs.columns = ['patientunitstayid'] + ['lab_' + c.lower() for c in labnames]

    df = df.merge(labs, on='patientunitstayid', how='left')
    df['in_hospital_mortality'] = df['hospitaldischargestatus'].map(lambda status: {'Alive': 0, 'Expired': 1}.get(status, np.nan))
    df = df.dropna(subset=['in_hospital_mortality'])
    
    num_feats = ['age', 'admissionweight'] + ['lab_' + c.lower() for c in labnames]
    for col in num_feats: df[col] = pd.to_numeric(df[col], errors='coerce')
    df[num_feats] = df[num_feats].fillna(df[num_feats].mean())

    maj = df[df['in_hospital_mortality'] == 0]
    min_ = df[df['in_hospital_mortality'] == 1]
    min_over = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    balanced = pd.concat([maj, min_over]).sample(n=300, random_state=42).reset_index(drop=True)
    
    return balanced, num_feats, 'in_hospital_mortality'

@st.cache_data
def load_foundational_data():
    X, y = make_classification(n_samples=300, n_features=6, n_informative=3, n_redundant=1, 
                               n_clusters_per_class=1, flip_y=0.1, random_state=42)
    features = ['gene_x_expression', 'protein_y_level', 'culture_ph', 'temperature_c', 'atp_concentration', 'ros_levels']
    df = pd.DataFrame(X, columns=features)
    
    df['gene_x_expression'] = (df['gene_x_expression'] * 10) + 50
    df['protein_y_level'] = (df['protein_y_level'] * 5) + 20
    df['culture_ph'] = (df['culture_ph'] * 0.5) + 7.4
    df['temperature_c'] = (df['temperature_c'] * 1.5) + 37.0
    df['atp_concentration'] = (df['atp_concentration'] * 2) + 10.0
    df['ros_levels'] = (df['ros_levels'] * 5) + 15.0
    
    df['cellular_apoptosis'] = y
    return df, features, 'cellular_apoptosis'

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Module Settings")
scientific_context = st.sidebar.radio(
    "Select Learning Context:", 
    ["Clinical (eICU)", "Foundational Science"],
    help="Toggle the terminology and dataset to match your specific field of study."
)

st.sidebar.markdown("---")

st.sidebar.title("Learning Activities")
mode = st.sidebar.radio(
    "Select an Activity:",
    [
        "Activity 1: Feature Engineering & Scaling", 
        "Activity 2: The Bias-Variance Tradeoff", 
        "Activity 3: Validation Strategies",
        "Activity 4: The 'What-If' Simulator"
    ],
    help="Navigate through the interactive activities to explore model generalizability."
)

# Load context data
if scientific_context == "Clinical (eICU)":
    df, all_features, target = load_clinical_data()
    context_desc = "Predicting in-hospital mortality using the eICU Database."
    default_feats = ['age', 'lab_glucose', 'lab_creatinine', 'lab_potassium']
    target_pos, target_neg = "Expired", "Alive"
else:
    df, all_features, target = load_foundational_data()
    context_desc = "Predicting cellular apoptosis using a simulated foundational science dataset."
    default_feats = ['gene_x_expression', 'protein_y_level', 'culture_ph', 'temperature_c']
    target_pos, target_neg = "Apoptosis (Death)", "Survival"

# Fix: Reset session state features if the context changes
if 'current_context' not in st.session_state or st.session_state.current_context != scientific_context:
    st.session_state.current_context = scientific_context
    st.session_state.selected_features = default_feats

# ==========================================
# ACTIVITY 1: FEATURE ENGINEERING
# ==========================================
if mode == "Activity 1: Feature Engineering & Scaling":
    st.title("Activity 1: Feature Engineering & Scaling")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        1. Select features from the dropdown to build your dataset.
        2. Toggle the Standard Scaler on and off. 
        3. Observe how unscaled data completely ruins the accuracy of distance-based models like KNN.
        """)
        
    st.session_state.selected_features = st.multiselect(
        "Select Features for your Model:",
        options=all_features,
        default=st.session_state.selected_features,
        help="Add or remove features. Notice how adding too many irrelevant features can degrade KNN performance (the 'Curse of Dimensionality')."
    )
    
    if len(st.session_state.selected_features) > 0:
        X = df[st.session_state.selected_features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        apply_scaling = st.checkbox("Apply Standard Scaler", value=True, help="Standardizes features by removing the mean and scaling to unit variance.")
        
        if apply_scaling:
            scaler = StandardScaler()
            X_train_final = scaler.fit_transform(X_train)
            X_test_final = scaler.transform(X_test)
        else:
            X_train_final = X_train
            X_test_final = X_test
            
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_final, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_final))
        
        col1, col2 = st.columns(2)
        col1.metric("Model Accuracy (Test Set)", f"{acc:.2%}")
        col2.info("KNN calculates the physical distance between points. If Glucose ranges from 80-200, and Potassium ranges from 3-5, the model will practically ignore Potassium unless the data is scaled.")
    else:
        st.warning("Please select at least one feature.")

# ==========================================
# ACTIVITY 2: BIAS-VARIANCE TRADEOFF
# ==========================================
elif mode == "Activity 2: The Bias-Variance Tradeoff":
    st.title("Activity 2: The Bias-Variance Tradeoff")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        1. Adjust the 'maximum k' slider. Observe where the Train and Test lines pull apart (overfitting).
        2. In the visualizer below, compare a 'Complex' model (Low k) to a 'Simple' model (High k) to see how decision boundaries physically change.
        """)
    
    X = df[st.session_state.selected_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    st.markdown("#### Accuracy Curve")
    k_max = st.slider("Select maximum k to test", 5, 50, 20)
    
    ks = range(1, k_max + 1)
    train_acc, test_acc = [], []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
        train_acc.append(accuracy_score(y_train, knn.predict(X_train_s)))
        test_acc.append(accuracy_score(y_test, knn.predict(X_test_s)))

    fig_acc, ax_acc = plt.subplots(figsize=(8, 3))
    ax_acc.plot(ks, train_acc, label='Train Accuracy', marker='o', color='#1f77b4')
    ax_acc.plot(ks, test_acc, label='Test Accuracy', marker='x', color='#ff7f0e')
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Number of Neighbors (k)")
    ax_acc.legend()
    st.pyplot(fig_acc)

    st.markdown("---")
    st.markdown("#### Visualizing Decision Boundaries (Top 2 Features)")
    
    if len(st.session_state.selected_features) >= 2:
        c1, c2 = st.columns(2)
        k_low = c1.number_input("Complex Model (k)", 1, 5, 1)
        k_high = c2.number_input("Simple Model (k)", 10, 50, 15)

        def draw_boundary(k_val, ax, title):
            X_2d = X_train_s[:, :2] 
            knn = KNeighborsClassifier(n_neighbors=k_val).fit(X_2d, y_train)
            x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
            y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, edgecolors='k', cmap='RdBu', s=30)
            ax.set_title(title)
            ax.set_xlabel(st.session_state.selected_features[0])
            ax.set_ylabel(st.session_state.selected_features[1])

        fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        draw_boundary(k_low, ax1, f"Overfitting (k={k_low})")
        draw_boundary(k_high, ax2, f"Underfitting (k={k_high})")
        st.pyplot(fig_b)
    else:
        st.warning("Please select at least two features in Activity 1 to view boundaries.")

# ==========================================
# ACTIVITY 3: VALIDATION STRATEGIES
# ==========================================
elif mode == "Activity 3: Validation Strategies":
    st.title("Activity 3: Model Stability and Validation")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        1. Adjust the 'Number of Folds' slider to see how stable the model performance is.
        2. Observe the variance in the boxplot.
        3. Compare K-Fold, Stratified K-Fold, and LOO-CV mean accuracies.
        """)
        
    X = df[st.session_state.selected_features]
    y = df[target]
        
    col_cv1, col_cv2 = st.columns(2)
    n_f = col_cv1.slider("Number of Folds", 2, 10, 5)
    k_cv = col_cv2.slider("Model Complexity (k)", 1, 20, 5)

    knn_cv = KNeighborsClassifier(n_neighbors=k_cv)
    kf = KFold(n_splits=n_f, shuffle=True, random_state=42)
    X_s = StandardScaler().fit_transform(X) 
    scores = cross_val_score(knn_cv, X_s, y, cv=kf)

    fig_cv, ax_cv = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=scores, color="#4c72b0", ax=ax_cv)
    sns.stripplot(x=scores, color="black", size=8, jitter=True, ax=ax_cv)
    ax_cv.set_title(f"Accuracy Distribution ({n_f} Folds)")
    st.pyplot(fig_cv)

    st.markdown("#### Comparing Validation Strategies")
    skf = StratifiedKFold(n_splits=n_f, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    
    with st.spinner("Calculating metrics..."):
        skf_s = cross_val_score(knn_cv, X_s, y, cv=skf)
        loo_s = cross_val_score(knn_cv, X_s, y, cv=loo)

    methods = ["K-Fold", "Stratified K-Fold", "LOO-CV"]
    means = [scores.mean(), skf_s.mean(), loo_s.mean()]
    sterr = [scores.std(), skf_s.std(), loo_s.std()]

    fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
    ax_bar.bar(methods, means, yerr=sterr, capsize=10, color=['#1f77b4', '#2ca02c', '#d62728'], alpha=0.8)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_ylabel("Mean Accuracy")
    st.pyplot(fig_bar)

# ==========================================
# ACTIVITY 4: THE SIMULATOR
# ==========================================
elif mode == "Activity 4: The 'What-If' Simulator":
    st.title("Activity 4: The 'What-If' Simulator")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        1. Select a k value to determine how your model evaluates neighbors.
        2. Adjust the sliders for the first 3 features to build a theoretical profile.
        3. Watch the model classify the profile in real-time based on the surrounding data points.
        """)
        
    X = df[st.session_state.selected_features]
    y = df[target]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    k_sim = st.slider("Select k for the Simulator", 1, 20, 5)
    knn_sim = KNeighborsClassifier(n_neighbors=k_sim).fit(X_s, y)
    
    st.subheader("Adjust Variables (First 3 Selected Features)")
    
    sim_feats = st.session_state.selected_features[:3]
    synthetic_profile = np.zeros((1, len(st.session_state.selected_features)))
    
    cols = st.columns(len(sim_feats))
    for idx, feat in enumerate(sim_feats):
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        with cols[idx]:
            val = st.slider(feat, float(df[feat].min()), float(df[feat].max()), float(mean_val))
            synthetic_profile[0, idx] = (val - mean_val) / std_val
            
    prediction = knn_sim.predict(synthetic_profile)[0]
    prob = knn_sim.predict_proba(synthetic_profile)[0]
    
    st.markdown("---")
    st.subheader("Real-Time Prediction")
    
    status = target_pos if prediction == 1 else target_neg
    color = "red" if prediction == 1 else "green"
    confidence = prob[1] if prediction == 1 else prob[0]
    
    st.markdown(f"<h2 style='color: {color}; text-align: center;'>Predicted Outcome: {status}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>Model Confidence: {confidence:.0%}</h4>", unsafe_allow_html=True)
    st.info(f"Because you selected k={k_sim}, the model looked at the {k_sim} closest data points to your synthetic profile to make this decision.")
