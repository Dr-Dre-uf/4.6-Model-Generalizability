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

# --- Page Configuration ---
st.set_page_config(page_title="Model Generalizability Sandbox", layout="wide")

# --- Data Loading (Cached) ---
@st.cache_data
def load_and_preprocess_data():
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
    
    num_feats = ['age', 'lab_glucose', 'lab_creatinine', 'lab_potassium']
    for col in num_feats: df[col] = pd.to_numeric(df[col], errors='coerce')
    df[num_feats] = df[num_feats].fillna(df[num_feats].mean())

    maj = df[df['in_hospital_mortality'] == 0]
    min_ = df[df['in_hospital_mortality'] == 1]
    min_over = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    balanced = pd.concat([maj, min_over])
    return balanced.sample(n=200, random_state=42).reset_index(drop=True)

df = load_and_preprocess_data()

# --- App UI ---
st.title("Model Generalizability Sandbox")
st.info("Instructions: Use the tabs below to explore how model complexity (k) and validation strategies impact a model's ability to predict patient mortality.")

tab1, tab2, tab3 = st.tabs(["Data Preview", "Overfitting and Underfitting", "Validation Strategies"])

with tab1:
    st.subheader("Dataset Quick-Look")
    st.markdown("We are using a balanced subset of the eICU Database focusing on 4 features: age, glucose, creatinine, and potassium.")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.header("The Bias-Variance Tradeoff")
    
    features = ['age', 'lab_glucose', 'lab_creatinine', 'lab_potassium']
    X = df[features]
    y = df['in_hospital_mortality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    st.markdown("#### 1. Accuracy Curve")
    st.caption("Instructions: Observe where the Train and Test lines start to pull apart. That gap represents overfitting.")
    
    k_max = st.slider("Select maximum k to test", 5, 50, 20, 
                      help="Higher k values lead to simpler models. Lower k values lead to complex models.")
    
    ks = range(1, k_max + 1)
    train_acc, test_acc = [], []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
        train_acc.append(accuracy_score(y_train, knn.predict(X_train_s)))
        test_acc.append(accuracy_score(y_test, knn.predict(X_test_s)))

    fig_acc, ax_acc = plt.subplots(figsize=(8, 3))
    ax_acc.plot(ks, train_acc, label='Train (Memorization)', marker='o', color='#1f77b4')
    ax_acc.plot(ks, test_acc, label='Test (Generalization)', marker='x', color='#ff7f0e')
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Number of Neighbors (k)")
    ax_acc.legend()
    st.pyplot(fig_acc)

    st.markdown("---")
    st.markdown("#### 2. Visualizing Decision Boundaries")
    st.caption("Instructions: Compare a 'Complex' model (Low k) to a 'Simple' model (High k). Notice how k=1 tries to hug every single data point.")
    
    c1, c2 = st.columns(2)
    k_low = c1.number_input("Complex Model (k)", 1, 5, 1, help="Low k is sensitive to noise.")
    k_high = c2.number_input("Simple Model (k)", 10, 50, 15, help="High k smooths out the boundaries.")

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

    fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    draw_boundary(k_low, ax1, f"Overfitting (k={k_low})")
    draw_boundary(k_high, ax2, f"Underfitting (k={k_high})")
    st.pyplot(fig_b)

with tab3:
    st.header("Model Stability")
    st.markdown("#### K-Fold Performance")
    st.caption("Instructions: Adjust the folds to see how stable the model performance is across different slices of data.")
    
    col_cv1, col_cv2 = st.columns(2)
    n_f = col_cv1.slider("Number of Folds", 2, 10, 5, help="Number of splits in K-Fold.")
    k_cv = col_cv2.slider("Model Complexity (k)", 1, 20, 5, help="The n_neighbors for the CV test.")

    knn_cv = KNeighborsClassifier(n_neighbors=k_cv)
    kf = KFold(n_splits=n_f, shuffle=True, random_state=42)
    X_s = scaler.fit_transform(X) 
    scores = cross_val_score(knn_cv, X_s, y, cv=kf)

    col_res1, col_res2 = st.columns([1, 2])
    col_res1.write("**Scores per Fold:**")
    col_res1.dataframe(pd.DataFrame({"Accuracy": scores}), use_container_width=True)
    
    fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=scores, color="#4c72b0", ax=ax_cv)
    sns.stripplot(x=scores, color="black", size=8, jitter=True, ax=ax_cv)
    ax_cv.set_title(f"Accuracy Distribution ({n_f} Folds)")
    col_res2.pyplot(fig_cv)

    st.markdown("---")
    st.markdown("#### Comparing Strategies")
    st.caption("Instructions: See how the 'Mean Accuracy' changes when you use different validation techniques.")
    
    skf = StratifiedKFold(n_splits=n_f, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    
    with st.spinner("Calculating Leave-One-Out..."):
        skf_s = cross_val_score(knn_cv, X_s, y, cv=skf)
        loo_s = cross_val_score(knn_cv, X_s, y, cv=loo)

    methods = ["K-Fold", "Stratified K-Fold", "LOO-CV"]
    means = [scores.mean(), skf_s.mean(), loo_s.mean()]
    sterr = [scores.std(), skf_s.std(), loo_s.std()]

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    ax_bar.bar(methods, means, yerr=sterr, capsize=10, color=['#1f77b4', '#2ca02c', '#d62728'], alpha=0.8)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_ylabel("Mean Accuracy")
    st.pyplot(fig_bar)