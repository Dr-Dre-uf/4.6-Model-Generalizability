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

# --- Page Configuration ---
st.set_page_config(page_title="Model Generalizability Sandbox", layout="wide")

# --- Data Loading Functions ---
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
    
    num_feats = ['age', 'lab_glucose', 'lab_creatinine', 'lab_potassium']
    for col in num_feats: df[col] = pd.to_numeric(df[col], errors='coerce')
    df[num_feats] = df[num_feats].fillna(df[num_feats].mean())

    maj = df[df['in_hospital_mortality'] == 0]
    min_ = df[df['in_hospital_mortality'] == 1]
    min_over = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    balanced = pd.concat([maj, min_over]).sample(n=200, random_state=42).reset_index(drop=True)
    
    return balanced, num_feats, 'in_hospital_mortality'

@st.cache_data
def load_foundational_data():
    X, y = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=1, 
                               n_clusters_per_class=1, flip_y=0.1, random_state=42)
    features = ['gene_x_expression', 'protein_y_level', 'culture_ph', 'temperature_c']
    df = pd.DataFrame(X, columns=features)
    # Scale synthetic data to look realistic
    df['gene_x_expression'] = (df['gene_x_expression'] * 10) + 50
    df['protein_y_level'] = (df['protein_y_level'] * 5) + 20
    df['culture_ph'] = (df['culture_ph'] * 0.5) + 7.4
    df['temperature_c'] = (df['temperature_c'] * 1.5) + 37.0
    
    df['cellular_apoptosis'] = y
    return df, features, 'cellular_apoptosis'

# --- Sidebar Configuration ---
st.sidebar.title("Domain Selection")
st.sidebar.info("Instructions: Select the domain context you want to explore.")
track = st.sidebar.radio(
    "Select Track:", 
    ["Clinical (eICU)", "Foundational Science"],
    help="Choose whether to predict patient mortality (Clinical) or cellular apoptosis (Foundational Science)."
)

# Load context-specific data
if track == "Clinical (eICU)":
    df, features, target = load_clinical_data()
    context_desc = "We are using a balanced subset of the eICU Database focusing on 4 features: age, glucose, creatinine, and potassium to predict in-hospital mortality."
else:
    df, features, target = load_foundational_data()
    context_desc = "We are using a simulated foundational science dataset focusing on 4 features: gene expression, protein levels, culture pH, and temperature to predict cellular apoptosis."

# --- App UI ---
st.title("Model Generalizability Sandbox")
st.info("Instructions: Use the tabs below to explore how model complexity (k) and validation strategies impact a model's ability to predict outcomes on unseen data.")

tab1, tab2, tab3 = st.tabs(["Data Preview", "Overfitting and Underfitting", "Validation Strategies"])

# Tab 1: Data Preview
with tab1:
    st.subheader(f"Dataset Quick-Look: {track}")
    st.markdown(context_desc)
    st.dataframe(df.head(10), use_container_width=True)

# Tab 2: Overfitting & Underfitting
with tab2:
    st.header("The Bias-Variance Tradeoff")
    
    X = df[features]
    y = df[target]
    
    test_size_ratio = st.slider(
        "Select Test Set Size Ratio", 
        0.1, 0.5, 0.3, 0.05,
        help="Determines the percentage of data held out for testing. A higher ratio leaves less data for training."
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    st.markdown("#### 1. Accuracy Curve")
    st.caption("Instructions: Observe where the Train and Test lines start to pull apart. The divergence indicates that the model is beginning to overfit the training data.")
    
    k_max = st.slider(
        "Select maximum k to test", 
        5, 50, 20, 
        help="Higher k values average across more neighbors, leading to simpler models. Lower k values listen to fewer neighbors, leading to complex models."
    )
    
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
    st.markdown("#### 2. Visualizing Decision Boundaries")
    st.caption("Instructions: Compare a 'Complex' model (Low k) to a 'Simple' model (High k) using the first two features of your dataset. Notice how a k of 1 tries to draw boundaries around every single outlier.")
    
    c1, c2 = st.columns(2)
    k_low = c1.number_input("Complex Model (k)", 1, 5, 1, help="A low k value makes the model highly sensitive to noise in the training data.")
    k_high = c2.number_input("Simple Model (k)", 10, 50, 15, help="A high k value smooths out the boundaries, potentially missing subtle patterns.")

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
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])

    fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    draw_boundary(k_low, ax1, f"Overfitting (k={k_low})")
    draw_boundary(k_high, ax2, f"Underfitting (k={k_high})")
    st.pyplot(fig_b)

# Tab 3: Validation Strategies
with tab3:
    st.header("Model Stability and Validation")
    st.markdown("#### K-Fold Performance Distribution")
    st.caption("Instructions: Adjust the number of folds to see how stable the model performance is across different slices of data. A wider boxplot indicates higher variance.")
    
    col_cv1, col_cv2 = st.columns(2)
    n_f = col_cv1.slider(
        "Number of Folds", 
        2, 10, 5, 
        help="The number of subsets to split the data into for cross-validation."
    )
    k_cv = col_cv2.slider(
        "Model Complexity (k)", 
        1, 20, 5, 
        help="The n_neighbors parameter used during the cross-validation test."
    )

    knn_cv = KNeighborsClassifier(n_neighbors=k_cv)
    kf = KFold(n_splits=n_f, shuffle=True, random_state=42)
    X_s = StandardScaler().fit_transform(X) 
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
    st.markdown("#### Comparing Validation Strategies")
    st.caption("Instructions: Observe how the Mean Accuracy and variance (error bars) change when you use different validation techniques on the same model.")
    
    skf = StratifiedKFold(n_splits=n_f, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    
    with st.spinner("Calculating metrics..."):
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
