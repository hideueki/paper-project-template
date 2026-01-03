# A Novel Prognostic Model for Metastatic Renal Cell Carcinoma in the Immune-Oncology Era

Hideto Ueki, MD, PhD, Takuto Hara, MD, PhD, Taisuke Tobe, MD, Naoto Wakita, MD, PhD, Yasuyoshi Okamura, MD, PhD, Kotaro Suzuki, MD, PhD, Yukari Bando, MD, PhD, Tomoaki Terakawa, MD, PhD, Akihisa Yao, MD, PhD, Koji Chiba, MD, PhD, Jun Teishima, MD, PhD, and Hideaki Miyake, MD, PhD

Department of Urology, Kobe University Graduate School of Medicine, Kobe, Japan

*Correspondence should be addressed to Hideto Ueki:
Tel: +81-78-382-5681
Fax: +81-78-382-5715
E-mail address: hideueki@med.kobe-u.ac.jp

---

## ABSTRACT

**Objective:** To develop and validate a prognostic model optimized for metastatic renal cell carcinoma (mRCC) patients receiving first-line immune checkpoint inhibitor (ICI)-based combination therapy.

**Design, Setting, and Participants:** This multicenter retrospective study included 446 patients with mRCC who received first-line ICI-based combination therapy (IO-IO: nivolumab plus ipilimumab, n=130; IO-TKI: ICI plus tyrosine kinase inhibitor, n=316) between August 2018 and September 2025 at 21 centers in Japan.

**Outcome Measurements and Statistical Analysis:** Prognostic factors were selected using least absolute shrinkage and selection operator (LASSO) Cox regression. Model discrimination was assessed using Harrell's C-index and compared with International Metastatic RCC Database Consortium (IMDC) and Meet-URO scores. Internal validation was performed using bootstrap resampling and cross-validation. Sensitivity analyses examined alternative thresholds for model components.

**Results and Limitations:** The proposed model comprises six factors: Karnofsky Performance Status <80 (1 point), time from diagnosis to treatment <1 year (1 point), anemia (1 point), elevated lactate dehydrogenase (1 point), modified Glasgow Prognostic Score (0–2 points), and ≥2 metastatic sites (1 point). The score stratified patients into favorable (0–2 points; median overall survival [OS], 64.0 mo), intermediate (3–4 points; 39.0 mo), and poor (≥5 points; 12.0 mo) risk groups (p<0.0001). The proposed model demonstrated superior discrimination (C-index 0.740; 95% confidence interval [CI], 0.698–0.775) compared with IMDC (0.700; p=0.006) and Meet-URO (0.691; p=0.001). Within IMDC intermediate-risk patients, the proposed model reclassified 55% as favorable (median OS, 68.0 mo) and 12% as poor (15.0 mo). Limitations include retrospective design and lack of external validation.

**Conclusions:** The proposed model is a novel prognostic model with superior discriminative ability compared with existing risk classification systems in mRCC patients receiving ICI-based combination therapy. The model provides meaningful reclassification within IMDC intermediate-risk patients and demonstrates robust internal validation.

**Patient Summary:** We developed a new risk scoring system for kidney cancer patients receiving immunotherapy combinations. This score uses six easily measured factors and predicts survival more accurately than existing models, helping doctors better counsel patients about their prognosis.

**Keywords:** renal cell carcinoma; immune checkpoint inhibitor; prognostic model; IMDC; risk stratification; modified Glasgow Prognostic Score

---

## 1. INTRODUCTION

Metastatic renal cell carcinoma (mRCC) is a heterogeneous disease with variable clinical outcomes. Over the past two decades, systemic treatment has evolved from cytokine therapy to vascular endothelial growth factor (VEGF)-targeted therapy, and more recently to immune checkpoint inhibitor (ICI)-based combination regimens [1,2]. The International Metastatic RCC Database Consortium (IMDC) risk model, developed by Heng and colleagues in 2009, has served as the standard prognostic tool for mRCC patients receiving VEGF-targeted therapy [3]. The IMDC model incorporates six clinical and laboratory parameters—Karnofsky Performance Status (KPS) <80%, time from diagnosis to treatment <1 year, anemia, neutrophilia, thrombocytosis, and hypercalcemia—to classify patients into favorable (0 factors), intermediate (1–2 factors), and poor (≥3 factors) risk groups. This model has been validated across multiple independent datasets [4].

The treatment landscape of mRCC has dramatically changed with the approval of ICI-based combination therapies. The CheckMate-214 trial established nivolumab plus ipilimumab (IO-IO) as the standard of care for IMDC intermediate- and poor-risk patients, demonstrating significant improvements in overall survival (OS) compared with sunitinib [5]. Subsequently, multiple phase III trials demonstrated the efficacy of ICI plus tyrosine kinase inhibitor (IO-TKI) combinations across all IMDC risk groups, including pembrolizumab plus axitinib (KEYNOTE-426) [6], pembrolizumab plus lenvatinib (CLEAR) [7], nivolumab plus cabozantinib (CheckMate-9ER) [8], and avelumab plus axitinib (JAVELIN Renal 101) [9]. Current guidelines recommend ICI-based combination therapy as first-line treatment for most mRCC patients, with treatment selection guided by IMDC risk classification [10,11]. These practice-changing trials have raised questions about the continued validity of the IMDC model in the immunotherapy era.

Several studies have reported suboptimal performance of the IMDC model in patients receiving ICI-based therapy. In the CheckMate-214 trial, the IMDC model achieved a C-index of only 0.63 in the nivolumab plus ipilimumab arm [12], substantially lower than the originally reported value of 0.73 [3]. The Meet-URO investigators reported a C-index of 0.59 for IMDC in patients receiving second-line nivolumab, prompting them to develop the Meet-URO score incorporating neutrophil-to-lymphocyte ratio (NLR) and bone metastases [13]. The IMmotion151 trial demonstrated that the modified Glasgow Prognostic Score (mGPS), a composite marker of systemic inflammation comprising C-reactive protein (CRP) and albumin, added prognostic value beyond IMDC (C-index 0.72 vs 0.68) [14]. Abuhelwa et al. demonstrated that CRP alone provided superior prognostic accuracy compared with the IMDC model in patients treated with atezolizumab plus bevacizumab [15]. However, recent real-world data from the IMDC consortium suggest that the model retains some prognostic utility even in the combination therapy era [16], highlighting the ongoing debate regarding optimal prognostic tools.

The biological rationale for incorporating inflammation-based markers in the ICI era is compelling. Systemic inflammation, reflected by elevated CRP and reduced albumin, indicates an immunosuppressive tumor microenvironment that may attenuate ICI efficacy [17,18]. Meta-analyses have consistently demonstrated the prognostic value of NLR in RCC [19], and the mGPS has shown strong associations with survival across multiple cancer types [20,21]. The tumor immune microenvironment plays a central role in determining response to ICIs, and systemic inflammatory markers may serve as surrogates for this complex biology [22].

The IMDC intermediate-risk group, comprising the majority of mRCC patients (approximately 60%), represents a particularly heterogeneous population. Sella and colleagues demonstrated significant survival differences between patients with one versus two IMDC risk factors (median OS 27.8 vs 15.0 mo, respectively), highlighting the need for improved stratification within this subgroup [23]. Buti et al. confirmed this heterogeneity in the ICI era, showing that IMDC intermediate-risk patients with one risk factor had outcomes approaching favorable-risk patients [24]. This clinical observation underscores the need for refined prognostic tools.

These observations suggest that the optimal prognostic model for the immunotherapy era should: (1) retain factors that remain prognostic regardless of treatment modality, (2) incorporate inflammation-based markers that reflect the tumor immune microenvironment, and (3) better capture the heterogeneity within current risk categories. To address these needs, we developed and internally validated a novel prognostic model specifically designed for mRCC patients receiving first-line ICI-based combination therapy.

---

## 2. PATIENTS AND METHODS

### 2.1. Study Design and Patients

This retrospective multicenter cohort study included patients with mRCC who received first-line ICI-based combination therapy between August 2018 and September 2025 at Kobe University Hospital and 20 affiliated institutions (21 centers total). Eligible patients were aged ≥18 yr with histologically confirmed RCC and measurable metastatic disease according to Response Evaluation Criteria in Solid Tumors (RECIST) version 1.1. Patients were excluded if they had incomplete baseline laboratory data required for risk score calculation or were lost to follow-up within 30 d of treatment initiation. This study adhered to the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD) guidelines [25].

Treatment regimens were classified into two categories: ICI-ICI combination (nivolumab plus ipilimumab; IO-IO) and ICI-tyrosine kinase inhibitor combination (IO-TKI), which included pembrolizumab plus axitinib, pembrolizumab plus lenvatinib, nivolumab plus cabozantinib, and avelumab plus axitinib. For sensitivity analyses, IO-TKI was further subdivided into IO plus selective VEGFR inhibitor (IO-AXI: pembrolizumab or avelumab plus axitinib) and IO plus multi-kinase inhibitor (IO-MKI: pembrolizumab plus lenvatinib or nivolumab plus cabozantinib) based on the distinct pharmacological profiles of these agents [26,27] (Supplementary Table 1). Notably, IO-IO combination was not approved for patients with IMDC favorable-risk disease in Japan during the study period, consistent with the CheckMate-214 trial results [5].

The study was conducted in accordance with the Declaration of Helsinki and approved by the institutional review board of Kobe University Hospital (approval number: [番号]). The requirement for informed consent was waived due to the retrospective nature of the study.

### 2.2. Data Collection and Outcomes

Clinical data were extracted from electronic medical records and included demographics, Eastern Cooperative Oncology Group performance status (converted to KPS), histological subtype, presence of sarcomatoid features, sites of metastatic disease, and laboratory values at baseline (within 30 d before treatment initiation). Laboratory parameters included hemoglobin, neutrophil count, lymphocyte count, platelet count, corrected calcium, lactate dehydrogenase (LDH), CRP, and albumin. The mGPS was calculated based on CRP and albumin levels according to the original definition [28]: mGPS 0 (CRP ≤10 mg/L), mGPS 1 (CRP >10 mg/L and albumin ≥35 g/L), and mGPS 2 (CRP >10 mg/L and albumin <35 g/L).

The primary endpoint was overall survival (OS), defined as the time from treatment initiation to death from any cause. Patients alive at last follow-up were censored. Secondary endpoints included comparison of prognostic discrimination between the proposed model and existing models (IMDC and Meet-URO), and reclassification analysis within IMDC risk groups.

### 2.3. Development of the Proposed Model

Candidate prognostic factors were identified based on clinical relevance and prior literature, including all six IMDC criteria and additional factors hypothesized to be prognostic in the ICI era: mGPS (0, 1, 2), NLR, number of metastatic sites, and specific metastatic locations (lung, bone, liver, brain, lymph node). To avoid overfitting and selection bias inherent to stepwise methods, we employed least absolute shrinkage and selection operator (LASSO) Cox regression for variable selection [29]. The optimal regularization parameter (λ) was determined by 10-fold cross-validation, selecting the λ that minimized the partial likelihood deviance.

Selected variables were assigned integer point values based on their regression coefficients, rounded to facilitate clinical applicability. The mGPS was incorporated as an ordinal variable (0, 1, or 2 points) consistent with its original definition [20]. Patients were stratified into three risk groups based on score distribution and survival patterns: favorable (0–2 points), intermediate (3–4 points), and poor (≥5 points).

Thresholds for continuous variables were selected based on clinical convention and prior literature rather than data-driven optimization to minimize overfitting. For LDH, we used >250 U/L, corresponding to approximately 1.1–1.2 times the upper limit of normal (ULN) in most participating institutions, consistent with the MSKCC model [30]. For metastatic burden, we used ≥2 organ sites based on the clinical distinction between oligometastatic and polymetastatic disease [31]. Sensitivity analyses examined alternative thresholds: LDH at >200, >225, >275, and >300 U/L; metastatic sites at ≥1 and ≥3; and mGPS as binary (0 vs 1–2) versus ordinal (0, 1, 2) scoring.

### 2.4. Statistical Analysis

Baseline characteristics were summarized using median and interquartile range (IQR) for continuous variables and frequency (percentage) for categorical variables. Standardized mean differences (SMD) were calculated to assess covariate balance between treatment groups [32].

Model discrimination was evaluated using Harrell's concordance index (C-index) with 95% CI calculated by bootstrap resampling (1000 iterations). The proposed model was compared with the IMDC risk model and Meet-URO score using the method described by Pencina et al. [33]. Internal validation was performed using optimism-corrected bootstrap validation (500 iterations) and 5-fold cross-validation. Model calibration was assessed using a calibration plot comparing predicted versus observed 2-year mortality across quintiles of predicted risk. The 2-year timepoint was selected because it represents a clinically meaningful horizon for treatment decisions and prognostic counseling, and is consistent with primary endpoints reported in major ICI combination trials [5–9]. Calibration was quantified using the calibration slope, calibration intercept, and integrated calibration index (ICI), with values of 1.0, 0.0, and 0.0 representing perfect calibration, respectively.

All statistical tests were two-sided, with p<0.05 considered statistically significant. Analyses were performed using Python 3.11 with lifelines 0.27, scikit-learn 1.2, and scipy 1.10 packages.

---

## 3. RESULTS

### 3.1. Patient Characteristics

A total of 446 patients met the inclusion criteria, of whom 130 (29%) received IO-IO and 316 (71%) received IO-TKI combination therapy (Table 1). The median age was 70 yr (IQR, 62–75), and 342 patients (77%) were male. Clear cell histology was present in 360 patients (81%), and sarcomatoid features were identified in 34 patients (8%). Prior nephrectomy had been performed in 312 patients (70%).

According to IMDC criteria, 77 patients (17%) were classified as favorable risk, 281 (63%) as intermediate risk, and 88 (20%) as poor risk. Consistent with treatment guidelines, no IMDC favorable-risk patients received IO-IO combination; all 130 IO-IO recipients had intermediate (n=100, 77%) or poor (n=30, 23%) IMDC risk. Within the IO-TKI cohort, 82 patients (26%) received IO-AXI and 234 (74%) received IO-MKI.

Patients receiving IO-IO had a higher prevalence of time from diagnosis to treatment <1 yr (83% vs 61%, SMD 0.50), lymph node metastases (43% vs 32%, SMD 0.23), and ≥2 metastatic sites (49% vs 36%, SMD 0.27) compared with those receiving IO-TKI.

At a median follow-up of 18 mo (IQR, 8–34), 167 patients (37%) had died. The median follow-up was longer in the IO-IO group (28 mo; IQR, 12–46) compared with the IO-TKI group (16 mo; IQR, 8–28).

### 3.2. Components of the Proposed Model

LASSO Cox regression with 10-fold cross-validation identified six factors independently associated with OS (Table 2, Fig. 1). Notably, two IMDC factors—neutrophilia and thrombocytosis—were not selected by LASSO, while three novel factors—elevated LDH, mGPS, and metastatic burden—were incorporated. The proposed model comprises:

- KPS <80 (1 point): present in 63 patients (14%)
- Time from diagnosis to systemic treatment <1 yr (1 point): present in 302 patients (68%)
- Anemia (hemoglobin below lower limit of normal; 1 point): present in 256 patients (57%)
- Elevated LDH (>250 U/L; 1 point): present in 73 patients (16%)
- mGPS (mGPS 1 = 1 point; mGPS 2 = 2 points): mGPS 1 in 77 patients (17%), mGPS 2 in 118 patients (26%)
- Two or more metastatic sites (1 point): present in 178 patients (40%)

The total score ranges from 0 to 7 points. In univariate analysis, each component demonstrated significant association with OS: KPS <80 (HR 3.09; 95% CI, 2.17–4.41; p<0.001), time to treatment <1 yr (HR 2.34; 95% CI, 1.57–3.50; p<0.001), anemia (HR 2.18; 95% CI, 1.56–3.06; p<0.001), LDH >250 U/L (HR 2.30; 95% CI, 1.60–3.30; p<0.001), mGPS 1 vs 0 (HR 1.33; 95% CI, 0.91–1.93; p=0.14), mGPS 2 vs 0 (HR 2.96; 95% CI, 2.18–4.02; p<0.001), and ≥2 metastatic sites (HR 2.25; 95% CI, 1.66–3.06; p<0.001).

### 3.3. Prognostic Performance by Risk Group

Based on the proposed model, 234 patients (52%) were classified as favorable risk (0–2 points), 122 (27%) as intermediate risk (3–4 points), and 90 (20%) as poor risk (≥5 points) (Table 3, Fig. 2A).

The three risk groups demonstrated significantly different survival outcomes (log-rank p<0.0001). Median OS was 64.0 mo (95% CI, 51.2–not reached [NR]) in the favorable-risk group, 39.0 mo (95% CI, 28.4–52.1) in the intermediate-risk group, and 12.0 mo (95% CI, 9.1–16.8) in the poor-risk group. Two-year OS rates were 78.2% (95% CI, 72.1–83.1), 65.8% (95% CI, 56.2–73.8), and 25.6% (95% CI, 17.1–34.9) for favorable, intermediate, and poor risk groups, respectively. Compared with the favorable-risk group, the HR for death was 2.09 (95% CI, 1.42–3.07; p<0.001) for intermediate risk and 5.48 (95% CI, 3.78–7.94; p<0.001) for poor risk.

### 3.4. Treatment-Stratified Analysis

The prognostic value of the proposed model was consistent across treatment regimens (Supplementary Fig. 1). In the IO-IO cohort (n=130), median OS was 67.0, 32.0, and 11.0 mo for favorable, intermediate, and poor risk groups, respectively (log-rank p<0.0001). In the IO-TKI cohort (n=316), corresponding median OS values were 62.0, 43.0, and 13.0 mo (log-rank p<0.0001). It should be noted that patients classified as favorable risk by the proposed model include those who were IMDC intermediate- or poor-risk and received IO-IO combination therapy; the proposed model is intended for prognostication rather than treatment selection.

Further stratification by IO-TKI subtype revealed consistent prognostic performance. In the IO-AXI subgroup (n=82), median OS was NR, 52.0, and 21.0 mo for favorable, intermediate, and poor risk groups, with 2-yr OS rates of 87.1%, 72.6%, and 32.6%, respectively. In the IO-MKI subgroup (n=234), corresponding median OS values were NR, 31.0, and 10.0 mo, with 2-yr OS rates of 89.4%, 61.0%, and 13.7% (Supplementary Table 1).

### 3.5. Comparison with Existing Prognostic Models

The proposed model demonstrated superior discrimination compared with both IMDC and Meet-URO models (Table 4). In the overall cohort, the C-index was 0.740 (95% CI, 0.698–0.775) for the proposed model, 0.700 (95% CI, 0.659–0.740) for IMDC, and 0.691 (95% CI, 0.650–0.735) for Meet-URO (Fig. 2B).

The proposed model significantly outperformed IMDC (ΔC-index 0.040; 95% CI, 0.012–0.068; p=0.006) and Meet-URO (ΔC-index 0.049; 95% CI, 0.019–0.079; p=0.001). The improvement in discrimination was observed in both treatment subgroups: in the IO-IO cohort, C-index was 0.671 (95% CI, 0.599–0.729) for the proposed model vs 0.635 (95% CI, 0.564–0.697) for IMDC vs 0.645 (95% CI, 0.576–0.710) for Meet-URO; in the IO-TKI cohort, C-index was 0.774 (95% CI, 0.719–0.820) for the proposed model vs 0.725 (95% CI, 0.671–0.779) for IMDC vs 0.715 (95% CI, 0.658–0.771) for Meet-URO.

### 3.6. Internal Validation

Internal validation confirmed the robustness of the proposed model (Supplementary Table 2). The apparent C-index in the training data was 0.741. The optimism-corrected C-index by bootstrap validation (500 iterations) was 0.741, with negligible optimism (−0.000), indicating minimal overfitting. Five-fold cross-validation yielded a mean C-index of 0.742 (SD, 0.033), with individual fold C-indices ranging from 0.701 to 0.788, indicating stable performance across data partitions.

The calibration plot demonstrated good agreement between predicted and observed 2-year mortality across quintiles of predicted risk (Fig. 2C). Quantitative calibration metrics confirmed adequate calibration: calibration slope 0.86 (ideal 1.0), calibration intercept −0.01 (ideal 0), and integrated calibration index (ICI) 0.065. These values indicate that the model's predicted probabilities closely approximate observed outcomes, with minimal systematic over- or under-estimation.

### 3.7. Sensitivity Analyses for Threshold Selection

Sensitivity analyses confirmed the robustness of the selected thresholds for model components (Supplementary Tables 3–5).

For the LDH threshold, C-indices were 0.747 at >200 U/L, 0.748 at >225 U/L, 0.746 at >250 U/L (selected), 0.743 at >275 U/L, and 0.745 at >300 U/L. The selected threshold of >250 U/L provided comparable discrimination while maintaining clinical interpretability aligned with historical definitions.

For mGPS weighting, the original ordinal scoring (0/1/2 points) achieved a C-index of 0.746, while binary scoring (0 vs 1–2) achieved 0.740. The superior performance of ordinal scoring supports the retention of the graded mGPS classification.

For metastatic sites threshold, C-indices were 0.735 at ≥1 site, 0.746 at ≥2 sites (selected), and 0.742 at ≥3 sites. The selected threshold of ≥2 sites provided optimal discrimination while reflecting the clinically meaningful distinction between limited and widespread metastatic disease.

### 3.8. Reclassification of IMDC Risk Groups

The proposed model provided substantial prognostic refinement within IMDC categories (Fig. 3). Among 281 IMDC intermediate-risk patients, the proposed model reclassified 156 (56%) as favorable risk, 77 (27%) as intermediate risk, and 48 (17%) as poor risk, with corresponding median OS of 68.0, 41.0, and 15.0 mo (log-rank p<0.0001; Fig. 3A, 3B). The proposed model achieved a C-index of 0.669 within this heterogeneous IMDC subgroup.

Among 88 IMDC poor-risk patients, the proposed model reclassified 26 (30%) as intermediate risk with median OS of 24.0 mo, compared with 60 (68%) remaining as poor risk with median OS of 10.0 mo, representing a 2.4-fold survival difference (log-rank p<0.0001; Fig. 3C). This suggests potential for identifying patients with better-than-expected prognosis within the IMDC poor-risk category.

---

## 4. DISCUSSION

In this multicenter study of 446 patients from 21 centers in Japan, we developed and validated a novel prognostic model for mRCC patients receiving first-line ICI-based combination therapy. The proposed model demonstrated superior discrimination compared with IMDC and Meet-URO (C-index 0.740 vs 0.700 and 0.691). This improvement was consistent across both IO-IO and IO-TKI treatment subgroups.

The IMDC model was developed in 2009 using data from patients treated with VEGF-targeted therapy [3]. Our finding that IMDC achieved a C-index of 0.700 is consistent with prior reports of reduced performance in the ICI era: the CheckMate-214 trial demonstrated a C-index of only 0.63 [12], and similar reductions have been reported in other cohorts [13,14]. This decline likely reflects different biological mechanisms of ICI versus VEGF-targeted therapy. The Meet-URO score was developed to address IMDC limitations by incorporating NLR and bone metastases [13], and has been validated in patients receiving cabozantinib [34]. However, in our first-line ICI cohort, Meet-URO (C-index 0.691) did not outperform IMDC (0.700), possibly reflecting population differences, as Meet-URO was derived from second-line nivolumab monotherapy patients. Recent IMDC consortium data reported that IMDC retained prognostic utility in contemporary first-line combinations [16]. Our findings confirm this (C-index 0.700) but demonstrate further improvement is achievable through inflammation-based markers and metastatic burden.

The proposed model differs from IMDC in key aspects. First, it excludes neutrophilia and thrombocytosis, which were not independently prognostic in our LASSO analysis, possibly reflecting their association with VEGF-driven rather than immune-mediated biology [3]. Second, elevated LDH (>250 U/L) was originally in the MSKCC model [30] but excluded from IMDC. Meta-analyses confirm LDH association with inferior OS (HR 2.13) [35], and LDH predicts nivolumab response [36]. Sensitivity analysis showed stable C-indices across thresholds from 200 to 300 U/L. Third, mGPS is validated across multiple cancer types [20,21,38]. Elevated CRP indicates an immunosuppressive microenvironment [17,18], and in IMmotion151, mGPS improved prognostic accuracy [14]. Ordinal scoring (0/1/2) outperformed binary classification (C-index 0.746 vs 0.740). Finally, metastatic burden (≥2 sites) captures disease extent, supported by Massari et al. [40], Mori et al. [41], and UK ROC data [42]. The ≥2 sites threshold provided optimal discrimination.

A key limitation of IMDC is that 63% of patients are classified as intermediate risk, representing a heterogeneous population with limited prognostic precision. The proposed model addresses this by reclassifying IMDC intermediate-risk patients (n=281) into three distinct groups: 55% as favorable (median OS 68.0 mo), 33% as intermediate (41.0 mo), and 12% as poor (15.0 mo)—a 4.5-fold difference in survival between extreme groups (C-index 0.669 within IMDC intermediate). This reclassification enables more precise prognostic counseling for the majority of patients who fall into the IMDC intermediate category. Additional advantages include routine availability of all components and maintenance of familiar three-tier classification. Importantly, the proposed model is intended for prognostication, not treatment selection. Treatment decisions should follow IMDC per current guidelines [10,11], as regulatory approvals are IMDC-based. The proposed model complements IMDC by providing refined prognostic information within each IMDC category.

Strengths of this study include multicenter design (21 institutions), LASSO regression minimizing overfitting, comprehensive internal validation (bootstrap and cross-validation), head-to-head comparison with existing models, and sensitivity analyses confirming robustness of threshold selections. Limitations include retrospective design, absence of IMDC favorable-risk patients in IO-IO (reflecting Japanese guidelines), lack of external validation, and unavailability of molecular biomarkers [22,43]. LDH threshold used absolute value rather than ULN ratio. External validation in international cohorts is essential.

---

## 5. CONCLUSIONS

The proposed model demonstrates superior discriminative ability compared with IMDC and Meet-URO in mRCC patients receiving first-line ICI-based combination therapy. By incorporating mGPS and metastatic burden while excluding factors that lose significance in the ICI era, it provides improved risk stratification. Notably, the proposed model meaningfully reclassifies the heterogeneous IMDC intermediate-risk population, enabling more precise prognostic counseling. External validation in international cohorts is warranted.

---

## REFERENCES

[1] Motzer RJ, Jonasch E, Agarwal N, et al. Kidney Cancer, Version 3.2022. J Natl Compr Canc Netw 2022;20:71-90.

[2] Lalani AA, McGregor BA, Albiges L, et al. Systemic Treatment of Metastatic Clear Cell Renal Cell Carcinoma in 2018. Eur Urol 2019;75:100-110.

[3] Heng DY, Xie W, Regan MM, et al. Prognostic factors for overall survival in patients with metastatic renal cell carcinoma treated with VEGF-targeted agents. J Clin Oncol 2009;27:5794-5799.

[4] Heng DY, Xie W, Regan MM, et al. External validation of the IMDC prognostic model. Lancet Oncol 2013;14:141-148.

[5] Motzer RJ, Tannir NM, McDermott DF, et al. Nivolumab plus ipilimumab versus sunitinib in advanced renal-cell carcinoma. N Engl J Med 2018;378:1277-1290.

[6] Rini BI, Plimack ER, Stus V, et al. Pembrolizumab plus axitinib versus sunitinib for advanced renal-cell carcinoma. N Engl J Med 2019;380:1116-1127.

[7] Motzer R, Alekseev B, Rha SY, et al. Lenvatinib plus pembrolizumab or everolimus for advanced renal cell carcinoma. N Engl J Med 2021;384:1289-1300.

[8] Choueiri TK, Powles T, Burotto M, et al. Nivolumab plus cabozantinib versus sunitinib for advanced renal-cell carcinoma. N Engl J Med 2021;384:829-841.

[9] Motzer RJ, Penkov K, Haanen J, et al. Avelumab plus axitinib versus sunitinib for advanced renal-cell carcinoma. N Engl J Med 2019;380:1103-1115.

[10] Grünwald V, Bergmann L, Brehmer B, et al. Systemic Therapy in Metastatic Renal Cell Carcinoma. World J Urol 2022;40:2381-2386.

[11] Powles T, Albiges L, Bex A, et al. ESMO Clinical Practice Guideline on immunotherapy in renal cell carcinoma. Ann Oncol 2021;32:1511-1519.

[12] Motzer RJ, Tannir NM, McDermott DF, et al. Conditional survival with nivolumab plus ipilimumab versus sunitinib. Cancer 2022;128:2085-2097.

[13] Rebuzzi SE, Signori A, Banna GL, et al. Meet-URO 15 study: development of a novel prognostic score. Ther Adv Med Oncol 2021;13:17588359211019642.

[14] Schmidinger M, Larkin J, Atkins MB, et al. mGPS predicts outcome more accurately than IMDC in IMmotion151. Ann Oncol 2022;33:914-924.

[15] Abuhelwa AY, Bellmunt J, Kichenadasse G, et al. CRP provides superior accuracy than IMDC. Front Oncol 2022;12:918993.

[16] Ernst MS, Navani V, Wells JC, et al. IMDC outcomes in contemporary first-line combinations. Eur Urol 2023;84:109-116.

[17] McMillan DC. The Glasgow Prognostic Score: a decade of experience. Cancer Treat Rev 2013;39:534-540.

[18] Proctor MJ, Morrison DS, Talwar D, et al. Inflammation-based prognostic scores in cancer. Eur J Cancer 2011;47:2633-2641.

[19] Hu K, Lou L, Ye J, Zhang S. NLR in renal cell carcinoma: meta-analysis. BMJ Open 2015;5:e006404.

[20] Hu X, Wang Y, Yang WX, et al. mGPS as a prognostic factor for RCC: meta-analysis. Cancer Manag Res 2019;11:6163-6173.

[21] Tong T, Guan Y, Xiong H, et al. Meta-analysis of GPS and mGPS in RCC. Front Oncol 2020;10:1541.

[22] Rebuzzi SE, Perrone F, Bersanelli M, et al. Molecular biomarkers in ICI-treated mRCC: systematic review. Expert Rev Mol Diagn 2020;20:169-185.

[23] Sella A, Michaelson MD, Matczak E, et al. Heterogeneity of intermediate-prognosis mRCC. Clin Genitourin Cancer 2017;15:291-299.e1.

[24] Guida A, Le Teuff G, Alves C, et al. Identification of IMDC intermediate-risk subgroups in patients with metastatic clear-cell renal cell carcinoma. Oncotarget 2020;11:4582-4592.

[25] Collins GS, Reitsma JB, Altman DG, Moons KG. TRIPOD statement. Ann Intern Med 2015;162:55-63.

[26] Rini BI, Powles T, Atkins MB, et al. IMmotion151: atezolizumab plus bevacizumab versus sunitinib. Lancet 2019;393:2404-2415.

[27] Choueiri TK, Escudier B, Powles T, et al. METEOR: cabozantinib versus everolimus. Lancet Oncol 2016;17:917-927.

[28] McMillan DC, Crozier JE, Canna K, et al. GPS evaluation in colorectal cancer. Int J Colorectal Dis 2007;22:881-886.

[29] Tibshirani R. The lasso method for variable selection in the Cox model. Stat Med 1997;16:385-395.

[30] Motzer RJ, Mazumdar M, Bacik J, et al. Survival stratification in advanced RCC. J Clin Oncol 1999;17:2530-2540.

[31] Weichselbaum RR, Hellman S. Oligometastases revisited. Nat Rev Clin Oncol 2011;8:378-382.

[32] Austin PC. Standardized difference to compare prevalence. Commun Stat Simul Comput 2009;38:1228-1234.

[33] Pencina MJ, D'Agostino RB Sr, D'Agostino RB Jr, Vasan RS. Evaluating added predictive ability. Stat Med 2008;27:157-172.

[34] Rebuzzi SE, Cerbone L, Signori A, et al. Meet-URO in cabozantinib-treated mRCC. Ther Adv Med Oncol 2022;14:17588359221079580.

[35] Zhang L, Zha Z, Qu W, et al. LDH in RCC: meta-analysis. PLoS One 2016;11:e0166482.

[36] Ishihara H, Takagi T, Kondo T, et al. LDH as prognostic biomarker for nivolumab. Anticancer Res 2019;39:4371-4377.

[37] Donskov F, von der Maase H. Immune parameters in mRCC. J Clin Oncol 2006;24:1997-2005.

[38] Fan L, Wang X, Chi C, et al. Prognostic nutritional index in mCRPC. Prostate 2017;77:1233-1241.

[39] Fukuda S, Saito K, Yasuda Y, et al. CRP flare-response in nivolumab-treated mRCC. J Immunother Cancer 2021;9:e001564.

[40] Massari F, Di Nunno V, Guida A, et al. Metastatic sites added to IMDC. Clin Genitourin Cancer 2021;19:32-40.

[41] Mori K, Quhal F, Yanagisawa T, et al. Tumor burden in nivolumab plus ipilimumab-treated mRCC. Cancer Immunol Immunother 2022;71:865-873.

[42] Frazer R, McGrane J, Challapalli A, et al. UK ROC real-world data in mRCC. ESMO Real World Data Digit Oncol 2024;3:100027.

[43] Braun DA, Hou Y, Bakouny Z, et al. Somatic alterations and immune infiltration in ccRCC. Nat Med 2020;26:909-918.

---

## TABLES AND FIGURES (Main: 4 Tables, 3 Figures)

**Table 1.** Patient Characteristics by Treatment Group

**Table 2.** Components of the Proposed Model

**Table 3.** Overall Survival by Proposed Model Risk Group

**Table 4.** Comparison of Prognostic Discrimination (C-index)

**Fig. 1.** Proposed Model Scoring System. The proposed model comprises six factors with point allocations: Karnofsky Performance Status <80 (1 point), time from diagnosis to treatment <1 year (1 point), anemia (1 point), elevated lactate dehydrogenase >250 U/L (1 point), modified Glasgow Prognostic Score (0–2 points), and ≥2 metastatic sites (1 point). Total scores range from 0 to 7, with risk classification as favorable (0–2 points), intermediate (3–4 points), or poor (≥5 points).

**Fig. 2.** Primary Survival Analysis and Model Performance. (A) Kaplan-Meier curves for overall survival stratified by proposed model risk groups (favorable, intermediate, poor). Log-rank P<0.001. (B) Forest plot comparing C-indices between the proposed model and IMDC with 95% confidence intervals. The proposed model demonstrated significantly superior discrimination (ΔC-index 0.040, P=0.006). (C) Calibration plot comparing predicted versus observed 2-year mortality across quintiles of predicted risk. Points represent quintile groups with 95% confidence intervals; dashed line indicates perfect calibration. Quantitative metrics shown: calibration slope 0.86, intercept −0.01, integrated calibration index (ICI) 0.065.

**Fig. 3.** Reclassification of IMDC Risk Groups by the Proposed Model. (A) Stacked bar chart showing the proportion of patients reclassified from each IMDC category (favorable, intermediate, poor) to proposed model risk categories. Percentages indicate the proportion within each IMDC category. (B) Kaplan-Meier curves for IMDC intermediate-risk patients stratified by the proposed model, demonstrating significant survival differences among reclassified groups (log-rank P<0.001). (C) Kaplan-Meier curves for IMDC poor-risk patients stratified by the proposed model, showing that patients reclassified as intermediate risk had substantially better survival than those remaining as poor risk (log-rank P<0.001).

---

## SUPPLEMENTARY MATERIAL

**Supplementary Table 1.** Treatment Details by Regimen and Proposed Model Risk Group

**Supplementary Table 2.** Internal Validation of the Proposed Model

**Supplementary Table 3.** Sensitivity Analysis for LDH Threshold

**Supplementary Table 4.** Sensitivity Analysis for mGPS Weighting

**Supplementary Table 5.** Sensitivity Analysis for Metastatic Sites Threshold

**Supplementary Fig. 1.** Treatment-Stratified Analysis. Kaplan-Meier curves for overall survival by proposed model risk group stratified by treatment regimen: (A) IO-IO cohort (nivolumab plus ipilimumab; n=130), (B) IO-TKI cohort (ICI plus tyrosine kinase inhibitor; n=316). The proposed model maintained prognostic discrimination in both treatment subgroups (log-rank p<0.0001 for both). Note: IO-TKI cohort x-axis limited to 60 months due to sparse follow-up beyond this time point.

**Supplementary Fig. 2.** Sensitivity Analyses. (A) C-index across different LDH thresholds. (B) Comparison of ordinal versus binary mGPS scoring.
