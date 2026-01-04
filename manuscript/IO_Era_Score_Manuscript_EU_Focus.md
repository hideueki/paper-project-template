# A Novel Prognostic Model for Metastatic Renal Cell Carcinoma in the Immune-Oncology Era

Hideto Ueki, MD, PhD, Takuto Hara, MD, PhD, Taisuke Tobe, MD, Naoto Wakita, MD, PhD, Yasuyoshi Okamura, MD, PhD, Kotaro Suzuki, MD, PhD, Yukari Bando, MD, PhD, Tomoaki Terakawa, MD, PhD, Akihisa Yao, MD, PhD, Koji Chiba, MD, PhD, Jun Teishima, MD, PhD, and Hideaki Miyake, MD, PhD

Department of Urology, Kobe University Graduate School of Medicine, Kobe, Japan

*Correspondence should be addressed to Hideto Ueki:
Tel: +81-78-382-5681
Fax: +81-78-382-5715
E-mail address: hideueki@med.kobe-u.ac.jp

---

## ABSTRACT

**Background and Objective:** The IMDC model shows reduced prognostic accuracy in patients receiving immune checkpoint inhibitor (ICI)-based therapy, with 63% falling into the heterogeneous intermediate-risk category. We developed a prognostic model for metastatic renal cell carcinoma (mRCC) patients receiving first-line ICI-based combination therapy.

**Methods:** This multicenter retrospective study included 446 mRCC patients receiving first-line ICI-based therapy at 21 Japanese centers (2018–2025). Prognostic factors were selected using LASSO Cox regression. Model discrimination (C-index) was compared with IMDC and Meet-URO. Internal validation used bootstrap and cross-validation.

**Key Findings and Limitations:** The model comprises six factors: KPS <80, time to treatment <1 year, anemia, elevated LDH, mGPS, and ≥2 metastatic sites. Patients were stratified into favorable (0–2 points; median OS 64 mo), intermediate (3–4 points; 39 mo), and poor (≥5 points; 12 mo) groups. The model showed superior discrimination (C-index 0.740) versus IMDC (0.700; p=0.006) and Meet-URO (0.691; p=0.001). Among IMDC intermediate-risk patients, 55% were reclassified as favorable (OS 68 mo) and 12% as poor (15 mo). Limitations include retrospective design and lack of external validation.

**Conclusions and Clinical Implications:** This model provides superior discrimination and meaningful reclassification of IMDC intermediate-risk patients. These findings support improved prognostic counseling. External validation is warranted.

**Keywords:** renal cell carcinoma; immune checkpoint inhibitor; prognostic model; IMDC; risk stratification; modified Glasgow Prognostic Score

---

## What does the study add?

Existing prognostic models for metastatic renal cell carcinoma were developed in the VEGF-targeted therapy era and show reduced accuracy in patients receiving immune checkpoint inhibitor combinations. We developed a novel prognostic model incorporating inflammation-based markers (mGPS) and metastatic burden that demonstrates superior discrimination (C-index 0.740 vs 0.700 for IMDC). The model meaningfully reclassifies 67% of IMDC intermediate-risk patients into distinct prognostic subgroups with median survival ranging from 15 to 68 months.

---

## Patient Summary

We developed a new risk scoring system for kidney cancer patients receiving immunotherapy combinations. This score uses six easily measured factors and predicts survival more accurately than existing models, helping doctors better counsel patients about their expected outcomes.

---

## Take Home Message

A novel prognostic model incorporating inflammation markers and metastatic burden outperforms IMDC in mRCC patients receiving ICI-based therapy, enabling refined risk stratification particularly within IMDC intermediate-risk patients.

---

## 1. INTRODUCTION

The International Metastatic RCC Database Consortium (IMDC) model has served as the standard prognostic tool for metastatic renal cell carcinoma (mRCC) [3,4]. However, with the approval of ICI-based combination therapies—including nivolumab plus ipilimumab [5], pembrolizumab plus axitinib [6], pembrolizumab plus lenvatinib [7], nivolumab plus cabozantinib [8], and avelumab plus axitinib [9]—the IMDC model has shown reduced performance. In CheckMate-214, IMDC achieved a C-index of only 0.63 [12], compared with 0.73 originally [3].

Several groups have proposed modifications. The Meet-URO score incorporates neutrophil-to-lymphocyte ratio (NLR) and bone metastases [13]. IMmotion151 demonstrated that modified Glasgow Prognostic Score (mGPS) added prognostic value beyond IMDC [14]. The biological rationale is compelling: systemic inflammation indicates an immunosuppressive microenvironment that may attenuate ICI efficacy [17,18].

A key limitation of IMDC is that 63% of patients fall into the heterogeneous intermediate-risk category. Sella et al. demonstrated significant survival differences between patients with one versus two IMDC factors [23], highlighting the need for improved stratification.

We developed and validated a novel prognostic model for mRCC patients receiving first-line ICI-based combination therapy, incorporating inflammation-based markers and metastatic burden.

---

## 2. MATERIALS (PATIENTS) AND METHODS

### 2.1. Study Design and Patients

This retrospective multicenter study included mRCC patients who received first-line ICI-based combination therapy (August 2018–September 2025) at 21 Japanese centers. Eligible patients had histologically confirmed RCC with measurable metastatic disease. Patients with incomplete baseline laboratory data were excluded. The study adhered to TRIPOD guidelines [25] and was approved by the institutional review board of Kobe University Hospital.

Treatment regimens included IO-IO (nivolumab plus ipilimumab) and IO-TKI (pembrolizumab plus axitinib, pembrolizumab plus lenvatinib, nivolumab plus cabozantinib, or avelumab plus axitinib). IO-IO was not approved for IMDC favorable-risk disease in Japan [5].

### 2.2. Data Collection and Outcomes

Clinical data included demographics, KPS, histology, sites of metastatic disease, and baseline laboratory values. The mGPS was calculated as: 0 (CRP ≤10 mg/L), 1 (CRP >10 mg/L and albumin ≥35 g/L), or 2 (CRP >10 mg/L and albumin <35 g/L) [28].

The primary endpoint was overall survival (OS). Secondary endpoints included comparison with IMDC and Meet-URO models, and reclassification analysis.

### 2.3. Model Development

LASSO Cox regression was used for variable selection to avoid overfitting [29]. Candidate factors included IMDC criteria plus mGPS, NLR, and metastatic burden. Selected variables were assigned integer points based on regression coefficients. Thresholds were based on prior literature: LDH >250 U/L [30] and ≥2 metastatic sites [31]. Sensitivity analyses examined alternative thresholds.

### 2.4. Statistical Analysis

Model discrimination was evaluated using C-index with bootstrap 95% CI (1000 iterations) and compared using Pencina's method [33]. Internal validation used bootstrap (500 iterations) and 5-fold cross-validation. Calibration was assessed at 2 years, consistent with major ICI trial endpoints [5–9], using calibration slope, intercept, and integrated calibration index.

---

## 3. RESULTS

### 3.1. Patient Characteristics

A total of 446 patients were included: 130 (29%) received IO-IO and 316 (71%) received IO-TKI (Table 1). Median age was 70 years; 77% were male. By IMDC, 77 (17%) were favorable, 281 (63%) intermediate, and 88 (20%) poor risk. No IMDC favorable-risk patients received IO-IO. At median follow-up of 18 months, 167 patients (37%) had died.

### 3.2. Model Components

LASSO regression identified six factors (Table 2, Fig. 1): KPS <80 (1 point), time to treatment <1 year (1 point), anemia (1 point), LDH >250 U/L (1 point), mGPS (1–2 points), and ≥2 metastatic sites (1 point). Notably, neutrophilia and thrombocytosis were not selected.

### 3.3. Prognostic Performance

The model stratified patients into favorable (52%), intermediate (27%), and poor (20%) risk (Fig. 2A). Median OS was 64, 39, and 12 months, respectively (p<0.0001). Two-year OS rates were 78%, 66%, and 26%.

### 3.4. Comparison with Existing Models

The model demonstrated superior discrimination (C-index 0.740) versus IMDC (0.700; p=0.006) and Meet-URO (0.691; p=0.001) (Fig. 2B, Supplementary Table 2). Improvement was consistent across treatment subgroups.

### 3.5. Internal Validation

Bootstrap validation showed minimal overfitting (optimism −0.000). Cross-validation yielded C-index 0.742±0.033. Calibration was adequate: slope 0.86, intercept −0.01, ICI 0.065 (Fig. 2C).

### 3.6. Reclassification Analysis

Among 281 IMDC intermediate-risk patients, 56% were reclassified as favorable (median OS 68 mo), 27% as intermediate (41 mo), and 17% as poor (15 mo) (p<0.0001; Fig. 3A, 3B). Among 88 IMDC poor-risk patients, 30% were reclassified as intermediate (OS 24 mo) versus 68% remaining poor (10 mo) (Fig. 3C).

---

## 4. DISCUSSION

We developed a prognostic model for mRCC patients receiving first-line ICI-based therapy that demonstrated superior discrimination compared with IMDC and Meet-URO (C-index 0.740 vs 0.700 and 0.691).

The model differs from IMDC in excluding neutrophilia and thrombocytosis while incorporating LDH, mGPS, and metastatic burden. LDH was in the original MSKCC model [30] and predicts ICI response [35,36]. mGPS reflects tumor-associated inflammation that may attenuate ICI efficacy [14,17,18,20,21]. Metastatic burden (≥2 sites) captures disease extent [37–39].

Importantly, the model meaningfully reclassifies IMDC intermediate-risk patients (63% of cohort) into three groups with median OS ranging from 15 to 68 months—a 4.5-fold difference. This enables more precise prognostic counseling. The model is intended for prognostication, not treatment selection; treatment decisions should follow IMDC per guidelines [10,11].

Strengths include multicenter design, LASSO regression, and comprehensive validation. Limitations include retrospective design, lack of external validation, and unavailability of molecular biomarkers [22,40]. External validation is essential.

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

[37] Massari F, Di Nunno V, Guida A, et al. Metastatic sites added to IMDC. Clin Genitourin Cancer 2021;19:32-40.

[38] Mori K, Quhal F, Yanagisawa T, et al. Tumor burden in nivolumab plus ipilimumab-treated mRCC. Cancer Immunol Immunother 2022;71:865-873.

[39] Frazer R, McGrane J, Challapalli A, et al. UK ROC real-world data in mRCC. ESMO Real World Data Digit Oncol 2024;3:100027.

[40] Braun DA, Hou Y, Bakouny Z, et al. Somatic alterations and immune infiltration in ccRCC. Nat Med 2020;26:909-918.

---

## TABLES AND FIGURES (Main: 2 Tables, 3 Figures)

**Table 1.** Baseline Patient Characteristics

**Table 2.** Components of the IO-Era Prognostic Score

**Fig. 1.** IO-Era Prognostic Model Scoring System. The model comprises six factors with point allocations: Karnofsky Performance Status <80 (1 point), time from diagnosis to treatment <1 year (1 point), anemia (1 point), elevated lactate dehydrogenase >250 U/L (1 point), modified Glasgow Prognostic Score (0–2 points), and ≥2 metastatic sites (1 point). Total scores range from 0 to 8, with risk classification as favorable (0–2 points), intermediate (3–4 points), or poor (≥5 points).

**Fig. 2.** Primary Survival Analysis and Model Performance. (A) Kaplan-Meier curves for overall survival stratified by proposed model risk groups (favorable, intermediate, poor). Log-rank P<0.001. (B) Forest plot comparing C-indices between the proposed model, IMDC, and Meet-URO with 95% confidence intervals. The proposed model demonstrated significantly superior discrimination (ΔC-index 0.040 vs IMDC, P=0.006). (C) Calibration plot comparing predicted versus observed 2-year mortality across quintiles of predicted risk. Points represent quintile groups with 95% confidence intervals; dashed line indicates perfect calibration. Quantitative metrics: calibration slope 0.86, intercept −0.01, integrated calibration index 0.065.

**Fig. 3.** Reclassification of IMDC Risk Groups by the Proposed Model. (A) Stacked bar chart showing the proportion of patients reclassified from each IMDC category to proposed model risk categories. (B) Kaplan-Meier curves for IMDC intermediate-risk patients stratified by the proposed model (log-rank P<0.001). (C) Kaplan-Meier curves for IMDC poor-risk patients stratified by the proposed model (log-rank P<0.001).

---

## SUPPLEMENTARY MATERIAL

**Supplementary Table 1.** Overall Survival by Proposed Model Risk Group

**Supplementary Table 2.** Comparison of Prognostic Discrimination (C-index)

**Supplementary Table 3.** Treatment Details by Regimen and Proposed Model Risk Group

**Supplementary Table 4.** Internal Validation of the Proposed Model

**Supplementary Table 5.** Sensitivity Analysis for LDH Threshold

**Supplementary Table 6.** Sensitivity Analysis for mGPS Weighting

**Supplementary Table 7.** Sensitivity Analysis for Metastatic Sites Threshold

**Supplementary Fig. 1.** Treatment-Stratified Analysis. Kaplan-Meier curves for overall survival by proposed model risk group stratified by treatment regimen: (A) IO-IO cohort (n=130), (B) IO-TKI cohort (n=316). Log-rank p<0.0001 for both. IO-TKI x-axis limited to 60 months.

**Supplementary Fig. 2.** Sensitivity Analyses. (A) C-index across different LDH thresholds. (B) Comparison of ordinal versus binary mGPS scoring.
