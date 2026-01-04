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

Metastatic renal cell carcinoma (mRCC) is a heterogeneous disease with variable clinical outcomes. The International Metastatic RCC Database Consortium (IMDC) model, developed by Heng et al. in 2009, has served as the standard prognostic tool for mRCC patients receiving VEGF-targeted therapy [3]. The IMDC model incorporates six clinical and laboratory parameters—Karnofsky Performance Status (KPS) <80%, time from diagnosis to treatment <1 year, anemia, neutrophilia, thrombocytosis, and hypercalcemia—to classify patients into favorable, intermediate, and poor risk groups [4].

The treatment landscape of mRCC has dramatically changed with the approval of ICI-based combination therapies. The CheckMate-214 trial established nivolumab plus ipilimumab as the standard of care for IMDC intermediate- and poor-risk patients [5]. Subsequently, multiple phase III trials demonstrated the efficacy of ICI plus tyrosine kinase inhibitor (TKI) combinations, including pembrolizumab plus axitinib [6], pembrolizumab plus lenvatinib [7], nivolumab plus cabozantinib [8], and avelumab plus axitinib [9]. Current guidelines recommend ICI-based combination therapy as first-line treatment for most mRCC patients [10,11].

Several studies have reported suboptimal performance of the IMDC model in patients receiving ICI-based therapy. In CheckMate-214, IMDC achieved a C-index of only 0.63 in the nivolumab plus ipilimumab arm [12], substantially lower than the originally reported value of 0.73 [3]. The Meet-URO score was developed to address IMDC limitations by incorporating neutrophil-to-lymphocyte ratio (NLR) and bone metastases [13]. The IMmotion151 trial demonstrated that the modified Glasgow Prognostic Score (mGPS), a composite marker of systemic inflammation, added prognostic value beyond IMDC [14]. The biological rationale for incorporating inflammation-based markers is compelling: systemic inflammation indicates an immunosuppressive tumor microenvironment that may attenuate ICI efficacy [17,18].

A key limitation of IMDC is that approximately 63% of patients fall into the heterogeneous intermediate-risk category. Sella et al. demonstrated significant survival differences between patients with one versus two IMDC factors (median OS 27.8 vs 15.0 months), highlighting the need for improved stratification within this subgroup [23].

We developed and internally validated a novel prognostic model specifically designed for mRCC patients receiving first-line ICI-based combination therapy, incorporating inflammation-based markers and metastatic burden.

---

## 2. MATERIALS (PATIENTS) AND METHODS

### 2.1. Study Design and Patients

This retrospective multicenter cohort study included patients with mRCC who received first-line ICI-based combination therapy between August 2018 and September 2025 at Kobe University Hospital and 20 affiliated institutions (21 centers total). Eligible patients were aged ≥18 years with histologically confirmed RCC and measurable metastatic disease according to RECIST version 1.1. Patients were excluded if they had incomplete baseline laboratory data required for risk score calculation or were lost to follow-up within 30 days of treatment initiation. The study adhered to TRIPOD guidelines [25] and was approved by the institutional review board of Kobe University Hospital.

Treatment regimens were classified into ICI-ICI combination (IO-IO: nivolumab plus ipilimumab) and ICI-TKI combination (IO-TKI: pembrolizumab plus axitinib, pembrolizumab plus lenvatinib, nivolumab plus cabozantinib, or avelumab plus axitinib). IO-IO was not approved for IMDC favorable-risk disease in Japan during the study period, consistent with CheckMate-214 results [5].

### 2.2. Data Collection and Outcomes

Clinical data were extracted from electronic medical records and included demographics, performance status (converted to KPS), histological subtype, sarcomatoid features, sites of metastatic disease, and laboratory values at baseline (within 30 days before treatment). Laboratory parameters included hemoglobin, neutrophil count, platelet count, corrected calcium, lactate dehydrogenase (LDH), C-reactive protein (CRP), and albumin. The mGPS was calculated as: 0 (CRP ≤10 mg/L), 1 (CRP >10 mg/L and albumin ≥35 g/L), or 2 (CRP >10 mg/L and albumin <35 g/L) [28].

The primary endpoint was overall survival (OS), defined as time from treatment initiation to death from any cause. Secondary endpoints included comparison of prognostic discrimination between the proposed model and existing models (IMDC and Meet-URO), and reclassification analysis within IMDC risk groups.

### 2.3. Model Development

Candidate prognostic factors included all six IMDC criteria plus additional factors: mGPS, NLR, number of metastatic sites, and specific metastatic locations. To avoid overfitting and selection bias inherent to stepwise methods, we employed LASSO Cox regression for variable selection [29]. The optimal regularization parameter was determined by 10-fold cross-validation.

Selected variables were assigned integer point values based on regression coefficients. Thresholds for continuous variables were based on prior literature rather than data-driven optimization: LDH >250 U/L consistent with the MSKCC model [30], and ≥2 metastatic sites based on the distinction between oligometastatic and polymetastatic disease [31]. Sensitivity analyses examined alternative thresholds for LDH (>200, >225, >275, >300 U/L), metastatic sites (≥1, ≥3), and mGPS (binary vs ordinal scoring).

### 2.4. Statistical Analysis

Baseline characteristics were summarized using median (IQR) for continuous variables and frequency (percentage) for categorical variables. Standardized mean differences (SMD) were calculated to assess covariate balance between treatment groups [32].

Model discrimination was evaluated using Harrell's C-index with 95% CI calculated by bootstrap resampling (1000 iterations). The proposed model was compared with IMDC and Meet-URO using Pencina's method [33]. Internal validation was performed using optimism-corrected bootstrap validation (500 iterations) and 5-fold cross-validation. Model calibration was assessed at 2 years using calibration slope, intercept, and integrated calibration index (ICI). The 2-year timepoint was selected for consistency with major ICI combination trial endpoints [5–9]. All analyses were performed using Python 3.11.

---

## 3. RESULTS

### 3.1. Patient Characteristics

A total of 446 patients met the inclusion criteria, of whom 130 (29%) received IO-IO and 316 (71%) received IO-TKI combination therapy (Table 1). The median age was 70 years (IQR, 62–75), and 342 patients (77%) were male. Clear cell histology was present in 360 patients (81%), and sarcomatoid features were identified in 34 patients (8%).

According to IMDC criteria, 77 patients (17%) were classified as favorable risk, 281 (63%) as intermediate risk, and 88 (20%) as poor risk. Consistent with treatment guidelines, no IMDC favorable-risk patients received IO-IO combination; all 130 IO-IO recipients had intermediate (n=100, 77%) or poor (n=30, 23%) IMDC risk. Patients receiving IO-IO had higher prevalence of time from diagnosis to treatment <1 year (83% vs 61%, SMD 0.50) and ≥2 metastatic sites (49% vs 36%, SMD 0.27) compared with IO-TKI.

At median follow-up of 18 months (IQR, 8–34), 167 patients (37%) had died. Median follow-up was longer in the IO-IO group (28 months) compared with IO-TKI (16 months).

### 3.2. Model Components

LASSO Cox regression with 10-fold cross-validation identified six factors independently associated with OS (Table 2, Fig. 1). Notably, two IMDC factors—neutrophilia and thrombocytosis—were not selected, while three novel factors—elevated LDH, mGPS, and metastatic burden—were incorporated. The proposed model comprises:

- KPS <80 (1 point): present in 63 patients (14%)
- Time from diagnosis to treatment <1 year (1 point): present in 302 patients (68%)
- Anemia (hemoglobin below lower limit of normal; 1 point): present in 256 patients (57%)
- Elevated LDH (>250 U/L; 1 point): present in 73 patients (16%)
- mGPS (mGPS 1 = 1 point; mGPS 2 = 2 points): mGPS 1 in 77 patients (17%), mGPS 2 in 118 patients (26%)
- ≥2 metastatic sites (1 point): present in 178 patients (40%)

The total score ranges from 0 to 8 points.

### 3.3. Prognostic Performance by Risk Group

Based on the proposed model, 234 patients (52%) were classified as favorable risk (0–2 points), 122 (27%) as intermediate risk (3–4 points), and 90 (20%) as poor risk (≥5 points) (Supplementary Table 1, Fig. 2A).

The three risk groups demonstrated significantly different survival outcomes (log-rank p<0.0001). Median OS was 64.0 months (95% CI, 51.2–NR) in the favorable-risk group, 39.0 months (95% CI, 28.4–52.1) in the intermediate-risk group, and 12.0 months (95% CI, 9.1–16.8) in the poor-risk group. Two-year OS rates were 78.2%, 65.8%, and 25.6% for favorable, intermediate, and poor risk groups, respectively. Compared with favorable risk, HR for death was 2.09 (95% CI, 1.42–3.07) for intermediate and 5.48 (95% CI, 3.78–7.94) for poor risk.

The prognostic value was consistent across treatment regimens (Supplementary Fig. 1). In IO-IO (n=130), median OS was 67, 32, and 11 months for favorable, intermediate, and poor risk, respectively. In IO-TKI (n=316), corresponding values were 62, 43, and 13 months.

### 3.4. Comparison with Existing Models

The proposed model demonstrated superior discrimination compared with both IMDC and Meet-URO (Supplementary Table 2, Fig. 2B). In the overall cohort, C-index was 0.740 (95% CI, 0.698–0.775) for the proposed model, 0.700 (95% CI, 0.659–0.740) for IMDC, and 0.691 (95% CI, 0.650–0.735) for Meet-URO.

The proposed model significantly outperformed IMDC (ΔC-index 0.040; 95% CI, 0.012–0.068; p=0.006) and Meet-URO (ΔC-index 0.049; 95% CI, 0.019–0.079; p=0.001). Improvement was consistent in both treatment subgroups: IO-IO (0.671 vs 0.635 for IMDC) and IO-TKI (0.774 vs 0.725 for IMDC).

### 3.5. Internal Validation

Internal validation confirmed model robustness (Supplementary Table 4). The apparent C-index was 0.741. Optimism-corrected C-index by bootstrap validation was 0.741 with negligible optimism (−0.000), indicating minimal overfitting. Five-fold cross-validation yielded mean C-index of 0.742 (SD, 0.033).

The calibration plot demonstrated good agreement between predicted and observed 2-year mortality (Fig. 2C). Quantitative metrics confirmed adequate calibration: slope 0.86, intercept −0.01, and ICI 0.065.

### 3.6. Sensitivity Analyses

Sensitivity analyses confirmed threshold robustness (Supplementary Tables 5–7). For LDH, C-indices ranged from 0.743 to 0.748 across thresholds (>200 to >300 U/L). For mGPS, ordinal scoring (0/1/2) achieved C-index 0.746 versus 0.740 for binary. For metastatic sites, ≥2 sites (0.746) outperformed ≥1 (0.735) and ≥3 (0.742).

### 3.7. Reclassification of IMDC Risk Groups

The proposed model provided substantial prognostic refinement within IMDC categories (Fig. 3). Among 281 IMDC intermediate-risk patients, 156 (56%) were reclassified as favorable (median OS 68.0 mo), 77 (27%) as intermediate (41.0 mo), and 48 (17%) as poor (15.0 mo) (log-rank p<0.0001; Fig. 3A, 3B). The model achieved C-index of 0.669 within this heterogeneous subgroup.

Among 88 IMDC poor-risk patients, 26 (30%) were reclassified as intermediate (median OS 24.0 mo) compared with 60 (68%) remaining as poor (10.0 mo), representing a 2.4-fold survival difference (Fig. 3C).

---

## 4. DISCUSSION

In this multicenter study of 446 patients from 21 centers in Japan, we developed and validated a novel prognostic model for mRCC patients receiving first-line ICI-based combination therapy. The proposed model demonstrated superior discrimination compared with IMDC and Meet-URO (C-index 0.740 vs 0.700 and 0.691), with improvement consistent across both IO-IO and IO-TKI subgroups. To our knowledge, this is the first prognostic model specifically developed for patients receiving contemporary ICI-based combination regimens in the first-line setting.

The IMDC model was developed in 2009 using data from patients treated with VEGF-targeted therapy [3]. Our finding that IMDC achieved a C-index of 0.700 is consistent with prior reports of reduced performance in the ICI era: CheckMate-214 demonstrated a C-index of 0.63 [12], and similar reductions have been observed in other cohorts [13,14]. This decline likely reflects fundamentally different biological mechanisms underlying response to ICI versus VEGF-targeted therapy. While VEGF inhibitors primarily target tumor angiogenesis, ICIs restore antitumor immunity by blocking inhibitory checkpoints, making tumor-immune interactions the primary determinant of treatment efficacy.

The Meet-URO score was developed to address IMDC limitations by incorporating NLR and bone metastases [13], and has been validated in patients receiving cabozantinib [34]. However, in our first-line ICI combination cohort, Meet-URO (C-index 0.691) did not outperform IMDC (0.700), possibly reflecting population differences as Meet-URO was derived from second-line nivolumab monotherapy patients. Recent IMDC consortium data reported that IMDC retained prognostic utility in contemporary first-line combinations [16], consistent with our findings. However, our results demonstrate that further improvement is achievable through incorporation of inflammation-based markers and metastatic burden.

The proposed model differs from IMDC in several key aspects that reflect the unique biology of ICI-based therapy. First, it excludes neutrophilia and thrombocytosis, which were not independently prognostic in our LASSO analysis. This finding may reflect their primary association with VEGF-driven tumor biology rather than immune-mediated mechanisms [3]. Notably, these factors were originally identified in patients receiving cytokine or VEGF-targeted therapy, and their prognostic significance may diminish when treatment efficacy depends on host antitumor immunity.

Second, elevated LDH (>250 U/L) was incorporated, consistent with the original MSKCC model [30] but excluded from IMDC. LDH elevation reflects high tumor burden and glycolytic metabolism, both of which have been associated with an immunosuppressive tumor microenvironment. Meta-analyses confirm LDH association with inferior OS in RCC (pooled HR 2.13) [35], and LDH independently predicts response to nivolumab in mRCC [36]. Our sensitivity analyses demonstrated stable C-indices across thresholds from 200 to 300 U/L, supporting the robustness of this component.

Third, the mGPS emerged as a key prognostic factor. This inflammation-based score, comprising CRP and albumin, is validated across multiple cancer types [20,21] and reflects the systemic inflammatory response to malignancy. Elevated CRP indicates activation of pro-inflammatory pathways that promote an immunosuppressive microenvironment through recruitment of myeloid-derived suppressor cells and regulatory T cells [17,18]. In IMmotion151, mGPS improved prognostic accuracy beyond IMDC in patients receiving atezolizumab plus bevacizumab [14]. Our finding that ordinal scoring (0/1/2) outperformed binary classification (C-index 0.746 vs 0.740) supports retention of the graded mGPS classification, allowing capture of the dose-response relationship between inflammation severity and prognosis.

Finally, metastatic burden (≥2 sites) was incorporated to capture disease extent. Multiple studies have demonstrated that metastatic burden independently predicts outcomes in ICI-treated patients [37–39]. The ≥2 sites threshold provided optimal discrimination while reflecting the clinically meaningful distinction between oligometastatic and polymetastatic disease states.

A key clinical advantage of the proposed model is the meaningful reclassification of IMDC intermediate-risk patients, who comprise 63% of the cohort. This heterogeneous population has long been recognized as problematic for clinical decision-making [23,24]. The proposed model stratifies these patients into three distinct groups with median OS ranging from 15 to 68 months—a 4.5-fold difference representing clinically meaningful prognostic separation. This reclassification enables more precise prognostic counseling for the majority of mRCC patients and may inform shared decision-making regarding treatment intensity and goals of care.

Importantly, the proposed model is intended for prognostication, not treatment selection. Treatment decisions should follow IMDC per current guidelines [10,11], as regulatory approvals for ICI combinations are IMDC-based. The proposed model complements IMDC by providing refined prognostic information within each IMDC category, allowing clinicians to better individualize discussions with patients regarding expected outcomes.

Strengths of this study include multicenter design (21 institutions) ensuring generalizability within Japan, use of LASSO regression minimizing overfitting and selection bias, comprehensive internal validation using both bootstrap and cross-validation, head-to-head comparison with established prognostic models, and extensive sensitivity analyses confirming robustness of threshold selections. All model components are routinely available clinical and laboratory parameters, facilitating implementation in daily practice.

Limitations warrant consideration. First, the retrospective design introduces potential selection and information biases. Second, absence of IMDC favorable-risk patients in the IO-IO cohort reflects Japanese regulatory guidelines and limits generalizability to this population. Third, the study lacks external validation; prospective validation in international cohorts is essential before clinical implementation. Fourth, molecular biomarkers that may further refine prognostication, such as tumor mutational burden or gene expression signatures, were unavailable [22,40]. Fifth, the LDH threshold was based on absolute values rather than upper limit of normal ratios, which may affect applicability across institutions with different reference ranges.

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
