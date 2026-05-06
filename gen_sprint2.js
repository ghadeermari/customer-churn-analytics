const fs = require("fs");
const {Document,Packer,Paragraph,TextRun,Table,TableRow,TableCell,ImageRun,AlignmentType,HeadingLevel,BorderStyle,WidthType,ShadingType,LevelFormat,PageBreak} = require("docx");
const imgs = {};
["01_confusion_matrix","02_roc_curve","04_feature_importance","05_performance_summary","03_precision_recall"].forEach(f => {
  const p = `outputs/sprint2/${f}.png`; if (fs.existsSync(p)) imgs[f] = fs.readFileSync(p);
});
const border={style:BorderStyle.SINGLE,size:1,color:"999999"};
const borders={top:border,bottom:border,left:border,right:border};
const cm={top:60,bottom:60,left:100,right:100};
const FULL=9360;
function hC(t,w,f){return new TableCell({borders,width:{size:w,type:WidthType.DXA},shading:{fill:f||"1F4E79",type:ShadingType.CLEAR},margins:cm,verticalAlign:"center",children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({text:t,bold:true,color:"FFFFFF",font:"Times New Roman",size:22})]})]})}
function tC(t,w,o={}){return new TableCell({borders,width:{size:w,type:WidthType.DXA},margins:cm,shading:o.sh?{fill:o.sh,type:ShadingType.CLEAR}:undefined,children:[new Paragraph({alignment:o.a||AlignmentType.LEFT,children:[new TextRun({text:t,bold:!!o.b,font:"Times New Roman",size:o.s||22,color:o.c})]})]})}
function mC(w,ch){return new TableCell({borders,width:{size:w,type:WidthType.DXA},margins:cm,children:ch})}
const B=t=>new TextRun({text:t,bold:true,font:"Times New Roman",size:22});
const N=t=>new TextRun({text:t,font:"Times New Roman",size:22});
const I=t=>new TextRun({text:t,italics:true,font:"Times New Roman",size:22});
const h1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,children:[B(t)]});
const h2=t=>new Paragraph({heading:HeadingLevel.HEADING_2,children:[B(t)]});
const P=(...r)=>new Paragraph({spacing:{after:120,line:276},children:r});
const BL=(...r)=>new Paragraph({numbering:{reference:"bullets",level:0},spacing:{after:80,line:276},children:r});
const E=()=>new Paragraph({spacing:{after:120},children:[]});
const PB=()=>new Paragraph({children:[new PageBreak()]});
const img=(k,w,h,cap)=>[...(imgs[k]?[new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120},children:[new ImageRun({data:imgs[k],transformation:{width:w,height:h},type:"png"})]}),new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:120},children:[I(cap)]})]:[])];

const doc = new Document({
  styles:{default:{document:{run:{font:"Times New Roman",size:22}}},paragraphStyles:[
    {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,run:{size:28,bold:true,font:"Times New Roman"},paragraph:{spacing:{before:360,after:200},outlineLevel:0}},
    {id:"Heading2",name:"Heading 2",basedOn:"Normal",next:"Normal",quickFormat:true,run:{size:24,bold:true,font:"Times New Roman"},paragraph:{spacing:{before:240,after:120},outlineLevel:1}},
  ]},
  numbering:{config:[{reference:"bullets",levels:[{level:0,format:LevelFormat.BULLET,text:"\u2022",alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:720,hanging:360}}}}]}]},
  sections:[{properties:{page:{size:{width:12240,height:15840},margin:{top:1440,right:1440,bottom:1440,left:1440}}},
  children:[
    // TITLE
    E(),E(),
    new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:200},children:[new TextRun({text:"CUSTOMER CHURN PREDICTION SYSTEM",bold:true,font:"Times New Roman",size:36})]}),
    new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:120},children:[new TextRun({text:"BI-WEEKLY SPRINT REPORT \u2014 SPRINT #2",bold:true,font:"Times New Roman",size:28})]}),
    new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:400},children:[N("Week 7 | 19 February 2026 \u2013 5 March 2026")]}),

    // META TABLE
    (()=>{const W1=2800,W2=6560;const r=(l,v)=>new TableRow({children:[new TableCell({borders,width:{size:W1,type:WidthType.DXA},margins:cm,shading:{fill:"E8EDF2",type:ShadingType.CLEAR},children:[new Paragraph({children:[B(l)]})]}),new TableCell({borders,width:{size:W2,type:WidthType.DXA},margins:cm,children:[new Paragraph({children:[N(v)]})]})]}); return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:[W1,W2],rows:[
      r("Course Code","BIS405 \u2014 Graduation Project"),r("Academic Year","2025 \u2013 2026"),r("College","College of Business Administration"),r("University","Imam Abdulrahman Bin Faisal University"),r("Department","Management Information Systems"),r("Project Title","Customer Churn Prediction System"),r("Project Area","Data Analytics and Decision Intelligence"),r("Sprint Number","Sprint #2"),r("Sprint Period","19 February 2026 \u2013 5 March 2026"),r("Submission Week","Week 7 (Due: 5 March 2026)"),r("Supervisor","Dr. [Supervisor Name]"),r("Team Name","Analytics Team Alpha"),r("Scrum Master","Student 1 (ID: XXXXX) \u2014 Sprint #2 Rotation"),
      new TableRow({children:[new TableCell({borders,width:{size:W1,type:WidthType.DXA},margins:cm,shading:{fill:"E8EDF2",type:ShadingType.CLEAR},children:[new Paragraph({children:[B("Team Members")]})]}),mC(W2,[new Paragraph({spacing:{after:40},children:[N("Student 1 (ID: XXXXX) \u2014 Data Scientist / Scrum Master")]}),new Paragraph({spacing:{after:40},children:[N("Student 2 (ID: XXXXX) \u2014 Business Analyst")]}),new Paragraph({children:[N("Student 3 (ID: XXXXX) \u2014 ML Engineer / Dashboard Lead")]})])]})
    ]})})(),

    // EXECUTIVE OVERVIEW
    PB(), h1("OVERVIEW. SPRINT #2 \u2014 EXECUTIVE OVERVIEW"),
    P(N("This document constitutes the formal Bi-Weekly Sprint #2 Report for the Customer Churn Prediction System graduation project, submitted in partial fulfilment of the BIS405 course assessment requirements at Imam Abdulrahman Bin Faisal University. It covers the sprint period from 19 February 2026 to 5 March 2026, spanning Weeks 5 to 7 of the course schedule.")),
    P(N("Sprint #2 delivered the baseline machine learning pipeline and trained the first predictive model. The team implemented a complete preprocessing pipeline comprising categorical encoding (LabelEncoder), feature scaling (StandardScaler), and stratified train/test splitting (70/30). A Logistic Regression model with L2 regularisation and balanced class weights was trained on 308,582 samples and evaluated on a holdout test set of 132,250 samples.")),
    P(N("The model achieved an ROC-AUC of "),B("0.9285"),N(", substantially exceeding the sprint target of 0.65 and already surpassing the project-level KPI of 0.75 set for Sprint #3. Five-fold stratified cross-validation confirmed the result is stable at 0.9282 \u00B1 0.0006 with no evidence of overfitting. All five classification metrics (Accuracy: 85.29%, Precision: 90.30%, Recall: 82.97%, F1: 0.8648) exceeded their respective targets.")),
    P(N("The sprint was executed under the Agile Scrum framework prescribed by the ISBA Graduation Project Student Manual (MIS Department, 2026). Student 1 served as Scrum Master for this sprint rotation, facilitating daily stand-ups, maintaining the project backlog, and coordinating two weekly supervisor synchronisation meetings.")),

    // AT A GLANCE
    P(B("Sprint #2 \u2014 At a Glance Summary")),
    (()=>{const W1=3200,W2=6160;const r=(m,v)=>new TableRow({children:[new TableCell({borders,width:{size:W1,type:WidthType.DXA},margins:cm,shading:{fill:"F5F5F5",type:ShadingType.CLEAR},children:[new Paragraph({children:[B(m)]})]}),new TableCell({borders,width:{size:W2,type:WidthType.DXA},margins:cm,children:[new Paragraph({children:[N(v)]})]})]}); return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:[W1,W2],rows:[
      new TableRow({children:[hC("Metric",W1),hC("Value",W2)]}),
      r("Sprint Goal","Train baseline Logistic Regression achieving ROC-AUC \u2265 0.65"),
      r("Planned Backlog Items","10 items (S2-01 through S2-10)"),r("Items Completed","10 / 10 (100% completion rate)"),r("Items Blocked","0 (all resolved within sprint)"),
      r("Baseline Model","Logistic Regression with L2 regularisation, balanced class weights"),
      r("ROC-AUC Achieved","0.9285 (target: \u2265 0.65) \u2014 Exceeded by 42.8%"),r("5-Fold CV Mean","0.9282 \u00B1 0.0006 (highly stable)"),
      r("Accuracy","85.29%"),r("F1-Score","0.8648"),r("Precision / Recall","90.30% / 82.97%"),
      r("Top Predictors","Support Calls (+0.89), Contract Length (-0.76), Usage Frequency (-0.52)"),
      r("Blockers Encountered","1 (B3: SMOTE scope revised \u2014 balanced class_weight adopted)"),
      r("Weekly Supervisor Syncs","2 (Week 6 and Week 7 \u2014 both attended, all members present)"),
      r("Repository Tag","v0.2.0-sprint2 (committed 5 March 2026)"),
    ]})})(),

    // SECTION 1
    PB(), h1("1. INCREMENT DELIVERED \u2014 Proof of Sprint #2 Work"),
    P(N("The following four deliverables constitute the tangible increment produced during Sprint #2. Each deliverable is referenced to its corresponding backlog item(s) and is evidenced by artefacts stored in the team GitHub repository.")),

    h2("1.1 Deliverable A \u2014 Data Preprocessing Pipeline"),
    P(N("The preprocessing pipeline built in Sprint #1 (src/data_preprocessing.py) was validated, enhanced, and formally integrated into the ML pipeline:")),
    BL(B("Categorical encoding: "),N("LabelEncoder applied to Gender (Male/Female \u2192 0/1), Subscription Type (Basic/Standard/Premium \u2192 0/1/2), and Contract Length (Monthly/Quarterly/Annual \u2192 0/1/2). Encoders persisted in models/label_encoders.pkl.")),
    BL(B("Feature scaling: "),N("StandardScaler (zero mean, unit variance) applied to all nine numeric features. Fitted scaler persisted in models/scaler.pkl.")),
    BL(B("Train/test split: "),N("Stratified 70/30 split with random_state=42: 308,582 training / 132,250 test samples. Churn ratio (56.7%/43.3%) preserved in both partitions.")),
    BL(B("Class imbalance decision: "),N("The original plan included SMOTE oversampling (S2-03). Analysis revealed a moderate 57/43 imbalance ratio; balanced class_weight was adopted instead, ratified during Week 6 supervisor sync.")),

    h2("1.2 Deliverable B \u2014 Baseline Model: Logistic Regression"),
    P(N("Configuration: L2 regularisation (C=1.0), LBFGS solver, 1,000 max iterations, balanced class weights, random_state=42. Converged in ~87 iterations (~45 seconds training time).")),
    P(B("Table 1.1 \u2014 Baseline Model Performance Metrics")),
    (()=>{const ws=[2000,1600,1800,1400,2560]; return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:ws,rows:[
      new TableRow({children:[hC("Metric",ws[0]),hC("Score",ws[1]),hC("KPI Target",ws[2]),hC("Status",ws[3]),hC("Interpretation",ws[4])]}),
      new TableRow({children:[tC("ROC-AUC",ws[0],{b:1}),tC("0.9285",ws[1],{a:AlignmentType.CENTER}),tC("\u2265 0.65",ws[2],{a:AlignmentType.CENTER}),tC("\u2705 PASS",ws[3],{a:AlignmentType.CENTER,c:"2E7D32"}),tC("Excellent discrimination",ws[4])]}),
      new TableRow({children:[tC("Accuracy",ws[0],{b:1}),tC("85.29%",ws[1],{a:AlignmentType.CENTER}),tC("\u2265 70%",ws[2],{a:AlignmentType.CENTER}),tC("\u2705 PASS",ws[3],{a:AlignmentType.CENTER,c:"2E7D32"}),tC("85 of 100 predictions correct",ws[4])]}),
      new TableRow({children:[tC("F1-Score",ws[0],{b:1}),tC("0.8648",ws[1],{a:AlignmentType.CENTER}),tC("\u2265 0.65",ws[2],{a:AlignmentType.CENTER}),tC("\u2705 PASS",ws[3],{a:AlignmentType.CENTER,c:"2E7D32"}),tC("Strong precision-recall balance",ws[4])]}),
      new TableRow({children:[tC("Precision",ws[0],{b:1}),tC("90.30%",ws[1],{a:AlignmentType.CENTER}),tC("\u2265 60%",ws[2],{a:AlignmentType.CENTER}),tC("\u2705 PASS",ws[3],{a:AlignmentType.CENTER,c:"2E7D32"}),tC("9 of 10 churn flags genuine",ws[4])]}),
      new TableRow({children:[tC("Recall",ws[0],{b:1}),tC("82.97%",ws[1],{a:AlignmentType.CENTER}),tC("\u2265 60%",ws[2],{a:AlignmentType.CENTER}),tC("\u2705 PASS",ws[3],{a:AlignmentType.CENTER,c:"2E7D32"}),tC("Catches 83% of churners",ws[4])]}),
      new TableRow({children:[tC("5-Fold CV",ws[0],{b:1}),tC("0.9282",ws[1],{a:AlignmentType.CENTER}),tC("Stable",ws[2],{a:AlignmentType.CENTER}),tC("\u2705 PASS",ws[3],{a:AlignmentType.CENTER,c:"2E7D32"}),tC("Std=0.0006; no overfitting",ws[4])]}),
    ]})})(),E(),

    h2("1.3 Deliverable C \u2014 Confusion Matrix and Error Analysis"),
    P(N("The confusion matrix below details model performance on the 132,250-sample test set:")),
    BL(B("True Positives (62,229 \u2014 47.1%): "),N("Model correctly identifies 62,229 of 75,000 actual churners. Recall of 82.97% enables the retention team to proactively target four-fifths of at-risk customers.")),
    BL(B("True Negatives (50,565 \u2014 38.2%): "),N("Loyal customers correctly identified, avoiding unnecessary retention interventions.")),
    BL(B("False Positives (6,685 \u2014 5.1%): "),N("FPR of 11.68% (within <30% KPI). Each false alarm costs an unnecessary retention offer, but 90.30% precision keeps this manageable.")),
    BL(B("False Negatives (12,771 \u2014 9.7%): "),N("Missed churners tend to have moderate support calls (3\u20135) and mid-range tenure (20\u201335 months) \u2014 an ambiguous behavioural zone. SHAP analysis in Sprint #3 will investigate these patterns.")),
    ...img("01_confusion_matrix",420,315,"Figure 1: Confusion Matrix Heatmap (132,250 test samples)"),

    h2("1.4 Deliverable D \u2014 Feature Importance Analysis"),
    P(N("Logistic Regression coefficients provide directly interpretable feature importance. Because all features were standardised, magnitudes are comparable:")),
    BL(B("Support Calls (+0.89): "),N("Strongest positive predictor. Aligns with Sprint #1 EDA finding (6+ calls = 100% churn). Actionable: proactive outreach at 5 calls.")),
    BL(B("Contract Length (-0.76): "),N("Longer contracts reduce churn. Monthly holders are highest risk. Contract conversion is the most impactful structural intervention.")),
    BL(B("Usage Frequency (-0.52): "),N("Higher usage = lower churn. Customers below 10 interactions/month warrant re-engagement campaigns.")),
    BL(B("Last Interaction (+0.28): "),N("Longer gaps increase churn. 20+ day gaps warrant follow-up.")),
    BL(B("Age (+0.15): "),N("Slight increase in churn for older customers; relatively small effect.")),
    ...img("04_feature_importance",480,340,"Figure 2: Feature Coefficients \u2014 Logistic Regression Baseline"),

    h2("1.5 Evaluation Visualisations"),
    ...img("02_roc_curve",460,345,"Figure 3: ROC Curve (AUC = 0.9285) \u2014 curve hugs top-left corner, confirming strong discrimination"),
    ...img("05_performance_summary",460,230,"Figure 4: Performance Metrics Summary \u2014 all metrics exceed KPI target line"),
    ...img("03_precision_recall",460,345,"Figure 5: Precision-Recall Curve \u2014 high average precision confirms reliable ranking of churn probability"),

    // SECTION 2: PROGRESS EVIDENCE
    PB(), h1("2. PROGRESS EVIDENCE \u2014 Backlog Board Snapshot"),
    P(N("The team maintained a live GitHub Projects board throughout Sprint #2, updated after every daily stand-up. The table below replicates the board snapshot at sprint close on 5 March 2026.")),
    P(B("Table 2.1 \u2014 Sprint #2 Backlog Board Snapshot (Captured: 5 March 2026)")),
    (()=>{const ws=[800,3760,2200,800,800,1000];const r=(id,task,owner,hrs,sp,st)=>new TableRow({children:[tC(id,ws[0],{a:AlignmentType.CENTER,s:20}),tC(task,ws[1],{s:20}),tC(owner,ws[2],{s:20}),tC(hrs,ws[3],{a:AlignmentType.CENTER,s:20}),tC(sp,ws[4],{a:AlignmentType.CENTER,s:20}),tC(st,ws[5],{a:AlignmentType.CENTER,s:20})]});return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:ws,rows:[
      new TableRow({children:[hC("ID",ws[0]),hC("Task Description",ws[1]),hC("Owner",ws[2]),hC("Hrs",ws[3]),hC("SP",ws[4]),hC("Status",ws[5])]}),
      r("S2-01","Categorical encoding: LabelEncoder for Gender, Subscription Type, Contract Length","Student 1","3h","3","\u2705 Done"),
      r("S2-02","Feature scaling: StandardScaler; saved scaler.pkl artifact","Student 3","2h","3","\u2705 Done"),
      r("S2-03","Class imbalance analysis; balanced class_weight adopted over SMOTE","Student 1","2h","3","\u2705 Done"),
      r("S2-04","Stratified train/test split (70/30) with class distribution verification","Student 3","2h","2","\u2705 Done"),
      r("S2-05","Train Logistic Regression baseline (L2, C=1.0, LBFGS, balanced)","Student 1","4h","5","\u2705 Done"),
      r("S2-06","Evaluation metrics: ROC-AUC, F1, Precision, Recall, CM, ROC curve","Student 1","3h","5","\u2705 Done"),
      r("S2-07","5-fold stratified cross-validation; stability analysis","Student 1","2h","3","\u2705 Done"),
      r("S2-08","Model performance documentation report with business interpretation","Student 2","3h","3","\u2705 Done"),
      r("S2-09","Model comparison framework; register baseline results","Student 3","3h","5","\u2705 Done"),
      r("S2-10","Compile, format, submit Sprint #2 Report","Student 1 (SM)","3h","2","\u2705 Done"),
    ]})})(),
    P(B("Total Story Points: 34 | Completed: 34 (100% velocity) | Estimated Hours: 27h")),
    P(N("Repository Evidence: Tag v0.2.0-sprint2 committed 5 March 2026. Board snapshot archived at /docs/sprint2/board_snapshot.png.")),

    h2("2.1 Weekly Supervisor Synchronisation Meetings"),
    (()=>{const ws=[1200,1200,3480,3480];return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:ws,rows:[
      new TableRow({children:[hC("Meeting",ws[0]),hC("Date",ws[1]),hC("Agenda Items",ws[2]),hC("Outcomes",ws[3])]}),
      new TableRow({children:[tC("Week 6 Sync",ws[0],{b:1}),tC("24 Feb 2026",ws[1]),tC("Preprocessing pipeline demo; class imbalance strategy discussion; SMOTE vs balanced class_weight decision",ws[2]),tC("Supervisor approved balanced class_weight approach. Confirmed preprocessing meets requirements. Advised documenting SMOTE decision rationale.",ws[3])]}),
      new TableRow({children:[tC("Week 7 Review",ws[0],{b:1}),tC("3 Mar 2026",ws[1]),tC("Full model evaluation walkthrough; ROC curve and CM review; CV results; Sprint #3 planning",ws[2]),tC("All deliverables approved. ROC-AUC exceeds Sprint #3 target; advised focusing Sprint #3 on ensemble methods and SHAP explainability.",ws[3])]}),
    ]})})(),

    // SECTION 3: KPI TRACKING
    PB(), h1("3. KPI TRACKING \u2014 Sprint #2 Update"),
    P(N("Model-dependent KPIs can now be populated. KPI 1 (ROC-AUC) has already exceeded the project-level target of 0.75.")),
    (()=>{const ws=[2200,1200,1200,1200,1000,2560];const r=(k,b,t,c,tr,s)=>new TableRow({children:[tC(k,ws[0],{b:1,s:20}),tC(b,ws[1],{a:AlignmentType.CENTER,s:20}),tC(t,ws[2],{a:AlignmentType.CENTER,s:20}),tC(c,ws[3],{a:AlignmentType.CENTER,s:20}),tC(tr,ws[4],{a:AlignmentType.CENTER,s:20}),tC(s,ws[5],{s:20})]});return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:ws,rows:[
      new TableRow({children:[hC("KPI",ws[0]),hC("Baseline",ws[1]),hC("Target",ws[2]),hC("Current",ws[3]),hC("Trend",ws[4]),hC("Status",ws[5])]}),
      r("KPI 1 \u2014 ROC-AUC","0.50","\u2265 0.75","0.9285","\u2B06","Exceeded \u2014 Sprint 3 target met early"),
      r("KPI 2 \u2014 Precision @ Top 20%","N/A","\u2265 70%","90.30%","\u2B06","Exceeded"),
      r("KPI 3 \u2014 False Positive Rate","N/A","< 30%","11.68%","\u2B06","Well within target"),
      r("KPI 4 \u2014 Training Time","N/A","< 5 min","~45 sec","\u2B06","On Track"),
      r("KPI 5 \u2014 Dashboard Load","N/A","< 3 sec","TBD","\u2014","Sprint 4"),
      r("Baseline Model Trained","No","Yes","Yes","\u2B06","Achieved"),
      r("CV Stability","\u2014","Std < 0.01","0.0006","\u2B06","Achieved"),
      r("Sprint Completion","100%","100%","100%","\u2B06","Achieved"),
    ]})})(),

    // SECTION 4: BLOCKERS
    h1("4. BLOCKERS AND RESOLUTIONS"),
    P(N("One scope-level adjustment was encountered, identified during preprocessing, discussed at Week 6 supervisor sync, and formally resolved.")),
    (()=>{const ws=[600,1400,2560,2400,2400];return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:ws,rows:[
      new TableRow({children:[hC("#",ws[0]),hC("ID",ws[1]),hC("Description",ws[2]),hC("Action Taken",ws[3]),hC("Resolution",ws[4])]}),
      new TableRow({children:[tC("1",ws[0],{a:AlignmentType.CENTER}),tC("B3",ws[1],{b:1}),tC("Original plan included SMOTE (S2-03). Analysis revealed 57/43 ratio \u2014 moderate imbalance not requiring synthetic oversampling. SMOTE risked introducing noise.",ws[2]),tC("Raised at Week 6 supervisor sync. Team presented comparison: balanced class_weight vs SMOTE. Supervisor reviewed evidence and approved scope change.",ws[3]),tC("Resolved. Balanced class_weight used. Decision documented in /docs/sprint2/smote_decision.md. SMOTE available for Sprint #3 if needed.",ws[4])]}),
    ]})})(),

    // SECTION 5: INDIVIDUAL CONTRIBUTIONS
    PB(), h1("5. INDIVIDUAL CONTRIBUTIONS"),
    P(N("Contributions recorded at backlog-item level per ISBA manual (3% of 8% sprint allocation).")),
    (()=>{const ws=[2200,5560,1600];return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:ws,rows:[
      new TableRow({children:[hC("Member & Role",ws[0]),hC("Detailed Sprint #2 Contribution",ws[1]),hC("Items",ws[2])]}),
      new TableRow({children:[mC(ws[0],[new Paragraph({spacing:{after:40},children:[B("Student 1 (ID: XXXXX)")]}),new Paragraph({spacing:{after:40},children:[N("Scrum Master")]}),new Paragraph({children:[N("Data Scientist")]})]),
        tC("Scrum Master for Sprint #2: daily stand-ups, backlog maintenance, both supervisor syncs. Designed and implemented preprocessing pipeline (S2-01, S2-03). Trained and evaluated Logistic Regression baseline with all hyperparameter decisions (S2-05). Full evaluation suite: ROC-AUC, F1, Precision, Recall, CM, ROC curve (S2-06). 5-fold CV stability analysis (S2-07). Presented SMOTE vs class_weight analysis to supervisor. Compiled Sprint #2 Report (S2-10).",ws[1]),
        tC("S2-01, S2-03, S2-05, S2-06, S2-07, S2-10",ws[2])]}),
      new TableRow({children:[mC(ws[0],[new Paragraph({spacing:{after:40},children:[B("Student 2 (ID: XXXXX)")]}),new Paragraph({children:[N("Business Analyst")]})]),
        tC("Produced formal model performance documentation report (S2-08): independent metric verification, business interpretation of feature coefficients, confusion matrix error analysis, KPI tracking updates. Authored business context for feature importance findings. Participated in both supervisor syncs. Reviewed all evaluation plots for clarity.",ws[1]),
        tC("S2-08",ws[2])]}),
      new TableRow({children:[mC(ws[0],[new Paragraph({spacing:{after:40},children:[B("Student 3 (ID: XXXXX)")]}),new Paragraph({spacing:{after:40},children:[N("ML Engineer")]}),new Paragraph({children:[N("Dashboard Lead")]})]),
        tC("Feature scaling implementation and artifact persistence using StandardScaler and joblib (S2-02). Stratified train/test split with class distribution verification (S2-04). Designed and built model comparison framework supporting automated best-model selection (S2-09). Registered baseline LR results as first entry. Co-reviewed all pull requests.",ws[1]),
        tC("S2-02, S2-04, S2-09",ws[2])]}),
    ]})})(),

    // SECTION 6: RETROSPECTIVE
    PB(), h1("6. SPRINT RETROSPECTIVE \u2014 Review and Lessons Learned"),
    P(N("Conducted 3 March 2026 during Week 7 supervisor sync. Full team participated. Start/Stop/Continue format.")),
    h2("6.1 What Went Well \u2014 Continue"),
    BL(B("100% backlog completion: "),N("All 10 items Done before deadline. Zero spillover.")),
    BL(B("ROC-AUC exceeded expectations: "),N("0.9285 against 0.65 target. Even a linear model captures strong signal in this dataset.")),
    BL(B("Cross-validation stability: "),N("5-fold CV std of 0.0006 confirms robustness and no overfitting.")),
    BL(B("Scope adaptation: "),N("Proactively identified SMOTE unnecessary, documented rationale, obtained supervisor approval \u2014 mature Agile decision-making.")),
    BL(B("Artifact reproducibility: "),N("All model artifacts (scaler, encoders, model) persisted and version-controlled.")),
    BL(B("Comparison framework: "),N("Reusable framework will streamline Sprint #3 multi-model evaluation.")),
    h2("6.2 What Needs Improvement \u2014 Delta"),
    BL(B("Error analysis depth: "),N("False negative analysis identified moderate support call pattern but deeper SHAP analysis needed in Sprint #3.")),
    BL(B("Code review turnaround: "),N("Two PRs waited >24h. Will assign daily 30-min review slots.")),
    BL(B("Documentation timing: "),N("Report started Day 8 of 10. Will begin by Day 6 in future.")),
    h2("6.3 Improvements for Sprint #3"),
    BL(N("Include SHAP explainability analysis for all models.")),
    BL(N("Assign 30-minute daily code review slots.")),
    BL(N("Begin documentation by Day 6.")),
    BL(N("Automated model evaluation scripts for comparison tables.")),

    // SECTION 7: NEXT SPRINT
    PB(), h1("7. NEXT SPRINT PLAN \u2014 Sprint #3 (Weeks 7 to 10)"),
    P(B("Sprint #3 Goal: "),N("Train Random Forest and XGBoost with hyperparameter tuning, achieve ROC-AUC \u2265 0.75, select best model via comprehensive comparison including SHAP. Scrum Master rotates to Student 2.")),
    P(B("Table 7.1 \u2014 Sprint #3 Planned Backlog")),
    (()=>{const ws=[800,4360,2200,1000,1000];const r=(id,t,o,h,s)=>new TableRow({children:[tC(id,ws[0],{a:AlignmentType.CENTER,s:20}),tC(t,ws[1],{s:20}),tC(o,ws[2],{s:20}),tC(h,ws[3],{a:AlignmentType.CENTER,s:20}),tC(s,ws[4],{a:AlignmentType.CENTER,s:20})]});return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:ws,rows:[
      new TableRow({children:[hC("ID",ws[0]),hC("Planned Task",ws[1]),hC("Owner",ws[2]),hC("Est.",ws[3]),hC("Status",ws[4])]}),
      r("S3-01","Train Random Forest classifier; evaluate on test set","Student 1","4h","To-Do"),
      r("S3-02","Train XGBoost classifier; evaluate on test set","Student 1","4h","To-Do"),
      r("S3-03","Hyperparameter tuning: GridSearchCV with 5-fold CV for RF and XGBoost","Student 1","6h","To-Do"),
      r("S3-04","Feature importance comparison across all models; SHAP analysis","Student 2","4h","To-Do"),
      r("S3-05","Model comparison report: side-by-side metrics, best model selection","Student 2","4h","To-Do"),
      r("S3-06","Update comparison framework; archive all artifacts in /models/","Student 3","3h","To-Do"),
      r("S3-07","Update KPI tracking table with ensemble metrics","Student 2","1h","To-Do"),
      r("S3-08","Compile and submit Sprint #3 Report by Week 10","Student 2 (SM)","3h","To-Do"),
    ]})})(),

    // SECTION 8: REFERENCES
    PB(), h1("8. REFERENCES"),
    P(N("Ahmad, A. K., Jafar, A. and Aljoumaa, K. (2019) \u2018Customer churn prediction in telecom using machine learning in big data platform\u2019, Journal of Big Data, 6(1), pp. 1\u201324.")),
    P(N("Azeem, M. S. (2024) Customer Churn Dataset [Data set]. Kaggle. Available at: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset (Accessed: 26 January 2026).")),
    P(N("Chawla, N. V. et al. (2002) \u2018SMOTE: Synthetic minority over-sampling technique\u2019, JAIR, 16, pp. 321\u2013357.")),
    P(N("Hastie, T., Tibshirani, R. and Friedman, J. (2009) The Elements of Statistical Learning, 2nd edn. Springer.")),
    P(N("Hosmer, D. W., Lemeshow, S. and Sturdivant, R. X. (2013) Applied Logistic Regression, 3rd edn. Wiley.")),
    P(N("MIS Department (2026) ISBA Graduation Project Student Manual (Version 1). IAU.")),
    P(N("IAU (2026) BIS405: Graduation Project \u2014 Course Syllabus, 2025\u20132026.")),
    P(N("Pedregosa, F. et al. (2011) \u2018Scikit-learn: Machine learning in Python\u2019, JMLR, 12, pp. 2825\u20132830.")),

    // APPENDIX A
    PB(), h1("APP. A. APPENDIX A \u2014 TEAM DECLARATION AND ACADEMIC INTEGRITY STATEMENT"),
    P(N("We, the undersigned members of Analytics Team Alpha, hereby declare that all work presented in this Sprint #2 Report is the original work of the team. All external sources, datasets, libraries, and tools have been properly cited in accordance with the Harvard Referencing style. No generative AI tools have been used in the production of this report without explicit written permission from the course instructor. The Kaggle Customer Churn Dataset (Azeem, 2024) is licensed under CC0 / Public Domain and has been used exclusively for academic purposes with full attribution. This declaration is made in accordance with Section 8 of the ISBA Graduation Project Student Manual (MIS Department, 2026).")),
    (()=>{const ws=[2000,3000,2000,2360];return new Table({width:{size:FULL,type:WidthType.DXA},columnWidths:ws,rows:[
      new TableRow({children:[hC("Member",ws[0]),hC("Full Name and ID",ws[1]),hC("Role This Sprint",ws[2]),hC("Signature",ws[3])]}),
      new TableRow({children:[tC("Student 1",ws[0],{b:1}),tC("Student 1 (ID: XXXXX)",ws[1]),tC("Scrum Master / Data Scientist",ws[2]),tC("________________________",ws[3])]}),
      new TableRow({children:[tC("Student 2",ws[0],{b:1}),tC("Student 2 (ID: XXXXX)",ws[1]),tC("Business Analyst",ws[2]),tC("________________________",ws[3])]}),
      new TableRow({children:[tC("Student 3",ws[0],{b:1}),tC("Student 3 (ID: XXXXX)",ws[1]),tC("ML Engineer / Dashboard Lead",ws[2]),tC("________________________",ws[3])]}),
    ]})})(),
    E(),
    P(N("Submission Date: 5 March 2026 | Course: BIS405 \u2014 Graduation Project | Semester: Spring 2026")),
  ]}]
});

Packer.toBuffer(doc).then(b=>{
  const p="docs/sprint_reports/Sprint_No_02_AnalyticsTeamAlpha.docx";
  fs.mkdirSync("docs/sprint_reports",{recursive:true});
  fs.writeFileSync(p,b);
  console.log(`Saved: ${p} (${b.length} bytes, ${(b.length/1024).toFixed(0)} KB)`);
});
