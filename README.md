# Dataset  

---

## A.1 Dataset Overview

We release a street-view-based walkability dataset constructed from Google Street View (GSV) imagery collected in Seoul, South Korea. The released dataset includes urban expert-validated walkability labels to support research on urban perception and walkability prediction.

### Released Dataset Components

| Component | Description |
|---|---|
| Street-View Images | Google Street View static images collected in Seoul |
| Walkability Labels | Urban expert-validated walkability scores |
| Geographic Coverage | Seoul, South Korea |
| Total Images | 3,410 |
| Dataset Access | [link](https://drive.google.com/file/d/1gABWyi01Ci0p559x-fGU6amQNLLqaOrE/view?usp=sharing)  |
| Metadata Access | [link](https://drive.google.com/drive/folders/1jX8LKIZ_0Lh-JEQlu2A8MyXcFhro0R4b?usp=sharing) |

## A.2 Dataset Construction
In this section, we describe the dataset construction pipeline used to build the proposed street-view-based walkability dataset. The pipeline consists of three stages: (1) GSV image collection and preprocessing, (2) filtering non-walkable areas, and (3) urban expert validation for label consistency.

### Survey-based Walkability Collection
To obtain walkability labels, we conducted an online survey with 2,510 residents of Seoul, South Korea. Participants rated the walkability of their residential areas using a 7-point Likert scale, reflecting perceived safety, comfort, and aesthetic quality. Detailed demographic statistics are summarized in Table 1.

**Age**

| | Number | % |
|---|---:|---:|
| 20s | 189 | 7.53 |
| 30s | 656 | 26.14 |
| 40s | 766 | 30.52 |
| 50s | 676 | 26.93 |
| 60s+ | 223 | 8.88 |
| **Total** | **2,510** | **100.00** |

**Length of Residence (years)**

| | Number | % |
|---|---:|---:|
| 0–5 | 690 | 27.49 |
| 5–10 | 605 | 24.10 |
| 10–15 | 476 | 18.96 |
| 15–20 | 269 | 10.72 |
| 20–25 | 202 | 8.05 |
| 25+ | 268 | 10.68 |
| **Total** | **2,510** | **100.00** |

*Table 1: Demographic distribution by age and years of residence*

---

###  GSV Image Collection and Preprocessing
We collect street-view imagery using Google Street View (GSV). Since panoramic images often contain geometric distortions near image boundaries, we extract four directional static views (front, right, back, and left) to better capture pedestrian-level perspectives.

<p align="center">
  <img src="figs/dataset.png" width="100%" alt="Comparison of panoramic image and static image"/>
</p>

*Figure 1: Comparison of panoramic image and static image*

### Filtering Non-Walkable Areas
Because GSV images are originally collected along vehicular roads, not all locations correspond to pedestrian-accessible environments. To ensure pedestrian relevance, we retain only GSV points spatially aligned with sidewalks and walkable areas.

Figure 2 illustrates the spatial distribution of GSV points along roads and sidewalks.

<p align="center">
  <img src="figs/ICDE_Spatial Distribution.png" width="65%" alt="Spatial distribution of GSV points"/>
</p>

*Figure 2: Spatial distribution of GSV points*

---

### Urban Expert Validation
A conceptual mismatch may exist between area-level survey scores and spot-level GSV imagery. To validate label consistency, we introduce an urban expert annotation protocol based on micro-scale pedestrian visual attributes. Figure 4(a) shows a case where an area receives a high walkability score, while the corresponding GSV image does not visually convey that level of walkability. Conversely, Figure 4(b) shows the opposite case — an area with a low walkability survey score whose GSV image appears highly walkable.

Following prior works highlighting that pedestrian experience depends on micro-scale attributes (e.g., sidewalk presence, maintenance, street aesthetics), each image was evaluated along two main criteria:

- **Negative visual elements**: presence of damaged or unpaved roads, visible cracks or aesthetically disruptive features such as trash, which correspond to lower walkability.
- **Positive visual elements**: presence of greenery, well-maintained sidewalks, and aesthetically appealing facades or streets, which indicate higher walkability.

<p align="center">
  <img src="figs/surve_annot_mismatch.png" width="80%" alt="Mismatch between survey-perceived walkability and expert annotation"/>
</p>

*Figure 4: Example of mismatch between survey-perceived walkability score and expert annotation*

#### (a) Urban Expert Annotation Guideline

| Features | Negative Elements | Positive Elements |
|---|:---:|:---:|
| Pavement | −1 | +1 |
| Road crack | −1 | — |
| Trash | −1 | — |
| Curbs | −1 | +1 |
| Parking | −1 (e.g., illegal parking) | +1 (e.g., well parked) |
| Road marking | −1 (e.g., faded marking) | +1 |
| Greenery | −1 (e.g., withering greenery) | +1 |
| Building facades | −1 (e.g., cracked, graffiti) | +1 |
| Sidewalk | −1 (e.g., no sidewalk) | +1 |
| Pole | −1 (e.g., messy cables) | — |

#### (b) Urban Expert-Annotated Walkability Score by Feature Score Range

| Walkability Score | Feature Score Range |
|:---:|---|
| 1 | $-10 \leq \text{score} < -6$ |
| 2 | $-6 \leq \text{score} < -3$ |
| 3 | $-3 \leq \text{score} < -1$ |
| 4 | $-1 \leq \text{score} < 1$ |
| 5 | $1 \leq \text{score} < 3$ |
| 6 | $3 \leq \text{score} < 5$ |
| 7 | $5 \leq \text{score} < 7$ |

*Table 3: Urban Expert Annotation Guideline and Scoring Criteria*

### Final Dataset Statistics
Following urban expert validation, we analyze the distribution of feature scores using kernel density estimation (KDE), as illustrated in Figure 3. Adjacent score categories exhibit highly overlapping distributions, motivating the consolidation of the original 7-point scale into a 4-class walkability benchmark.

<p align="center">
  <img src="figs/kde.png" width="65%" alt="KDE plot of Feature Score Distribution"/>
</p>

*Figure 3: KDE plot of Feature Score Distribution*


| Label  | 1   | 2   | 3   | 4   | Total |
|--------|-----|-----|-----|-----|-------|
| Images | 815 | 934 | 954 | 707 | 3410  |

*Table 2: Class Distribution of the Final Dataset*
