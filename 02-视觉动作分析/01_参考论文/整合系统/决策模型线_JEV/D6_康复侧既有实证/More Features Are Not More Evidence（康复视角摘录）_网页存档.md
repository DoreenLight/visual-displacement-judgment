# [2609.36154] More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev

> **这是网页快照，不是同行评审论文。**
> 来源：<https://arxiv.org/abs/2609.36154>
> 抓取时间：2026-10-01 20:37
> HTTP 状态：200　｜　原始字节：43,241　｜　SHA256：`fc0999f7b05c8b51`
> **用途**：本条在语料四中的证据等级为「第三方实测」。引用时必须标注为网页型证据，
> 不得与同行评审论文并列。抓取日期之后页面可能变化，引用前请核对原链接。

**页面描述**：Abstract page for arXiv paper 2609.36154: More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev


---

[2609.36154] More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev 
 
 - 
 - 
 - 
 - 
 - 

 - 
 - 
 - 
 - 

 - 

 - - 

 Skip to main content 

 Search arXiv 

 Press Enter to search · Advanced search 

# Computer Science > Artificial Intelligence

 arXiv:2609.36154 (cs)

- 

 [Submitted on 28 Sep 2026]

# Title: More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev

 Authors: Orhan Konak 
 View a PDF of the paper titled More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev, by Orhan Konak

 View PDF 
 HTML (experimental) 

 Abstract: General-purpose models promise sensor-based decisions without training a task-specific classifier, which could reduce the dependence of Human Activity Recognition (HAR) on labeled data. Yet it remains unclear whether such models can directly interpret deterministic descriptions of physical sensor signals well enough to replace or complement trained HAR models. We study this question using Jev, a fixed general-purpose probabilistic decision model, on 1,800 class-balanced accelerometer windows from WISDM, UCI341, and PAMAP2. Jev receives no labeled examples, retrieval context, or HAR-specific parameter updates. We evaluate three deterministic sensor representations and compare 5,400 Jev decisions with a generative baseline and three supervised HAR models. Jev remains far below supervised recognition, with its strongest representation reaching macro-F1 of 0.038, 0.118, and 0.089 across the three datasets, compared with 0.686 to 0.907 for the supervised models. More numerical features do not improve Jev. Instead, they reduce recognition on all three datasets, while augmenting the same numerical evidence with a deterministic semantic rendering partially recovers performance, although the experiment does not isolate semantics from the accompanying serialization and redundancy changes. Jev is fast and inexpensive to query, but its probabilities are not reliably calibrated for recognition. A post-hoc fusion analysis finds a small improvement on WISDM that does not replicate on UCI341 or PAMAP2. These results show that training-free sensor decisions depend not only on the information available in the signal, but also on whether the model can use the representation through which that information is exposed. The sensor-to-model interface should therefore be treated as part of the model evaluation rather than as a neutral preprocessing step.

 Comments: 
 15 pages, 4 figures, 8 tables 

 Subjects: 
 
 Artificial Intelligence (cs.AI) ; Machine Learning (cs.LG) 

 Cite as: 
 arXiv:2609.36154 [cs.AI] 

 (or 
 arXiv:2609.36154v1 [cs.AI] for this version)

 https://doi.org/10.48550/arXiv.2609.36154 

 Focus to learn more 

 arXiv-issued DOI via DataCite (pending registration)

## Submission history
 From: Orhan Konak [ view email ] 
 [v1] 
 Mon, 28 Sep 2026 19:26:05 UTC (199 KB)

 Full-text links: 
 
## Access Paper:

View a PDF of the paper titled More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev, by Orhan Konak
- View PDF 
- HTML (experimental) 
- TeX Source

 view license 

### Current browse context:

 cs.AI

 < prev 
 
 | 
 next > 

 new 
 | 
 recent 
 | 2026-09 

 Change to browse by:
 
 cs 
 cs.LG 

### References & Citations

 - NASA ADS 
- Google Scholar 

 - Semantic Scholar 

 export BibTeX citation 
 Loading... 

## BibTeX formatted citation

 × 

 loading... 

 Data provided by: 

### Bookmark

 Bibliographic Tools 

# Bibliographic and Citation Tools

 Bibliographic Explorer Toggle 

 Bibliographic Explorer ( What is the Explorer? ) 

 Connected Papers Toggle 

 Connected Papers ( What is Connected Papers? ) 

 Litmaps Toggle 

 Litmaps ( What is Litmaps? ) 

 scite.ai Toggle 

 scite Smart Citations ( What are Smart Citations? ) 

 Code, Data, Media 

# Code, Data and Media Associated with this Article

 alphaXiv Toggle 

 alphaXiv ( What is alphaXiv? ) 

 Links to Code Toggle 

 CatalyzeX Code Finder for Papers ( What is CatalyzeX? ) 

 DagsHub Toggle 

 DagsHub ( What is DagsHub? ) 

 GotitPub Toggle 

 Gotit.pub ( What is GotitPub? ) 

 Huggingface Toggle 

 Hugging Face ( What is Huggingface? ) 

 ScienceCast Toggle 

 ScienceCast ( What is ScienceCast? ) 

 Demos 

# Demos

 Replicate Toggle 

 Replicate ( What is Replicate? ) 

 Spaces Toggle 

 Hugging Face Spaces ( What is Spaces? ) 

 Spaces Toggle 

 TXYZ.AI ( What is TXYZ.AI? ) 

 Related Papers 

# Recommenders and Search Tools

 Link to Influence Flower 

 Influence Flower ( What are Influence Flowers? ) 

 Core recommender toggle 

 CORE Recommender ( What is CORE? ) 

 - Author 

 - Venue 

 - Institution 

 - Topic 

 About arXivLabs

# arXivLabs: experimental projects with community collaborators

 arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

 Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

 Have an idea for a project that will add value for arXiv's community? Learn more about arXivLabs .

 Which authors of this paper are endorsers? |
 Disable MathJax ( What is MathJax? )