# SciPy Statistical Validation Layer

The Lightcast job description mentions SciPy, so this project uses SciPy directly instead of only depending on it through scikit-learn.

## Where SciPy is used

### 1. Statistical validation: `src/scipy_insights.py`

The pipeline creates `reports/statistical_tests.csv` using:

- `scipy.stats.spearmanr` to test whether postings with more extracted skills tend to have higher salary midpoints.
- `scipy.stats.kruskal` to test whether salary distributions differ across role families.
- `scipy.stats.chi2_contingency` to test whether role family and experience level are associated.
- `scipy.stats.mannwhitneyu` to compare salary distributions for postings that mention a specific skill versus postings that do not.

### 2. Role similarity: `src/scipy_insights.py`

The pipeline creates `reports/role_skill_similarity.csv` using:

- `scipy.spatial.distance.cosine` to calculate role-to-role similarity from role-by-skill demand vectors.

This helps explain which roles have overlapping skill requirements. For example, Data Scientist and ML Engineer may share Python, statistics, machine learning, and model evaluation.

### 3. Forecast optimization: `src/forecasting.py`

The skill-demand forecast uses:

- `scipy.optimize.minimize` to fit a transparent regularized forecasting model with lag features.

The model minimizes mean squared error plus a small L2 penalty, which keeps the model stable on small portfolio datasets.

## How to explain in interview

“I used SciPy for the statistical validation layer. After extracting skills and building labor-market analytics, I used SciPy statistical tests to check whether observed patterns were meaningful. For example, I used Spearman correlation to study salary versus skill count, Kruskal-Wallis to compare salary distributions across roles, chi-square testing for role and experience-level association, and cosine distance to find similar roles based on skill-demand profiles. I also used SciPy optimization in the forecasting module to fit a regularized lag-based demand model.”
