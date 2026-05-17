"""
AI/ML Internship Week 4: Supervised Learning - Regression
Task Solution Script
"""

# ==============================================================================
# PART A - DATA PREPARATION & BASELINE MODEL (STEPS 1-5)
# ==============================================================================

# STEP 1: Environment Setup & Data Loading [cite: 474, 475]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load alternative dataset 
california = fetch_california_housing(as_frame=True)
df = california.frame

print(f"Data shape after loading: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# STEP 2: Feature Selection & Target Preparation [cite: 481, 482]
# Using all California Housing features. Target is MedHouseVal.
X = df.drop('MedHouseVal', axis=1)
target_untransformed = df['MedHouseVal']

print(f"Target skewness before log: {target_untransformed.skew():.2f}")
# Apply log1p transform [cite: 482]
y = np.log1p(target_untransformed)
print(f"Target skewness after log: {y.skew():.2f}")

# STEP 3: Train-Test Split & Scaling [cite: 489, 490]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Sizes - X_train: {X_train.shape}, X_test: {X_test.shape}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test) # Transform only [cite: 490]

print(f"X_train_s mean: {np.mean(X_train_s):.2f}, std: {np.std(X_train_s):.2f}")

# STEP 4: Utility Functions [cite: 494, 500]
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, n_features: int) -> dict:
    """Computes regression metrics and returns them as a dictionary."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"\n--- {model_name} ---")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R2:       {r2:.4f}")
    print(f"Adj R2:   {adj_r2:.4f}")
    print(f"MAPE:     {mape:.2f}%")
    
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Adj_R2': adj_r2, 'MAPE': mape}

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Creates a scatter plot of actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Actual (Log scale)')
    plt.ylabel('Predicted (Log scale)')
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.annotate(f'R2: {r2:.3f}\nRMSE: {rmse:.3f}', xy=(0.05, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
    plt.show()

# STEP 5: Baseline Linear Regression [cite: 505, 506]
lr = LinearRegression()
lr.fit(X_train_s, y_train)

# Evaluate train and test
_ = evaluate_model(y_train, lr.predict(X_train_s), "Linear Regression (Train)", X_train_s.shape[1])
lr_test_metrics = evaluate_model(y_test, lr.predict(X_test_s), "Linear Regression (Test)", X_test_s.shape[1])

plot_actual_vs_predicted(y_test, lr.predict(X_test_s), "Linear Regression")

# Top influential features
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coef', ascending=False).head(15)

plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette=colors)
plt.title('Top Influential Features - Linear Regression')
plt.show()


# ==============================================================================
# PART B - POLYNOMIAL & REGULARIZED MODELS (STEPS 6-11)
# ==============================================================================

# STEP 6: Polynomial Regression [cite: 513, 514]
poly_results = []
for d in [1, 2, 3]:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_tr_poly = poly.fit_transform(X_train_s)
    X_te_poly = poly.transform(X_test_s)
    
    lr_p = LinearRegression()
    lr_p.fit(X_tr_poly, y_train)
    
    tr_r2 = r2_score(y_train, lr_p.predict(X_tr_poly))
    te_r2 = r2_score(y_test, lr_p.predict(X_te_poly))
    
    poly_results.append({
        'Degree': d, 
        'Features': X_tr_poly.shape[1],
        'Train_R2': tr_r2, 
        'Test_R2': te_r2, 
        'Gap': tr_r2 - te_r2
    })

poly_df = pd.DataFrame(poly_results)
print("\nPolynomial Regression Comparison:")
print(poly_df)

plt.figure(figsize=(8, 5))
plt.plot(poly_df['Degree'], poly_df['Train_R2'], label='Train R2', marker='o')
plt.plot(poly_df['Degree'], poly_df['Test_R2'], label='Test R2', marker='s')
plt.title('Polynomial Degree vs R2')
plt.xlabel('Degree')
plt.ylabel('R2 Score')
plt.legend()
plt.show()

# STEP 7: Ridge Regression Exploration [cite: 518, 519, 521]
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
ridge_res = []

for a in alphas:
    r = Ridge(alpha=a)
    r.fit(X_train_s, y_train)
    pred = r.predict(X_test_s)
    ridge_res.append({'alpha': a, 'log_alpha': np.log10(a),
                      'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                      'R2': r2_score(y_test, pred)})

r_df = pd.DataFrame(ridge_res)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(r_df['log_alpha'], r_df['RMSE'], marker='o')
axes[0].set_title('Ridge: RMSE vs Log(Alpha)')
axes[1].plot(r_df['log_alpha'], r_df['R2'], marker='s', color='orange')
axes[1].set_title('Ridge: R2 vs Log(Alpha)')
plt.show()

ridge_grid = GridSearchCV(Ridge(), {'alpha': alphas}, cv=5, scoring='r2')
ridge_grid.fit(X_train_s, y_train)
best_ridge = ridge_grid.best_estimator_
best_ridge_metrics = evaluate_model(y_test, best_ridge.predict(X_test_s), "Best Ridge", X_test_s.shape[1])

# STEP 8: Lasso Regression Exploration [cite: 526, 528, 529]
lasso_alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
lasso_res = []

for a in lasso_alphas:
    l = Lasso(alpha=a, max_iter=10000)
    l.fit(X_train_s, y_train)
    zeros = np.sum(l.coef_ == 0)
    pred = l.predict(X_test_s)
    lasso_res.append({'alpha': a, 'zeroed_features': zeros, 
                      'pct_eliminated': zeros / len(l.coef_) * 100,
                      'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                      'R2': r2_score(y_test, pred)})

l_df = pd.DataFrame(lasso_res)
plt.bar(l_df['alpha'].astype(str), l_df['zeroed_features'])
plt.title('Lasso: Eliminated Features vs Alpha')
plt.show()

lasso_grid = GridSearchCV(Lasso(max_iter=10000), {'alpha': lasso_alphas}, cv=5, scoring='r2')
lasso_grid.fit(X_train_s, y_train)
best_lasso = lasso_grid.best_estimator_
best_lasso_metrics = evaluate_model(y_test, best_lasso.predict(X_test_s), "Best Lasso", X_test_s.shape[1])

survived = X.columns[best_lasso.coef_ != 0]
eliminated = X.columns[best_lasso.coef_ == 0]
print(f"\nLasso Survived Features: {list(survived)}")
print(f"Lasso Eliminated Features: {list(eliminated)}")

# STEP 9: ElasticNet [cite: 534, 535, 536]
enet_alphas = [0.001, 0.01, 0.1, 1]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

enet_grid = GridSearchCV(ElasticNet(max_iter=10000), 
                         {'alpha': enet_alphas, 'l1_ratio': l1_ratios}, 
                         cv=5, scoring='r2')
enet_grid.fit(X_train_s, y_train)
best_enet = enet_grid.best_estimator_
best_enet_metrics = evaluate_model(y_test, best_enet.predict(X_test_s), "Best ElasticNet", X_test_s.shape[1])

# Extract results for heatmap
cv_results = pd.DataFrame(enet_grid.cv_results_)
pivot_table = cv_results.pivot(index='param_alpha', columns='param_l1_ratio', values='mean_test_score')
sns.heatmap(pivot_table, annot=True, cmap='viridis')
plt.title('ElasticNet CV R2 Heatmap')
plt.show()

# STEP 10: Poly + Ridge Pipeline [cite: 546, 547]
pipe = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
param_grid = {'poly__degree': [1, 2], 'ridge__alpha': [0.01, 0.1, 1, 10, 100]}

pipe_grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2')
pipe_grid.fit(X_train, y_train) # Pass unscaled data! [cite: 547]
best_pipe = pipe_grid.best_estimator_
pipe_preds = best_pipe.predict(X_test)
best_pipe_metrics = evaluate_model(y_test, pipe_preds, "Best Poly+Ridge", X_test.shape[1])

# STEP 11: Model Comparison Table [cite: 552, 553, 554]
models_metrics = [lr_test_metrics, best_ridge_metrics, best_lasso_metrics, best_enet_metrics, best_pipe_metrics]
comp_df = pd.DataFrame(models_metrics)

# Convert RMSE back to dollar terms (using np.expm1) [cite: 553]
comp_df['Dollar_RMSE'] = np.expm1(comp_df['RMSE'])
print("\nFINAL MODEL COMPARISON:")
print(comp_df)

comp_df.plot(x='Model', y=['RMSE', 'R2'], kind='bar', secondary_y='R2', figsize=(10, 6))
plt.title('Model Comparison: RMSE and R2')
plt.show()


# ==============================================================================
# PART C - RESIDUAL ANALYSIS, CROSS-VALIDATION & DIAGNOSTICS (STEPS 12-15)
# ==============================================================================

# Identify strictly the best model predictions for residuals
best_overall_preds = best_pipe.predict(X_test)
residuals = y_test - best_overall_preds

# STEP 12: Residual Analysis 2x2 Plot [cite: 560, 561, 562, 563]
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Top-Left: Residuals vs Fitted
axs[0, 0].scatter(best_overall_preds, residuals, alpha=0.5)
axs[0, 0].axhline(0, color='r', linestyle='--')
axs[0, 0].set_title('Residuals vs Fitted')

# Top-Right: Histogram + KDE
sns.histplot(residuals, kde=True, ax=axs[0, 1])
axs[0, 1].set_title('Residual Distribution')

# Bottom-Left: Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axs[1, 0])
axs[1, 0].set_title('Q-Q Plot')

# Bottom-Right: Scale-Location
axs[1, 1].scatter(best_overall_preds, np.sqrt(np.abs(residuals)), alpha=0.5)
axs[1, 1].axhline(np.mean(np.sqrt(np.abs(residuals))), color='r', linestyle='--')
axs[1, 1].set_title('Scale-Location')

plt.tight_layout()
plt.show()

# Shapiro-Wilk [cite: 564]
stat, p_val = stats.shapiro(residuals[:5000]) # Sample limit for Shapiro
print(f"Shapiro-Wilk Test p-value: {p_val:.4e}")


# STEP 13: 5-Fold Cross Validation Boxplot [cite: 567]
models_for_cv = {
    'Linear': LinearRegression(),
    'Ridge': best_ridge,
    'Lasso': best_lasso,
    'ElasticNet': best_enet,
    'Poly+Ridge': best_pipe 
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results_r2 = {}

for name, m in models_for_cv.items():
    if name != 'Poly+Ridge':
        p = Pipeline([('scaler', StandardScaler()), ('model', m)])
    else:
        p = m # already a pipeline
    scores = cross_val_score(p, X_train, y_train, cv=kf, scoring='r2')
    cv_results_r2[name] = scores

cv_df = pd.DataFrame(cv_results_r2)
sns.boxplot(data=cv_df)
plt.title('5-Fold CV R2 Scores Across Models')
plt.show()


# STEP 14: Learning Curves [cite: 570, 571, 572]
def plot_learning_curve(estimator, title, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2')
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="orange", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    plt.title(title)
    plt.legend(loc="best")

fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_learning_curve(Pipeline([('s', StandardScaler()), ('m', LinearRegression())]), "Learning Curve: Linear Reg", X_train, y_train)
plt.subplot(1, 2, 2)
plot_learning_curve(best_pipe, "Learning Curve: Poly+Ridge", X_train, y_train)
plt.show()


# STEP 15: Coefficient Path Visualization [cite: 577, 578, 579]
path_alphas = np.logspace(-3, 3, 50)
ridge_coefs = []
lasso_coefs = []

for a in path_alphas:
    r = Ridge(alpha=a).fit(X_train_s, y_train)
    l = Lasso(alpha=a).fit(X_train_s, y_train)
    ridge_coefs.append(r.coef_)
    lasso_coefs.append(l.coef_)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(np.log10(path_alphas), ridge_coefs)
axs[0].set_title('Ridge Regularization Path')
axs[0].set_xlabel('Log(Alpha)')
axs[0].set_ylabel('Coefficients')

axs[1].plot(np.log10(path_alphas), lasso_coefs)
axs[1].set_title('Lasso Regularization Path')
axs[1].set_xlabel('Log(Alpha)')
plt.show()


# ==============================================================================
# PART D - FINAL DASHBOARD, PREDICTIONS & WRITTEN REPORT (STEPS 16-18)
# ==============================================================================

# STEP 16: Complete 6-Chart Model Evaluation Dashboard [cite: 589, 590, 591, 592, 593, 594, 595, 596]
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Regression Model Evaluation Dashboard - [Your Name]', fontsize=16)

# Chart 1: Actual vs Pred (All Models)
axes[0, 0].scatter(y_test, lr.predict(X_test_s), alpha=0.3, label=f'Linear (R2: {lr_test_metrics["R2"]:.2f})')
axes[0, 0].scatter(y_test, best_pipe.predict(X_test), alpha=0.3, label=f'Best Poly+Ridge (R2: {best_pipe_metrics["R2"]:.2f})')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_title('1. Actual vs Predicted (Best Models)')
axes[0, 0].legend()

# Chart 2: Residuals vs Fitted (Best Model)
axes[0, 1].scatter(best_overall_preds, residuals, c=np.where(residuals>0, 'blue', 'red'), alpha=0.5)
axes[0, 1].axhline(0, color='k', linestyle='--')
axes[0, 1].set_title('2. Residuals vs Fitted (Poly+Ridge)')

# Chart 3: CV Boxplots
sns.boxplot(data=cv_df, ax=axes[1, 0])
axes[1, 0].set_title('3. 5-Fold CV R2 Distribution')

# Chart 4: Feature Coefs (Ridge representation here since Poly+Ridge has expanded features)
c_df = pd.DataFrame({'Feature': X.columns, 'Coef': best_ridge.coef_}).sort_values(by='Coef', key=abs, ascending=False)
sns.barplot(x='Coef', y='Feature', data=c_df.head(20), ax=axes[1, 1])
axes[1, 1].set_title('4. Top Feature Coefficients (Best Ridge)')

# Chart 5: Alpha tuning
axes[2, 0].plot(r_df['log_alpha'], r_df['RMSE'], color='blue', label='RMSE')
ax2 = axes[2, 0].twinx()
ax2.plot(r_df['log_alpha'], r_df['R2'], color='orange', label='R2')
axes[2, 0].set_title('5. Alpha Tuning (Ridge)')

# Chart 6: Learning Curve (Best Model)
train_sizes, tr_sc, te_sc = learning_curve(best_pipe, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
axes[2, 1].plot(train_sizes, np.mean(tr_sc, axis=1), label='Train')
axes[2, 1].plot(train_sizes, np.mean(te_sc, axis=1), label='Validation')
axes[2, 1].fill_between(train_sizes, np.mean(tr_sc, axis=1)-np.std(tr_sc, axis=1), np.mean(tr_sc, axis=1)+np.std(tr_sc, axis=1), alpha=0.1)
axes[2, 1].fill_between(train_sizes, np.mean(te_sc, axis=1)-np.std(te_sc, axis=1), np.mean(te_sc, axis=1)+np.std(te_sc, axis=1), alpha=0.1)
axes[2, 1].set_title('6. Learning Curve (Poly+Ridge)')
axes[2, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('week4_dashboard.png', dpi=150)
print("\nDashboard saved to 'week4_dashboard.png'")


# STEP 17: Generate Predictions on Test Set & Save Model [cite: 599, 600, 601, 602, 603, 604]
actual_dollars = np.expm1(y_test)
pred_dollars = np.expm1(best_overall_preds)

preds_df = pd.DataFrame({
    'index': y_test.index,
    'actual_price': actual_dollars,
    'predicted_price': pred_dollars
})
preds_df['error'] = preds_df['predicted_price'] - preds_df['actual_price']
preds_df['pct_error'] = (np.abs(preds_df['error']) / preds_df['actual_price']) * 100

print("\nTop 10 Largest Prediction Errors:")
print(preds_df.sort_values(by='pct_error', ascending=False).head(10))

# Joblib operations
joblib.dump(best_pipe, 'week4_best_model.pkl')
loaded_model = joblib.load('week4_best_model.pkl')
sample_pred = loaded_model.predict(X_test.iloc[[0]])
print(f"\nVerification Prediction from Loaded Model (Log Scale): {sample_pred[0]:.4f}")

# Over/Under predictions
over_pred = preds_df.sort_values(by='error', ascending=False).head(15)
under_pred = preds_df.sort_values(by='error', ascending=True).head(15)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x='error', y=over_pred['index'].astype(str), data=over_pred, ax=axs[0], color='red')
axs[0].set_title('Top 15 Over-Predicted')
sns.barplot(x='error', y=under_pred['index'].astype(str), data=under_pred, ax=axs[1], color='blue')
axs[1].set_title('Top 15 Under-Predicted')
plt.show()

# ==============================================================================
# STEP 18: Written Analysis Report (Template Generation) [cite: 607, 608, 609, 610, 611, 612, 613, 614, 615]
# ==============================================================================
markdown_report_template = """
# Written Analysis Report

## (1) Executive Summary
*Requires 80+ words summarizing the 5 models trained, the best performer, and the metrics achieved.*

## (2) Feature Engineering Impact
*Requires 80+ words detailing which features from the preprocessing helped most in the models.*

## (3) Model-by-Model Analysis
*Requires 150+ words total discussing Linear, Ridge, Lasso, ElasticNet, and Poly+Ridge.*

## (4) Regularization Insights
*Requires 100+ words mathematically comparing Ridge vs Lasso vs ElasticNet, and which features Lasso eliminated.*

## (5) Residual Analysis Findings
*Requires 80+ words on diagnostic plots and linear regression assumptions.*

## (6) Best Model Recommendation
*Requires 80+ words on the technical and business justification for the final deployed model.*

## (7) Reflection
*Requires 80+ words discussing the hardest concept, surprises, and future steps.*
"""

with open("Week4_Written_Report_Template.md", "w") as f:
    f.write(markdown_report_template)
print("\nMarkdown report template created as 'Week4_Written_Report_Template.md'.")
