import matplotlib.pyplot as plt
import pandas as pd

# Prepare DataFrame
data = {
    "Model": [
        "LGBM", "LGBM", "LGBM", "LGBM-1", "LGBM-2",
        "XGB", "XGB", "XGB", "XGB",
        "RF", "RF", "RF", "RF"
    ],
    "Step": [
        "full_Default", "opt-feat1_default", "opt-feat1_opt", "opt-feat1_opt_opt-feat2", "opt-feat1_opt_opt-feat2",
        "full_Default", "opt-feat1_default", "opt-feat1_opt", "opt-feat1_opt_opt-feat2",
        "full_Default", "opt-feat1_default", "opt-feat1_opt", "opt-feat1_opt_opt-feat2"
    ],
    "k_features": [
        132, 67, 67, 121, 71,
        132, 73, 73, 63,
        132, 27, 27, 27
    ],
    "R2_score": [
        0.73184, 0.7529, 0.7529, 0.7666, 0.7748,
        0.73488, 0.7531, 0.7516, 0.7764,
        0.7076, 0.7218, 0.7189, 0.7189
    ]
}

df = pd.DataFrame(data)

# Define consistent parameters
step_labels = ["full_Default", "opt-feat1_default", "opt-feat1_opt", "opt-feat1_opt_opt-feat2"]
group_spacing = 1.5  # space between steps
bar_width = 0.3     # width of each bar
model_order = ["LGBM-1", "LGBM-2", "LGBM", "XGB", "RF"]
model_colors = {
    "LGBM": "#1f77b4",
    "LGBM-1": "#1f77b4",
    "LGBM-2": "#1f77b4",
    "XGB": "#ff7f0e",
    "RF": "#2ca02c"
}
legend_labels = {"LGBM": "LGBM", "XGB": "XGB", "RF": "RF"}

# Determine unique models at each step
models_per_step = {
    "full_Default": ["LGBM", "XGB", "RF"],
    "opt-feat1_default": ["LGBM", "XGB", "RF"],
    "opt-feat1_opt": ["LGBM", "XGB", "RF"],
    "opt-feat1_opt_opt-feat2": ["LGBM-1", "LGBM-2", "XGB", "RF"]
}

# Calculate x positions for each step
x_positions = []
for i, step in enumerate(step_labels):
    n_models = len(models_per_step[step])
    step_center = i * group_spacing
    offsets = []
    if n_models % 2 == 0:
        # Even number of bars
        start = - (n_models / 2 - 0.5) * (bar_width + 0.05)
        offsets = [start + j * (bar_width + 0.05) for j in range(n_models)]
    else:
        # Odd number of bars
        start = - (n_models // 2) * (bar_width + 0.05)
        offsets = [start + j * (bar_width + 0.05) for j in range(n_models)]
    for model, offset in zip(models_per_step[step], offsets):
        x_positions.append((step, model, step_center + offset))

# Plotting
plt.figure(figsize=(14, 7))
for (step, model, x) in x_positions:
    row = df[(df['Step'] == step) & (df['Model'] == model)]
    if row.empty:
        continue
    row = row.iloc[0]
    plt.bar(
        x,
        row['R2_score'],
        width=bar_width,
        color=model_colors.get(model, 'grey'),
        label=model if model in legend_labels else ""  # only label main models
    )
    plt.text(
        x,
        row['R2_score'] + 0.002,
        f'k={row["k_features"]}\nR²={row["R2_score"]:.3f}',
        ha='center', va='bottom', fontsize=9
    )

# X-axis labels
step_centers = [i * group_spacing for i in range(len(step_labels))]
plt.xticks(step_centers, step_labels, rotation=45, ha='right')
plt.ylabel("R² Score")
plt.xlabel("Model Optimization Steps and History")
plt.ylim(0.70, 0.79)
plt.title("Model Performance Comparison (R² Scores)")

# Legend (manually add unique entries)
handles = []
labels = []
for model, color in legend_labels.items():
    handles.append(plt.bar(0, 0, color=model_colors[model], label=model))
plt.legend(handles, legend_labels.values(), title="Model")

# Annotation
note = (
    "Model Optimization steps (4): No. of features changes from: default > "
    "No. feature optimized based on SHAP feature importance value while ranked by R² score for default model ('opt-feat1') > "
    "Bayesian Hyperparameter Tuned optimized model > No. features optimized based on SHAP feature importance value while ranked by R² score for the Bayes optimized model ('opt-feat2'). "
    "**Only for LGBM, started with 50 features as opt-feat1 ended up giving better performance and a smaller feature set.**"
)
plt.figtext(0.5, -0.05, note, wrap=True, horizontalalignment='center', fontsize=8, color='red',
            bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})

plt.tight_layout()
plt.show()