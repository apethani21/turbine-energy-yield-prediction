import os
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import probplot
from statsmodels.stats.stattools import jarque_bera
from sklearn import metrics


def combine_data():
    data = []
    for file in sorted(os.listdir("pp_gas_emission")):
        df = pd.read_csv(f"./pp_gas_emission/{file}")
        df['year'] = int(file[3:7])
        data.append(df)
    df = pd.concat(data)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["target"] = df["TEY"].shift(-1)
    df.drop(df.tail(1).index, inplace=True)
    df.to_parquet("pp_gas_emission.parquet")


def evaluate(y_test, y_pred):
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    r2 = metrics.r2_score(y_test, y_pred)
    explained_var_score = metrics.explained_variance_score(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAE = {mae}")
    print(f"MSE = {mse}")
    print(f"RMSE = {rmse}")
    print(f"MAPE = {mape}")
    print(f"R^2 = {r2}")
    print(f"Explained Variance Score = {explained_var_score}")

    residuals = y_pred - y_test
    print("\nResiduals summary stats")
    print(tabulate(
        pd.Series(residuals).describe().to_frame(),
        numalign="right", tablefmt="fancy_grid",
    ))

    with sns.axes_style("darkgrid"):
        g = sns.jointplot(x=y_pred, y=y_pred-y_test, s=8, alpha=0.75, height=12)
        g.set_axis_labels('predicted', 'predicted - actual')
        g.fig.set_figwidth(15)
        g.fig.set_figheight(6)
        g.ax_marg_x._visible = False

        plt.figure(figsize=(10, 8))
        g = sns.jointplot(x=y_pred, y=y_test, s=8, alpha=0.75, height=12)
        g.set_axis_labels('predicted', 'actual')

        fig, ax = plt.subplots(1, 1, figsize=(6, 10))
        probplot(residuals, sparams=(0, 1), plot=ax, fit=False)
        ax.set_title("Residuals probability plot")

        plt.figure(figsize=(20, 5))
        plt.plot(y_pred, alpha=0.5, label="prediction")
        plt.plot(y_test, alpha=0.5, label="actual")
        plt.legend()
        plt.title("Test set prediction over time")
        
        plt.figure(figsize=(20, 5))
        plt.plot(100*(residuals)/y_test, linewidth=0.5)
        plt.title("Test set percentage error over time")
    
    res_mean = residuals.mean()
    res_std = residuals.std()

    one_std_condition = ((residuals >= res_mean - 1*res_std) & (residuals <= res_mean + 1*res_std))
    two_std_condition = ((residuals >= res_mean - 2*res_std) & (residuals <= res_mean + 2*res_std))
    three_std_condition = ((residuals >= res_mean - 3*res_std) & (residuals <= res_mean + 3*res_std))
    perc_within_one_std = 100*len(residuals[np.where(one_std_condition)])/len(residuals)
    perc_within_two_std = 100*len(residuals[np.where(two_std_condition)])/len(residuals)
    perc_within_three_std = 100*len(residuals[np.where(three_std_condition)])/len(residuals)

    print("\nDeviation from mean of residuals")
    print(tabulate(pd.DataFrame({
        "μ ± σ": {"actual": perc_within_one_std, "expected": 68.2},
        "μ ± 2σ": {"actual": perc_within_two_std, "expected": 95.4},
        "μ ± 3σ": {"actual": perc_within_three_std, "expected": 99.7},  
    }).T, headers=["Interval", "Actual", "Expected"], tablefmt="fancy_grid"))
    
    print("\nJarque-Bera Test on Residuals")
    result = jarque_bera(residuals)
    result_labels = ("JB-Statistic", "Chi2-P-value", "Skew", "Kurtosis")
    print(tabulate(list(zip(result_labels, result)), tablefmt="fancy_grid"))


def plot_importance(model):
    with plt.style.context('seaborn'):
        fig, axes = plt.subplots(1, 3, figsize=(18, 18))
        fig.tight_layout(pad=8)
        axes[0] = xgb.plot_importance(model, ax=axes[0],
                                      importance_type="weight",
                                      xlabel="weight")
        axes[1] = xgb.plot_importance(model, ax=axes[1],
                                      importance_type="gain",
                                      xlabel="gain")
        axes[2] = xgb.plot_importance(model, ax=axes[2],
                                      importance_type="cover",
                                      xlabel="cover")
        for ax, title in zip(axes, ["weight", "gain", "cover"]):
            ax.set_title(f"Features ranked by {title}")
            ax.set_ylabel(None)
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            for txt_obj in ax.texts:
                curr_text = txt_obj.get_text()
                new_text = round(float(curr_text), 1)
                txt_obj.set_text(new_text)
    return


def plot_history(model):
    with plt.style.context('seaborn'):
        history = model.evals_result()
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(range(len(history['validation_0']['rmse'])),
                history['validation_0']['rmse'],
                label="Training RMSE", linewidth=2)
        ax.plot(range(len(history['validation_1']['rmse'])),
                history['validation_1']['rmse'],
                label="Testing RMSE", linewidth=2)
        ax.set_title('RMSE')
        ax.grid(True)
        ax.legend()
    return


def rolling_means(df, window, min_periods, cols=None):
    cols = cols or df.columns.to_list()
    df_roll = (df[cols]
               .rolling(window, min_periods=min_periods)
               .mean()
               .fillna(df))
    df_roll.columns = df_roll.columns.map(lambda c: f"{c}_ROLL{window}_MEAN")
    return df_roll


def rolling_standard_deviation(df, window, min_periods, cols=None):
    cols = cols or df.columns.to_list()
    df_roll = (df[cols]
               .rolling(window, min_periods=min_periods)
               .std()
               .fillna(1))
    df_roll.columns = df_roll.columns.map(lambda c: f"{c}_ROLL{window}_STD")
    return df_roll


def lagged_features(df, lags, cols=None):
    if isinstance(lags, int):
        lags = range(1, lags+1)
    cols = cols or df.columns.to_list()
    data = []
    for lag in lags:
        lagged_data = df[cols].shift(lag).fillna(df)
        lagged_data.columns = lagged_data.columns.map(lambda c: f"{c}_LAG{lag}")
        data.append(lagged_data)
    return pd.concat(data, axis="columns")
