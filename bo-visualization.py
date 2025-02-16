import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from scipy.stats import norm

# -- Global Settings --
st.set_page_config(layout="wide")
N_ITER = 8
N_INITIAL_SAMPLES = 2

# -----------------------------
# 1) Objective Functions
# -----------------------------
def objective_function(x):
    """Simple 1D objective function."""
    return np.sin(3 * x) + 0.5 * np.cos(5 * x) - 0.3 * x

def f(x):
    """Rauhe Funktion: schnell oszillierend mit Rauschen."""
    np.random.seed(42)
    return np.sin(3 * x) + 0.5 * np.sin(7 * x) + 0.2 * np.random.randn(*x.shape)

# -- Search Bounds --
BOUNDS = np.array([[-1, 2]])

# -----------------------------
# 2) Kernel Selection
# -----------------------------
def get_kernel(name, **params):
    if name == "Matern":
        length_scale = params.get("length_scale", 0.5)
        nu = params.get("nu", 2.5)
        return Matern(length_scale=length_scale, nu=nu)
    elif name == "RBF":
        length_scale = params.get("length_scale", 0.5)
        return RBF(length_scale=length_scale)
    else:
        return Matern(length_scale=params.get("length_scale", 0.5), nu=params.get("nu", 2.5))

def get_default_kernel(name):
    if name == "Matern":
        return Matern(length_scale=0.5, nu=2.5)
    elif name == "RBF":
        return RBF(length_scale=0.5)
    else:
        return Matern(length_scale=0.5, nu=2.5)

# -----------------------------
# 3) Acquisition Functions
# -----------------------------
def expected_improvement(X, gp, y_max, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - y_max - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei

def probability_of_improvement(X, gp, y_max, xi=0.01):
    """Berechnet die Probability of Improvement (PI)."""
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - y_max - xi) / sigma
    pi = norm.cdf(Z)
    return pi

def knowledge_gradient(X, gp, X_sample, num_mc_samples=50):
    y_sample = gp.predict(X_sample)
    y_max_current = np.max(y_sample)
    kg_vals = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        x = np.atleast_2d(x)
        mu_x, sigma_x = gp.predict(x, return_std=True)
        sigma_x = np.maximum(sigma_x, 1e-9)
        samples = np.random.normal(mu_x, sigma_x, size=num_mc_samples)
        improvement = np.maximum(samples - y_max_current, 0)
        kg_vals[i] = np.mean(improvement)
    return kg_vals

def entropy_search(X, gp):
    _, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    es_vals = 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
    return es_vals

def predictive_entropy_search(X, gp, num_representer_points=100):
    _, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    pes_vals = 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
    return pes_vals

def acquisition_function(X, gp, X_sample, method="ei", acq_params=None):
    acq_params = acq_params or {}
    X = np.atleast_2d(X)
    y_sample = gp.predict(X_sample)
    y_max = np.max(y_sample)

    if method == "ei":
        xi = acq_params.get("xi", 0.01)
        return expected_improvement(X, gp, y_max, xi)
    elif method == "pi":
        xi = acq_params.get("xi", 0.01)
        return probability_of_improvement(X, gp, y_max, xi)
    elif method == "kg":
        num_mc_samples = acq_params.get("num_mc_samples", 50)
        return knowledge_gradient(X, gp, X_sample, num_mc_samples)
    elif method == "es":
        return entropy_search(X, gp)
    elif method == "pes":
        num_representer_points = acq_params.get("num_representer_points", 100)
        return predictive_entropy_search(X, gp, num_representer_points)
    else:
        return np.zeros(X.shape[0])

# -----------------------------
# 4) Full BO Loop storing History
# -----------------------------
def run_bayes_opt(acq_method, kernel_choice, kernel_params, n_iter=5, xi=0.01, acq_params=None):
    kernel = get_kernel(kernel_choice, **kernel_params)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    np.random.seed(42)
    X_sample = np.random.uniform(BOUNDS[0, 0], BOUNDS[0, 1], (N_INITIAL_SAMPLES, 1))
    Y_sample = objective_function(X_sample)

    history = []
    for i in range(n_iter):
        gp.fit(X_sample, Y_sample)
        X_plot = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], 200).reshape(-1, 1)
        mu, std = gp.predict(X_plot, return_std=True)
        X_candidates = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], 100).reshape(-1, 1)
        acq_vals = acquisition_function(X_candidates, gp, X_sample, acq_method, acq_params)
        history.append({
            "X_plot": X_plot,
            "mu": mu,
            "std": std,
            "X_sample": X_sample.copy(),
            "Y_sample": Y_sample.copy(),
            "X_candidates": X_candidates,
            "acq_values": acq_vals,
        })
        X_next = X_candidates[np.argmax(acq_vals)]
        Y_next = objective_function(X_next)
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.append(Y_sample, Y_next)
    return history

# -----------------------------
# 5) Variante 1: Vergleich von EI, PI und KG
# -----------------------------
def plot_variante_1(kernel_choice, kernel_params):
    """
    Zeigt ein 2x3 Gitter:
      - Obere Reihe: GP-Fit (wahre Funktion, GP-Mittelwert, Vertrauensintervalle) inkl. der bisherigen Beobachtungen.
      - Untere Reihe: dazugehörige Akquisitionsfunktionen.
    Es werden die Funktionen EI, PI und KG verglichen.
    """
    iteration_to_show = st.sidebar.slider("Iteration to show", 0, N_ITER - 1, N_ITER - 1)
    af_methods = ["ei", "pi", "kg"]
    n_af = len(af_methods)
    fig, axes = plt.subplots(2, n_af, figsize=(6 * n_af, 10), sharex='col')

    for idx, method in enumerate(af_methods):
        history = run_bayes_opt(
            acq_method=method,
            kernel_choice=kernel_choice,
            kernel_params=kernel_params,
            n_iter=N_ITER,
            xi=0.01,
            acq_params=None  # Standardparameter werden genutzt
        )
        data = history[iteration_to_show]
        ax_gp = axes[0, idx]
        ax_gp.plot(data["X_plot"], objective_function(data["X_plot"]), 'r--', label="True Objective")
        ax_gp.plot(data["X_plot"], data["mu"], 'k-', label="GP mean")
        ax_gp.fill_between(data["X_plot"].ravel(),
                           data["mu"] - 1.96 * data["std"],
                           data["mu"] + 1.96 * data["std"],
                           alpha=0.2, color='k')
        ax_gp.scatter(data["X_sample"], data["Y_sample"], color='b', label="Samples")
        best_idx = np.argmax(data["Y_sample"])
        ax_gp.scatter(data["X_sample"][best_idx], data["Y_sample"][best_idx],
                      color='g', marker='*', s=150, label="Best Obs.")
        ax_gp.set_title(f"{method.upper()} (Kernel: {kernel_choice})")
        ax_gp.set_xlim([BOUNDS[0, 0], BOUNDS[0, 1]])
        ax_gp.set_ylim([-2.5, 2.5])
        if idx == 0:
            ax_gp.legend(loc="upper left", fontsize='small')

        ax_acq = axes[1, idx]
        ax_acq.plot(data["X_candidates"], data["acq_values"], 'g-', label="Acquisition")
        ax_acq.set_title("Acquisition Function")
        ax_acq.set_xlim([BOUNDS[0, 0], BOUNDS[0, 1]])
        ax_acq.set_ylim([0, 1])
        ax_acq.set_xlabel("x")
        ax_acq.set_ylabel("Acquisition Value")
        if idx == 0:
            ax_acq.legend(loc="upper left", fontsize='small')

    fig.tight_layout()
    st.pyplot(fig)

def main_variante_1():
    st.title("View 1: Compare EI, PI, and KG Acquisition Functions")
    kernel_choice = st.sidebar.selectbox("Choose kernel", ["RBF", "Matern"], index=0)
    if kernel_choice == "Matern":
        matern_length_scale = st.sidebar.slider("Matern Length Scale", 0.1, 5.0, 0.5, 0.1)
        matern_nu = st.sidebar.slider("Matern ν (nu)", 0.5, 5.0, 2.5, 0.5)
        kernel_params = {"length_scale": matern_length_scale, "nu": matern_nu}
    elif kernel_choice == "RBF":
        rbf_length_scale = st.sidebar.slider("RBF Length Scale", 0.1, 5.0, 0.5, 0.1)
        kernel_params = {"length_scale": rbf_length_scale}

    plot_variante_1(kernel_choice, kernel_params)

# -----------------------------
# 6) Variante 2: Vergleich von zwei Kerneln für eine ausgewählte Akquisitionsfunktion
# -----------------------------
def run_bayes_opt_for_kernel(kernel_name, acq_method, n_iter=5, acq_params=None):
    kernel = get_default_kernel(kernel_name)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    np.random.seed(42)
    X_sample = np.random.uniform(BOUNDS[0, 0], BOUNDS[0, 1], (N_INITIAL_SAMPLES, 1))
    Y_sample = objective_function(X_sample)

    history = []
    for i in range(n_iter):
        gp.fit(X_sample, Y_sample)
        X_plot = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], 200).reshape(-1, 1)
        mu, std = gp.predict(X_plot, return_std=True)
        X_candidates = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], 100).reshape(-1, 1)
        acq_vals = acquisition_function(X_candidates, gp, X_sample, acq_method, acq_params)
        history.append({
            "X_plot": X_plot,
            "mu": mu,
            "std": std,
            "X_sample": X_sample.copy(),
            "Y_sample": Y_sample.copy(),
            "X_candidates": X_candidates,
            "acq_values": acq_vals
        })
        X_next = X_candidates[np.argmax(acq_vals)]
        Y_next = objective_function(X_next)
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.append(Y_sample, Y_next)
    return history

def plot_variante_2(acq_method, acq_params):
    """
    Zeigt ein 2x2 Gitter:
      - Links: Matern-Kernel
      - Rechts: RBF-Kernel
    """
    iteration_to_show = st.sidebar.slider("Iteration to show", 0, N_ITER - 1, N_ITER - 1)
    kernels = ["Matern", "RBF"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col')

    for col, kernel_name in enumerate(kernels):
        history = run_bayes_opt_for_kernel(
            kernel_name=kernel_name,
            acq_method=acq_method,
            n_iter=N_ITER,
            acq_params=acq_params
        )
        data = history[iteration_to_show]
        ax_top = axes[0, col]
        ax_top.plot(data["X_plot"], objective_function(data["X_plot"]), 'r--', label="True")
        ax_top.plot(data["X_plot"], data["mu"], 'k-', label="GP mean")
        ax_top.fill_between(data["X_plot"].ravel(),
                            data["mu"] - 1.96 * data["std"],
                            data["mu"] + 1.96 * data["std"],
                            alpha=0.2, color='k')
        ax_top.scatter(data["X_sample"], data["Y_sample"], color='b', label="Samples")
        best_idx = np.argmax(data["Y_sample"])
        ax_top.scatter(data["X_sample"][best_idx], data["Y_sample"][best_idx],
                       color='g', marker='*', s=150, label="Best Obs.")
        ax_top.set_title(f"{kernel_name}-Kernel (AF: {acq_method.upper()})")
        ax_top.set_xlim([BOUNDS[0, 0], BOUNDS[0, 1]])
        ax_top.set_ylim([-2.5, 2.5])
        if col == 0:
            ax_top.legend()
        ax_bottom = axes[1, col]
        ax_bottom.plot(data["X_candidates"], data["acq_values"], 'g-')
        ax_bottom.set_title("Acquisition Function")
        ax_bottom.set_xlim([BOUNDS[0, 0], BOUNDS[0, 1]])
        ax_bottom.set_ylim([0, 1])

    fig.tight_layout()
    st.pyplot(fig)

def main_variante_2():
    st.title("View 2: Compare Kernel Functions")
    acq_method = st.sidebar.selectbox("Choose acquisition function", ["ei", "pi", "kg", "es", "pes"], index=0)
    acq_params = {}
    if acq_method in ["ei", "pi"]:
        xi_slider = st.sidebar.slider("xi", 0.0, 1.0, 0.01, 0.01)
        acq_params["xi"] = xi_slider
    elif acq_method == "kg":
        num_mc_samples = st.sidebar.slider("KG: num_mc_samples", 10, 100, 50, 1)
        acq_params["num_mc_samples"] = num_mc_samples

    plot_variante_2(acq_method, acq_params)

# -------------------------------------------------------------------------
# Streamlit App mit zwei Tabs: Variante 1 und Variante 2
# -------------------------------------------------------------------------
def main():
    st.sidebar.title("Navigation")
    obj_choice = st.sidebar.selectbox("Choose objective function", ["Simple function", "Rough function"], index=0)
    global objective_function
    if obj_choice == "Rough function":
        objective_function = f

    choice = st.sidebar.radio("Choose view:", ["Compare Acquisition Functions", "Compare Kernel Functions"])
    if choice == "Compare Acquisition Functions":
        main_variante_1()
    else:
        main_variante_2()

if __name__ == "__main__":
    main()
