import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator

# ===================
# Matplotlib settings
# ===================
rcParams.update({
    
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 14,               
    "axes.labelsize": 7,          
    "axes.titlesize": 8,          
    "xtick.labelsize": 6,         
    "ytick.labelsize": 6,
    "legend.fontsize": 6,        
    "figure.titlesize": 8,

    "lines.linewidth": 0.8,
    "lines.markersize": 3,

    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "axes.linewidth": 0.4,

    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "axes.edgecolor": "0.3",
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,

    # Tight layout control
    "figure.autolayout": False
})

# =================
# FireDisplay class
# =================
class FireDisplay:

    def __init__(self, height, width, wind_direction, elevation):

        self.height = height
        self.width = width
        self.wind_direction = wind_direction
        self.elevation = elevation

        # Wind components
        self.wind_u = np.cos(self.wind_direction * 2 * np.pi)
        self.wind_v = np.sin(self.wind_direction * 2 * np.pi)

    # ---------------- Visualization Initialization ----------------
    def _initialize_visualization(self):

        fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.0))
        axes = axes.ravel()
        fig.subplots_adjust(wspace=0.35, hspace=0.4, top=0.88)

        fig.suptitle(
            rf"Wind Direction: {self.wind_direction * 2 * np.pi:.2f}\,rad",
            fontsize=8,
            y=0.97
        )

        titles = [
            "Target Fire", "Predicted Fire", "Fire Error",
            "Target Fuel", "Predicted Fuel", "Fuel Error"
        ]

        images = {}

        def add_subplot(ax, title, cmap, vmin=0, vmax=1):
            img = ax.imshow(np.zeros((self.height, self.width)), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(img, ax=ax, fraction=0.035, pad=0.03)
            return img

        images["true_fire"] = add_subplot(axes[0], titles[0], "viridis")
        images["pred_fire"] = add_subplot(axes[1], titles[1], "viridis")
        images["fire_error"] = add_subplot(axes[2], titles[2], "coolwarm", vmin=-1)
        images["true_fuel"] = add_subplot(axes[3], titles[3], "YlGn")
        images["pred_fuel"] = add_subplot(axes[4], titles[4], "YlGn")
        images["fuel_error"] = add_subplot(axes[5], titles[5], "coolwarm", vmin=-1)

        return fig, images

    # ---------------- Visualization Update ----------------
    def _update(self, images, predicted_fire, predicted_fuel, true_fire, true_fuel):

        images["pred_fire"].set_data(predicted_fire)
        images["pred_fuel"].set_data(predicted_fuel)
        images["true_fire"].set_data(true_fire)
        images["true_fuel"].set_data(true_fuel)
        images["fire_error"].set_data(predicted_fire - true_fire)
        images["fuel_error"].set_data(predicted_fuel - true_fuel)

        plt.pause(0.01)

    # ---------------- Final Visualization ----------------
    def show(self, fire_history, fire_truth):

        n_steps = len(fire_history)
        h, w = self.height, self.width
        elev = self.elevation

        # --- Compute arrival times for simulated fire ---
        fire_stack = np.array(fire_history) > 0.5
        arrival_time_sim = np.zeros((h, w))
        for t in range(n_steps):
            new_fire = (fire_stack[t]) & (arrival_time_sim == 0)
            arrival_time_sim[new_fire] = t + 1

        burned_sim = arrival_time_sim > 0
        vmin, vmax = np.min(arrival_time_sim[burned_sim]), np.max(arrival_time_sim[burned_sim])
        levels = np.linspace(vmin, vmax, 10)

        # --- Compute arrival times for ground truth ---
        if isinstance(fire_truth, (list, np.ndarray)) and np.array(fire_truth).ndim == 3:
            fire_truth = np.array(fire_truth)
            n_steps_truth = len(fire_truth)
            arrival_time_truth = np.zeros((h, w))
            for t in range(n_steps_truth):
                new_fire = (fire_truth[t] > 0.5) & (arrival_time_truth == 0)
                arrival_time_truth[new_fire] = t + 1
        else:
            arrival_time_truth = (np.array(fire_truth) > 0.5).astype(float) * n_steps 

        burned_truth = arrival_time_truth > 0

        # --- Create figure with two subplots ---
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), constrained_layout=True)
        titles = ["Simulated Fire Spread", "Ground Truth Fire Spread"]

        for i, (ax, arrival_time, burned, title) in enumerate(zip(
            axes,
            [arrival_time_sim, arrival_time_truth],
            [burned_sim, burned_truth],
            titles
        )):
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("South", fontsize=7)
            ax.set_ylabel("West", fontsize=7)

            # Elevation
            ax.imshow(elev, cmap="terrain", origin="upper", vmin=0, vmax=1)

            # Contour fire arrival times
            filled = ax.contourf(arrival_time, levels=levels, cmap="Reds", alpha=0.7)

            # Wind vectors
            yy, xx = np.mgrid[0:h, 0:w]
            step = max(1, int(min(h, w) // 40))
            ax.quiver(xx[::step, ::step], yy[::step, ::step],
                    self.wind_u, self.wind_v,
                    scale=70, width=0.002, color="0.3", alpha=0.8)

            # Mark ignition
            if np.any(burned):
                first_t = np.min(arrival_time[burned])
                iy, ix = np.where(arrival_time == first_t)
                ax.scatter(ix, iy, s=6, c="red", marker="o",
                        edgecolors="black", linewidths=0.2)

        cbar = fig.colorbar(filled, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03)
        cbar.set_label("Fire arrival time (timesteps)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        plt.show()
        
# ====================
# MetricsDisplay class
# ====================
class MetricsDisplay:
    """Metrics roll-out visualization"""

    def __init__(self, jaccard_scores, hausdorff_distances):
        
        self.jaccard_scores = np.asarray(jaccard_scores, dtype=float)
        self.hausdorff_distances = np.asarray(hausdorff_distances, dtype=float)

        self.n = len(self.jaccard_scores)

    def plot(self, x_label = "Time step", y_left = "JSC score", y_right = "HD score", figsize = (6, 4)):

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 14,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.linewidth": 1.0,
            }
        )

        x = np.arange(self.n)

        col_jaccard = "#A580F7"
        col_hd = "#A8E5E8"  

        fig, ax_left = plt.subplots(figsize=figsize)
        ax_right = ax_left.twinx()

        # --- left axis: Jaccard ---
        l1 = ax_left.plot(
            x,
            self.jaccard_scores,
            linestyle="--",
            linewidth=1.8,
            marker=None,
            color=col_jaccard,
            label="JSC",
        )[0]

        ax_left.set_ylabel(y_left)
        ax_left.set_ylim(0, 1.05)

        # --- right axis: Hausdorff ---
        l2 = ax_right.plot(
            x,
            self.hausdorff_distances,
            linestyle="-",
            linewidth=1.8,
            marker=None,
            color=col_hd,
            label="HD",
        )[0]

        ax_right.set_ylabel(y_right)

        ax_left.set_xlabel(x_label)
        ax_left.set_xlim(x[0], x[-1])

        for ax in (ax_left, ax_right):
            ax.tick_params(
                direction="out",
                length=4,
                width=1.0,
                top=False,
                right=(ax is ax_right),
            )

        for spine in ax_left.spines.values():
            spine.set_visible(True)
        for spine in ax_right.spines.values():
            if spine in ("top", "right"):
                ax_right.spines[spine].set_visible(False)

        lines = [l1, l2]
        labels = [ln.get_label() for ln in lines]

        ax_left.legend(
            lines,
            labels,
            loc="center left",     
            frameon=True,          
            framealpha=0.9,        
            fancybox=False, 
            edgecolor="black",
            borderpad=0.4,
        )

        fig.tight_layout()

        plt.show()

class SimulationDisplay:

    def __init__(self, jsc_scores, hd_scores, runtimes, config_labels):

        self.jsc = np.asarray(jsc_scores)
        self.hd = np.asarray(hd_scores)
        self.runtimes = np.asarray(runtimes)
        self.config_labels = np.asarray(config_labels)
        self.x = np.arange(len(self.jsc))

        self.colors = {
            "jsc":  "#A8E5E8",
            "hd" :  "#A580F7",
            "time": "#A8E5E8",
        }

    def _prepare_ax(self, xlabel, ylabel, figsize):
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xticks(self.x)
        ax.set_xticklabels(self.config_labels)

        ax.grid(axis="y", linestyle=":", linewidth=0.4, color="0.85")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(direction="in", which="both", top=True, right=True)

        for s in ax.spines.values():
            s.set_linewidth(0.4)

        return fig, ax

    def draw_jsc(self, figsize=(3.4, 2.2)):
        fig, ax = self._prepare_ax("Configuration", "Average value", figsize)

        ax.bar(self.x, self.jsc,
               width=0.35, label="JSC",
               color=self.colors["jsc"],
               edgecolor="black", linewidth=0.3)

        ax.legend(frameon=False, ncol=1, handlelength=1.6)
        plt.tight_layout(pad=0.3)
        plt.show()

    def draw_hd(self, figsize=(3.4, 2.2)):
        fig, ax = self._prepare_ax("Configuration", "Average value", figsize)

        ax.bar(self.x, self.hd,
               width=0.35, label="HD",
               color=self.colors["hd"],
               edgecolor="black", linewidth=0.3)

        ax.legend(frameon=False, ncol=1, handlelength=1.6)
        plt.tight_layout(pad=0.3)
        plt.show()

    def draw_time(self, figsize=(3.4, 2.2), baseline_time=None):
        fig, ax = self._prepare_ax("Configuration", "Time (s, log scale)", figsize)

        ax.set_yscale("log")

        ax.bar(self.x, self.runtimes,
            width=0.5, label="ConvLSTM runtime",
            color=self.colors["time"],
            edgecolor="black", linewidth=0.3)

        ax.axhline(baseline_time, linestyle="--", linewidth=1.2,
            color="#A580F7", label=f"CA model ({baseline_time:.1f} s)")

        ax.legend(frameon=False)
        plt.tight_layout(pad=0.3)
        plt.show()


if __name__ == "__main__":

    # ----- First configuration set -----
    config_labels = [
        r"$100\times100$",
        r"$150\times150$",
        r"$200\times200$",
        r"$250\times250$",
        r"$300\times300$",
    ]

    jsc_means = [0.836148738861084, 0.8072457313537598, 0.7406565546989441, 0.611312985420227, 0.6122322678565979]


    hd_means = [1.9995886087417603, 6.905071258544922, 9.425954818725586, 12.703063011169434, 18.564306259155273]


    viewer = SimulationDisplay(jsc_means, hd_means, [], config_labels)
    viewer.draw_jsc()
    viewer.draw_hd()

    # ----- Second configuration set -----
    config_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    jsc_means = [0.60483717918396, 0.5909613370895386, 0.6353501677513123, 0.6425272226333618, 0.7406565546989441, 0.7484269738197327, 0.7279040813446045, 0.7325802445411682, 0.8272727727890015]


    hd_means = [10.89936637878418, 14.116035461425781, 14.709515571594238, 12.751317977905273, 9.425954818725586, 8.265338897705078, 10.990752220153809, 6.069785118103027, 2.634164571762085]


    time_means = [0.3018938899040222, 0.3585212826728821, 0.44894880056381226, 0.34664666652679443, 0.43496522307395935, 0.6512069702148438, 0.4140872359275818, 0.5858067274093628, 1.0448417663574219]


    viewer = SimulationDisplay(jsc_means, hd_means, time_means, config_labels)
    viewer.draw_jsc()
    viewer.draw_hd()
    viewer.draw_time(baseline_time=54.91)

    # print(54.91/np.array(time_means))

    # ----- Third configuration set -----
    config_labels = [
        r"l=1",
        r"l=2",
        r"l=3",
        r"l=4",
    ]

    jsc_means = [0.60483717918396, 0.6790817975997925, 0.6323289275169373, 0.5501046776771545]
    hd_means = [10.89936637878418, 14.7562837600708, 19.15401840209961, 22.70357894897461]

    viewer = SimulationDisplay(jsc_means, hd_means, [], config_labels)
    viewer.draw_jsc()
    viewer.draw_hd()