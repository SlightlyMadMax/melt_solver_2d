import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ================= ПАРАМЕТРЫ СИСТЕМЫ =================
H = 0.2  # Высота ячейки, м
T_top = 5.0  # Температура верхней стенки, °C
T_bot = -10.0  # Температура нижней стенки, °C
T_c = 3.98  # Температура максимальной плотности воды, °C
T_phi = 0.0  # Температура фазового перехода, °C

k_ice = 2.26  # Теплопроводность льда, Вт/(м·К)
k_w = 0.566  # Теплопроводность воды, Вт/(м·К)
c_w = 4212.0  # Удельная теплоёмкость воды, Дж/(кг·К)
rho_0 = 999.8  # Плотность воды (опорная), кг/м³
mu = 1.7888e-3  # Динамическая вязкость, Па·с
alpha_star = 9.30e-6  # Коэффициент теплового расширения, K^(-q)
q_exp = 1.895  # Степень в законе плотности
g = 9.81  # Ускорение свободного падения, м/с²

# ================= ВЫЧИСЛЯЕМЫЕ СВОЙСТВА ================
nu = mu / rho_0  # Кинематическая вязкость, м²/с
kappa = k_w / (rho_0 * c_w)  # Температуропроводность, м²/с

# Параметры для корреляции Nu(Ra) из статьи Wang et al.
Ra_cr = 1708  # Критическое число Рэлея
C1 = 0.88
C2_base = 0.27
beta_exp = 0.27

# Коэффициент A из баланса льда и стабильного слоя
A = (k_w * (T_top - T_c)) / (k_ice * (T_phi - T_bot))


# ================= КОРРЕЛЯЦИЯ NU(RA) ИЗ СТАТЬИ =================
def get_nu_wang(Ra_e):
    """
    Эмпирическая корреляция Nue(Rae) из статьи Wang et al. (2021)
    """
    if Ra_e <= Ra_cr:
        # Режим чистой теплопроводности
        return 1.0
    else:
        xi = (Ra_e - Ra_cr) / Ra_cr

        if xi <= 1.23:
            # Переходный режим
            return 1.0 + C1 * xi
        else:
            # Развитая конвекция
            C2 = C2_base * (Ra_cr**beta_exp)
            return C2 * (xi**beta_exp)


# ================= ЦЕЛЕВАЯ ФУНКЦИЯ =================
def flux_balance(h0):
    """
    Баланс тепловых потоков: q_ice - q_conv
    Равновесие при flux_balance(h0) == 0
    """
    if h0 <= 0 or h0 >= H:
        return np.inf

    # Толщина неустойчивого (конвективного) слоя
    L_us = H - (1.0 + A) * h0
    if L_us <= 1e-6:
        return np.inf

    # Эффективное число Рэлея для конвективного подслоя
    # Ra_e = g * alpha_star * (T_c - T_phi)^q * L_us^3 / (nu * kappa)
    delta_T = T_c - T_phi  # = 4 K
    Ra_e = (g * alpha_star * (delta_T**q_exp) * (L_us**3)) / (nu * kappa)

    Nu = get_nu_wang(Ra_e)

    # Потоки
    q_ice = k_ice * (T_phi - T_bot) / h0
    q_us = Nu * k_w * (T_c - T_phi) / L_us

    return q_ice - q_us


# ================= ЧИСЛЕННОЕ РЕШЕНИЕ =================
h_min = 1e-4
h_max = H / (1.0 + A) - 1e-4

print(f"Поиск решения в диапазоне h0 = [{h_min*1000:.2f}, {h_max*1000:.2f}] мм")
print(f"Толщина стабильного слоя: H - h4 = {A*1000:.2f} * h0 мм")
print("-" * 60)

try:
    sol = root_scalar(flux_balance, bracket=[h_min, h_max], method="brentq")
    h0_eq = sol.root
    converged = True
except ValueError as e:
    print(f"⚠️ Не удалось найти равновесное состояние: {e}")
    print(
        "Возможно, при данных параметрах лёд полностью растает или заполнит всю ячейку."
    )
    converged = False

if converged:
    L_us = H - (1.0 + A) * h0_eq
    L_st = A * h0_eq
    h4_eq = h0_eq + L_us

    # Пересчёт Ra_e и Nu для финального состояния
    delta_T = T_c - T_phi
    Ra_e = (g * alpha_star * (delta_T**q_exp) * (L_us**3)) / (nu * kappa)
    Nu = get_nu_wang(Ra_e)
    q_eq = k_ice * (T_phi - T_bot) / h0_eq

    print("✅ РЕЗУЛЬТАТЫ РАСЧЁТА (модель Wang et al., 2021):")
    print(f"Равновесное положение границы льда (h0) : {h0_eq*1000:.2f} мм")
    print(f"Толщина льда (H - h0)                    : {(H-h0_eq)*1000:.2f} мм")
    print(f"Толщина конвективного слоя (h4 - h0)    : {L_us*1000:.2f} мм")
    print(f"Толщина стабильного слоя (H - h4)       : {L_st*1000:.2f} мм")
    print(f"Положение изотермы Tc (h4)              : {h4_eq*1000:.2f} мм")
    print(f"Эффективное число Рэлея (Ra_e)          : {Ra_e:.2e}")
    print(f"Число Нуссельта (Nu_e)                  : {Nu:.2f}")
    print(f"Равновесный тепловой поток (q)          : {q_eq:.2f} Вт/м²")
    print("-" * 60)

    # Проверка режима конвекции
    if Ra_e < Ra_cr:
        print("📊 Режим: ЧИСТАЯ ТЕПЛОПРОВОДНОСТЬ (Ra_e < Ra_cr)")
    elif (Ra_e - Ra_cr) / Ra_cr <= 1.23:
        print("📊 Режим: ПЕРЕХОДНЫЙ (слабая конвекция)")
    else:
        print("📊 Режим: РАЗВИТАЯ ТУРБУЛЕНТНАЯ КОНВЕКЦИЯ")

    # ================= ВИЗУАЛИЗАЦИЯ БАЛАНСА =================
    h_vals = np.linspace(h_min * 10, h_max, 200)
    q_ice_vals = k_ice * (T_phi - T_bot) / h_vals
    q_us_vals = np.zeros_like(h_vals)
    Ra_vals = np.zeros_like(h_vals)

    for i, h in enumerate(h_vals):
        L = H - (1 + A) * h
        if L > 1e-6:
            Ra = (g * alpha_star * (delta_T**q_exp) * (L**3)) / (nu * kappa)
            Ra_vals[i] = Ra
            Nu_loc = get_nu_wang(Ra)
            q_us_vals[i] = Nu_loc * k_w * (T_c - T_phi) / L
        else:
            q_us_vals[i] = np.nan
            Ra_vals[i] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # График 1: Баланс потоков
    ax1 = axes[0, 0]
    ax1.plot(h_vals * 1000, q_ice_vals, "b-", lw=2, label=r"$q_{ice}$")
    ax1.plot(h_vals * 1000, q_us_vals, "r-", lw=2, label=r"$q_{conv}$")
    ax1.axvline(
        h0_eq * 1000,
        color="k",
        ls="--",
        lw=1.5,
        label=f"Равновесие: {h0_eq*1000:.2f} мм",
    )
    ax1.set_xlabel("Положение границы льда $h_0$, мм", fontsize=11)
    ax1.set_ylabel("Тепловой поток, Вт/м²", fontsize=11)
    ax1.set_title("Баланс тепловых потоков", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # График 2: Зависимость Ra_e от h0
    ax2 = axes[0, 1]
    ax2.plot(h_vals * 1000, Ra_vals, "g-", lw=2)
    ax2.axhline(Ra_cr, color="r", ls="--", lw=1.5, label=f"Ra_cr = {Ra_cr}")
    ax2.axvline(h0_eq * 1000, color="k", ls="--", lw=1.5)
    ax2.set_xlabel("Положение границы льда $h_0$, мм", fontsize=11)
    ax2.set_ylabel("Эффективное число Рэлея Ra_e", fontsize=11)
    ax2.set_title("Число Рэлея конвективного слоя", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # График 3: Зависимость Nu от Ra
    ax3 = axes[1, 0]
    valid_mask = ~np.isnan(Ra_vals) & (Ra_vals > 0)
    if np.any(valid_mask):
        Nu_plot = np.array([get_nu_wang(Ra) for Ra in Ra_vals[valid_mask]])
        ax3.loglog(Ra_vals[valid_mask], Nu_plot, "m-", lw=2)
        ax3.axvline(Ra_cr, color="r", ls="--", lw=1.5, label=f"Ra_cr = {Ra_cr}")
        ax3.set_xlabel("Число Рэлея Ra_e", fontsize=11)
        ax3.set_ylabel("Число Нуссельта Nu_e", fontsize=11)
        ax3.set_title(
            "Корреляция Nu(Ra) из Wang et al.", fontsize=12, fontweight="bold"
        )
        ax3.grid(True, alpha=0.3, which="both")
        ax3.legend()

    # График 4: Схема слоёв
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Рисуем схему
    y_ice = 0
    y_h0 = h0_eq / H * 100
    y_h4 = h4_eq / H * 100
    y_top = 100

    # Лёд
    rect_ice = plt.Rectangle(
        (0, y_ice), 100, y_h0, color="lightblue", ec="black", lw=2, label="Лёд"
    )
    ax4.add_patch(rect_ice)
    ax4.text(
        50,
        y_h0 / 2,
        f"Лёд\n{H-h0_eq:.1f} мм",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Конвективный слой
    rect_conv = plt.Rectangle(
        (0, y_h0),
        100,
        y_h4 - y_h0,
        color="orange",
        ec="black",
        lw=2,
        alpha=0.6,
        label="Конвекция",
    )
    ax4.add_patch(rect_conv)
    ax4.text(
        50,
        (y_h0 + y_h4) / 2,
        f"Конвективный\n{L_us:.1f} мм\nRa={Ra_e:.1e}",
        ha="center",
        va="center",
        fontsize=9,
    )

    # Стабильный слой
    rect_st = plt.Rectangle(
        (0, y_h4),
        100,
        y_top - y_h4,
        color="yellow",
        ec="black",
        lw=2,
        alpha=0.6,
        label="Стабильный",
    )
    ax4.add_patch(rect_st)
    ax4.text(
        50,
        (y_h4 + y_top) / 2,
        f"Стабильный\n{L_st:.1f} мм",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Границы
    ax4.plot([0, 100], [y_h0, y_h0], "k-", lw=2, label=f"h0 = {h0_eq*1000:.1f} мм")
    ax4.plot([0, 100], [y_h4, y_h4], "k--", lw=2, label=f"h4 = {h4_eq*1000:.1f} мм")

    ax4.set_ylim(0, 110)
    ax4.set_xlim(0, 100)
    ax4.set_title("Структура слоёв системы", fontsize=12, fontweight="bold", pad=10)
    ax4.legend(loc="upper right", fontsize=9)
    ax4.set_aspect("auto")

    plt.tight_layout()
    plt.show()

    # ================= АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ =================
    print("\n📈 АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ:")
    print("Влияние температуры верхней стенки T_top:")

    for T_top_test in [3, 5, 7, 10]:
        A_test = (k_w * (T_top_test - T_c)) / (k_ice * (T_phi - T_bot))

        def flux_test(h):
            L = H - (1 + A_test) * h
            if L <= 1e-6 or h <= 0:
                return np.inf
            Ra = (g * alpha_star * (delta_T**q_exp) * (L**3)) / (nu * kappa)
            Nu = get_nu_wang(Ra)
            q_ice = k_ice * (T_phi - T_bot) / h
            q_us = Nu * k_w * (T_c - T_phi) / L
            return q_ice - q_us

        try:
            h_min_t = 1e-4
            h_max_t = H / (1.0 + A_test) - 1e-4
            if h_max_t > h_min_t:
                sol_t = root_scalar(
                    flux_test, bracket=[h_min_t, h_max_t], method="brentq"
                )
                h_test = sol_t.root
                L_us_t = H - (1 + A_test) * h_test
                Ra_t = (g * alpha_star * (delta_T**q_exp) * (L_us_t**3)) / (nu * kappa)
                print(
                    f"  T_top = {T_top_test}°C → h0 = {h_test*1000:.1f} мм, Ra_e = {Ra_t:.1e}"
                )
        except:
            print(f"  T_top = {T_top_test}°C → нет решения (полное промерзание)")
