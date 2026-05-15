# Destructive Size Plot

import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

def plot_dsize_reference():
    """Static reference chart (no user data)"""
    labels = ['D1', 'D1.5', 'D2', 'D2.5', 'D3', 'D3.5', 'D4', 'D4.5', 'D5']
    min_t    = [1, 10, 75, 250, 750, 2500, 7500, 25000, 75000]
    max_t    = [10, 75, 250, 750, 2500, 7500, 25000, 75000, 200000]
    typical_t = [10, None, 100, None, 1000, None, 10000, None, 100000]

    colors = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c',
              '#fc4e2a', '#e31a1c', '#bd0026', '#800026']

    fig, ax = plt.subplots(figsize=(14, 8))

    for i in range(len(labels)):
        ax.fill_between([i - 0.35, i + 0.35], min_t[i], max_t[i],
                        color=colors[i], alpha=0.95, edgecolor='black', linewidth=2)

        if typical_t[i] is not None:
            ax.plot([i - 0.35, i + 0.35], [typical_t[i], typical_t[i]],
                    color='black', linewidth=6)
            label_text = "<10 t" if i == 0 else f'{typical_t[i]:,} t'
            ax.text(i, typical_t[i] * 1.18, label_text,
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    path_effects=[withStroke(linewidth=2, foreground='white')])

        ax.text(i, min_t[i] * 0.94 if min_t[i] > 0 else 1, f'{min_t[i]:,}', 
                ha='center', va='top', fontsize=10)
        if i == 0:
            pass
        elif i == 8:
            ax.text(i, max_t[i] * 1.08, "∞", ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax.text(i, max_t[i] * 1.08, f'{max_t[i]:,}', ha='center', va='bottom', fontsize=10)

    ax.plot([], [], color='black', linewidth=6, label='Typical value from Destructive Scale')
    ax.legend(loc='upper left', fontsize=11)

    ax.set_yscale('log')
    ax.set_ylabel('Avalanche Mass (tonnes)', fontsize=14)
    ax.set_title('Avalanche Destructive Size (D-Size) Classification\n'
                 'Mass Ranges and Typical Values (Log Scale)', fontsize=16, pad=20)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
    ax.set_yticklabels(['1', '10', '100', '1,000', '10,000', '100,000'])

    plt.tight_layout()
    return fig


def plot_dsize_with_user_mass(user_mass: float, low_mass: float = None, high_mass: float = None):
    """Dynamic chart showing the user's calculated mass + uncertainty range"""
    fig = plot_dsize_reference()  # reuse the static reference chart
    ax = fig.axes[0]

    # User's estimated mass (thick black line)
    ax.axhline(y=user_mass, color='black', linewidth=3, linestyle='-.',
               label=f'Your estimated mass: {user_mass:,.0f} t')

    # Uncertainty range (shaded band + legend)
    if low_mass is not None and high_mass is not None:
        ax.fill_between([0, 8], low_mass, high_mass,
                        color='blue', alpha=0.5, 
                        label=f'Uncertainty range: {low_mass:,.0f} – {high_mass:,.0f} t')

    ax.legend(
        loc='upper left', 
        fontsize=14, 
        handlelength=4.5,      # longer line segments
        handleheight=2.0,      # taller symbols
        handletextpad=0.8      # spacing between symbol and text
    )
    ax.set_title('Your Avalanche vs D-Size Classification\n'
                 'Mass Ranges and Typical Values (Log Scale)', fontsize=16, pad=20)

    return fig