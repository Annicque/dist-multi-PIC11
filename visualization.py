"""
Visualisation Interactive pour Distillation Multicomposants
=============================================================
Utilise Plotly pour créer des graphiques interactifs (style D3.js)

Auteur: Prof. BAKHER Zine Elabidine
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrow, Rectangle

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class DistillationVisualizer:
    """
    Classe pour visualiser les résultats de distillation multicomposants
    """
    
    def __init__(self, compound_names):
        """
        Parameters:
        -----------
        compound_names : list of str
            Noms des composés
        """
        self.compound_names = compound_names
        self.n_comp = len(compound_names)
        self.colors = plt.cm.Set3(np.linspace(0, 1, self.n_comp))
        
        if PLOTLY_AVAILABLE:
            self.plotly_colors = px.colors.qualitative.Set3[:self.n_comp]
    
    def plot_material_balance(self, F, D, B, z_F, x_D, x_B, save_path='bilan_matiere.png'):
        """
        Visualise les bilans matières
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Bilans Matières de la Colonne de Distillation',
                     fontsize=14, fontweight='bold')
        
        # Graphique 1: Débits
        streams = ['Alimentation', 'Distillat', 'Résidu']
        flows = [F, D, B]
        colors_streams = ['blue', 'green', 'red']
        
        bars = ax1.bar(streams, flows, color=colors_streams, alpha=0.7,
                      edgecolor='black', linewidth=2)
        
        for bar, flow in zip(bars, flows):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{flow:.1f}\nkmol/h',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_ylabel('Débit (kmol/h)', fontsize=11, fontweight='bold')
        ax1.set_title('Débits des flux', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, max(flows) * 1.2])
        
        # Graphique 2: Compositions
        x = np.arange(self.n_comp)
        width = 0.25
        
        bars1 = ax2.bar(x - width, z_F, width, label='Alimentation',
                       color='blue', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x, x_D, width, label='Distillat',
                       color='green', alpha=0.7, edgecolor='black')
        bars3 = ax2.bar(x + width, x_B, width, label='Résidu',
                       color='red', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Composé', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Fraction molaire', fontsize=11, fontweight='bold')
        ax2.set_title('Compositions des flux', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.compound_names, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_shortcut_results(self, results, save_path='shortcut_results.png'):
        """
        Visualise les résultats des méthodes simplifiées
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Résultats du Dimensionnement (Méthodes Simplifiées)',
                     fontsize=16, fontweight='bold')
        
        # 1. Schéma de la colonne
        ax1 = fig.add_subplot(gs[:, 0])
        self._draw_column_schematic(ax1, results)
        
        # 2. Fenske - N_min
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.6, f"N_min = {results['N_min']:.2f}",
                ha='center', va='center', fontsize=20, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.4, f"α_avg = {results['alpha_avg']:.3f}",
                ha='center', va='center', fontsize=14,
                transform=ax2.transAxes)
        ax2.text(0.5, 0.9, 'Équation de Fenske',
                ha='center', va='top', fontsize=12, fontweight='bold',
                transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.axis('off')
        
        # 3. Underwood - R_min
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.6, f"R_min = {results['R_min']:.3f}",
                ha='center', va='center', fontsize=20, fontweight='bold',
                transform=ax3.transAxes)
        ax3.text(0.5, 0.4, f"θ = {results['theta']:.3f}",
                ha='center', va='center', fontsize=14,
                transform=ax3.transAxes)
        ax3.text(0.5, 0.9, 'Méthode d\'Underwood',
                ha='center', va='top', fontsize=12, fontweight='bold',
                transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax3.axis('off')
        
        # 4. Gilliland
        ax4 = fig.add_subplot(gs[1, 1])
        N_range = np.linspace(results['N_min'], results['N_min'] * 3, 50)
        R_range = []
        for N in N_range:
            Y = (N - results['N_min']) / (N + 1)
            if Y >= 0.999:
                R_range.append(results['R_min'])
            else:
                X_num = np.linspace(0.01, 0.99, 100)
                exponent = (1 + 54.4*X_num) * (X_num - 1) / ((11 + 117.2*X_num) * np.sqrt(X_num))
                Y_curve = 1 - np.exp(exponent)
                idx = np.argmin(np.abs(Y_curve - Y))
                X = X_num[idx]
                R = results['R_min'] + X * (1 + results['R_min']) / (1 - X) if X < 0.999 else results['R_min'] * 10
                R_range.append(min(R, results['R_min'] * 5))
        
        ax4.plot(R_range, N_range, 'b-', linewidth=2.5, label='Courbe de Gilliland')
        ax4.plot(results['R'], results['N_theoretical'], 'ro', markersize=12,
                label=f'Point de fonctionnement\n(R={results["R"]:.2f}, N={results["N_theoretical"]:.1f})',
                zorder=5)
        ax4.axhline(y=results['N_min'], color='g', linestyle='--', linewidth=2,
                   label=f'N_min = {results["N_min"]:.1f}')
        ax4.axvline(x=results['R_min'], color='orange', linestyle='--', linewidth=2,
                   label=f'R_min = {results["R_min"]:.2f}')
        ax4.set_xlabel('Rapport de reflux (R)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Nombre de plateaux (N)', fontsize=11, fontweight='bold')
        ax4.set_title('Corrélation de Gilliland', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([results['R_min'] * 0.9, max(R_range) * 1.1])
        
        # 5. Kirkbride
        ax5 = fig.add_subplot(gs[1, 2])
        stages = list(range(1, results['N_real'] + 1))
        colors_stages = ['lightcoral' if i < results['feed_stage'] 
                        else 'lightblue' if i == results['feed_stage']
                        else 'lightgreen' for i in stages]
        
        ax5.barh(stages, [1]*len(stages), color=colors_stages, edgecolor='black')
        ax5.axhline(y=results['feed_stage'], color='blue', linewidth=3,
                   label=f'Plateau alimentation: {results["feed_stage"]}')
        ax5.set_xlabel('Section', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Numéro de plateau', fontsize=11, fontweight='bold')
        ax5.set_title('Position du plateau d\'alimentation\n(Équation de Kirkbride)',
                     fontsize=12, fontweight='bold')
        ax5.set_xlim([0, 1.2])
        ax5.set_ylim([0, results['N_real'] + 1])
        ax5.set_xticks([])
        ax5.legend()
        ax5.invert_yaxis()
        
        # Annotations
        ax5.text(0.5, results['N_R']/2, f'Rectification\n({results["N_R"]} plateaux)',
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax5.text(0.5, results['feed_stage'] + results['N_S']/2,
                f'Épuisement\n({results["N_S"]} plateaux)',
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 6. Tableau récapitulatif
        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.axis('off')
        
        table_data = [
            ['Paramètre', 'Valeur', 'Unité', 'Méthode'],
            ['N_min', f'{results["N_min"]:.2f}', 'plateaux', 'Fenske'],
            ['R_min', f'{results["R_min"]:.3f}', '-', 'Underwood'],
            ['R opératoire', f'{results["R"]:.3f}', '-', f'{results["R"]/results["R_min"]:.2f}×R_min'],
            ['N théorique', f'{results["N_theoretical"]:.2f}', 'plateaux', 'Gilliland'],
            ['Efficacité', f'{results["efficiency"]*100:.1f}', '%', 'Donnée'],
            ['N réel', f'{results["N_real"]}', 'plateaux', 'N_théo/Efficacité'],
            ['Plateau alimentation', f'{results["feed_stage"]}', '-', 'Kirkbride'],
            ['Débit distillat', f'{results["D"]:.2f}', 'kmol/h', 'Bilan matière'],
            ['Débit résidu', f'{results["B"]:.2f}', 'kmol/h', 'Bilan matière'],
        ]
        
        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.3, 0.2, 0.15, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Mise en forme
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def _draw_column_schematic(self, ax, results):
        """
        Dessine un schéma de la colonne de distillation
        """
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        ax.set_title('Schéma de la colonne', fontsize=12, fontweight='bold')
        
        # Corps de la colonne
        col_width = 0.25
        col_height = 0.65
        col_x = 0.375
        col_y = 0.15
        
        column = FancyBboxPatch((col_x, col_y), col_width, col_height,
                                boxstyle="round,pad=0.01", edgecolor='black',
                                facecolor='lightblue', linewidth=2.5)
        ax.add_patch(column)
        
        # Condenseur
        cond_width = col_width + 0.1
        cond_height = 0.06
        cond_x = col_x - 0.05
        cond_y = col_y + col_height
        
        condenseur = Rectangle((cond_x, cond_y), cond_width, cond_height,
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(condenseur)
        ax.text(col_x + col_width/2, cond_y + cond_height/2, 'Condenseur',
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Rebouilleur
        reb_width = cond_width
        reb_height = cond_height
        reb_x = cond_x
        reb_y = col_y - reb_height
        
        rebouilleur = Rectangle((reb_x, reb_y), reb_width, reb_height,
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
        ax.add_patch(rebouilleur)
        ax.text(col_x + col_width/2, reb_y + reb_height/2, 'Rebouilleur',
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Plateau d'alimentation
        feed_ratio = results['feed_stage'] / results['N_real']
        feed_y = col_y + col_height * (1 - feed_ratio)
        
        ax.plot([col_x - 0.15, col_x], [feed_y, feed_y], 'b-', linewidth=3)
        ax.plot(col_x - 0.15, feed_y, 'bo', markersize=8)
        ax.text(col_x - 0.18, feed_y, 'F', ha='right', va='center',
               fontsize=11, fontweight='bold', color='blue')
        
        # Distillat
        ax.arrow(col_x + col_width/2, cond_y + cond_height, 0, 0.04,
                head_width=0.03, head_length=0.02, fc='green', ec='green', linewidth=2)
        ax.text(col_x + col_width/2, cond_y + cond_height + 0.07, 'D',
                ha='center', va='center', fontsize=11, fontweight='bold', color='green')
        
        # Résidu
        ax.arrow(col_x + col_width/2, reb_y, 0, -0.04,
                head_width=0.03, head_length=0.02, fc='red', ec='red', linewidth=2)
        ax.text(col_x + col_width/2, reb_y - 0.07, 'B',
                ha='center', va='center', fontsize=11, fontweight='bold', color='red')
        
        # Reflux
        ax.annotate('', xy=(col_x + col_width + 0.02, col_y + col_height - 0.05),
                   xytext=(col_x + col_width + 0.02, cond_y + cond_height/2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='purple', linestyle='--'))
        ax.text(col_x + col_width + 0.08, col_y + col_height - 0.02, 'Reflux',
                ha='left', va='center', fontsize=8, color='purple')
        
        # Informations
        info_text = f"""N total = {results['N_real']}
Reflux = {results['R']:.2f}
N_rect = {results['N_R']}
N_strip = {results['N_S']}
Plateau alim = {results['feed_stage']}"""
        
        ax.text(0.05, 0.5, info_text, fontsize=9, verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def plot_composition_profiles_matplotlib(self, stages, x_profiles, y_profiles,
                                            feed_stage, save_path='composition_profiles.png'):
        """
        Trace les profils de composition avec matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle('Profils de Composition dans la Colonne',
                     fontsize=14, fontweight='bold')
        
        # Profils liquides
        for i in range(self.n_comp):
            ax1.plot(x_profiles[:, i], stages, 'o-', linewidth=2.5,
                    markersize=5, label=self.compound_names[i],
                    color=self.colors[i])
        
        ax1.axhline(y=feed_stage, color='blue', linestyle='--', linewidth=2,
                   label='Plateau alimentation')
        ax1.set_xlabel('Fraction molaire liquide (x)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Numéro de plateau', fontsize=11, fontweight='bold')
        ax1.set_title('Phase Liquide', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        ax1.set_xlim([0, 1])
        
        # Profils vapeur
        for i in range(self.n_comp):
            ax2.plot(y_profiles[:, i], stages, 's-', linewidth=2.5,
                    markersize=5, label=self.compound_names[i],
                    color=self.colors[i])
        
        ax2.axhline(y=feed_stage, color='blue', linestyle='--', linewidth=2,
                   label='Plateau alimentation')
        ax2.set_xlabel('Fraction molaire vapeur (y)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Numéro de plateau', fontsize=11, fontweight='bold')
        ax2.set_title('Phase Vapeur', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        ax2.set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_composition_profiles_plotly(self, stages, x_profiles, y_profiles, feed_stage):
        """
        Trace les profils de composition avec Plotly (interactif)
        """
        if not PLOTLY_AVAILABLE:
            print("⚠ Plotly non disponible, utiliser plot_composition_profiles_matplotlib")
            return None
        
        # Créer subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Phase Liquide', 'Phase Vapeur'),
            horizontal_spacing=0.12
        )
        
        # Phase liquide
        for i in range(self.n_comp):
            fig.add_trace(
                go.Scatter(
                    x=x_profiles[:, i],
                    y=stages,
                    mode='lines+markers',
                    name=self.compound_names[i],
                    line=dict(color=self.plotly_colors[i], width=2.5),
                    marker=dict(size=6),
                    legendgroup='compounds',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Ligne plateau alimentation
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[feed_stage, feed_stage],
                mode='lines',
                name='Plateau alimentation',
                line=dict(color='blue', width=2, dash='dash'),
                legendgroup='feed',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Phase vapeur
        for i in range(self.n_comp):
            fig.add_trace(
                go.Scatter(
                    x=y_profiles[:, i],
                    y=stages,
                    mode='lines+markers',
                    name=self.compound_names[i],
                    line=dict(color=self.plotly_colors[i], width=2.5, dash='dot'),
                    marker=dict(size=6, symbol='square'),
                    legendgroup='compounds',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Ligne plateau alimentation
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[feed_stage, feed_stage],
                mode='lines',
                name='Plateau alimentation',
                line=dict(color='blue', width=2, dash='dash'),
                legendgroup='feed',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Mise en forme
        fig.update_xaxes(title_text='Fraction molaire liquide (x)', row=1, col=1,
                        range=[0, 1])
        fig.update_xaxes(title_text='Fraction molaire vapeur (y)', row=1, col=2,
                        range=[0, 1])
        fig.update_yaxes(title_text='Numéro de plateau', row=1, col=1,
                        autorange='reversed')
        fig.update_yaxes(title_text='Numéro de plateau', row=1, col=2,
                        autorange='reversed')
        
        fig.update_layout(
            title_text='Profils de Composition dans la Colonne (Interactif)',
            title_font_size=16,
            height=600,
            hovermode='closest',
            template='plotly_white'
        )
        
        # Sauvegarder en HTML
        fig.write_html('composition_profiles_interactive.html')
        print("✓ Graphique interactif sauvegardé: composition_profiles_interactive.html")
        
        # Afficher
        fig.show()
        
        return fig
    
    def plot_temperature_profile(self, stages, temperatures, feed_stage,
                                save_path='temperature_profile.png'):
        """
        Trace le profil de température
        """
        fig, ax = plt.subplots(figsize=(8, 10))
        
        ax.plot(temperatures - 273.15, stages, 'o-', linewidth=3,
                markersize=8, color='orangered', label='Température')
        
        ax.axhline(y=feed_stage, color='blue', linestyle='--', linewidth=2,
                  label=f'Plateau alimentation ({feed_stage})')
        
        ax.set_xlabel('Température (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Numéro de plateau', fontsize=12, fontweight='bold')
        ax.set_title('Profil de Température dans la Colonne',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        # Annotations
        T_top = temperatures[0] - 273.15
        T_bottom = temperatures[-1] - 273.15
        ax.text(T_top, 1, f'  {T_top:.1f}°C', ha='left', va='center',
               fontsize=10, fontweight='bold', color='darkred')
        ax.text(T_bottom, len(stages), f'  {T_bottom:.1f}°C', ha='left', va='center',
               fontsize=10, fontweight='bold', color='darkred')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()

def print_design_summary(shortcut_results, compound_names):
    """
    Affiche un résumé formaté du dimensionnement
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "  RÉSUMÉ DU DIMENSIONNEMENT DE LA COLONNE".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    print(f"\n{'BILANS MATIÈRES':-^80}")
    print(f"  • Débit alimentation:      {shortcut_results['D'] + shortcut_results['B']:.2f} kmol/h")
    print(f"  • Débit distillat:         {shortcut_results['D']:.2f} kmol/h")
    print(f"  • Débit résidu:            {shortcut_results['B']:.2f} kmol/h")
    
    print(f"\n{'COMPOSITION DISTILLAT':-^80}")
    for i, name in enumerate(compound_names):
        print(f"  • {name:15s}: {shortcut_results['x_D'][i]*100:6.2f}%")
    
    print(f"\n{'COMPOSITION RÉSIDU':-^80}")
    for i, name in enumerate(compound_names):
        print(f"  • {name:15s}: {shortcut_results['x_B'][i]*100:6.2f}%")
    
    print(f"\n{'PARAMÈTRES DE CONCEPTION':-^80}")
    print(f"  • N minimum (Fenske):      {shortcut_results['N_min']:.2f} plateaux")
    print(f"  • R minimum (Underwood):   {shortcut_results['R_min']:.3f}")
    print(f"  • R opératoire:            {shortcut_results['R']:.3f}")
    print(f"  • N théorique (Gilliland): {shortcut_results['N_theoretical']:.2f} plateaux")
    print(f"  • Efficacité:              {shortcut_results['efficiency']*100:.1f}%")
    print(f"  • N réel:                  {shortcut_results['N_real']} plateaux")
    
    print(f"\n{'LOCALISATION':-^80}")
    print(f"  • Plateaux rectification:  {shortcut_results['N_R']}")
    print(f"  • Plateaux épuisement:     {shortcut_results['N_S']}")
    print(f"  • Plateau alimentation:    {shortcut_results['feed_stage']}")
    
    print(f"\n{'DÉBITS INTERNES':-^80}")
    print(f"  • Liquide rectification:   {shortcut_results['L']:.2f} kmol/h")
    print(f"  • Vapeur rectification:    {shortcut_results['V']:.2f} kmol/h")
    print(f"  • Liquide épuisement:      {shortcut_results['L_prime']:.2f} kmol/h")
    
    print("\n" + "=" * 80)