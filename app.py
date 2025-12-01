from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distillation_multicomposants import Compound, ThermodynamicPackage, ShortcutDistillation

app = Flask(__name__)

def get_composition_profiles(results, thermo, N_real, feed_stage, z_F):
    """
    Helper to calculate composition profiles for visualization
    """
    stages = np.arange(1, N_real + 1)
    compounds = thermo.compounds
    
    x_profiles = np.zeros((N_real, len(compounds)))
    y_profiles = np.zeros((N_real, len(compounds)))
    temperatures = np.zeros(N_real)
    
    for j, stage in enumerate(stages):
        # Linear interpolation between distillate and residue
        if stage <= feed_stage:
            # Rectification section
            ratio = (stage - 1) / feed_stage
            x_stage = results['x_D'] + ratio * (z_F - results['x_D'])
        else:
            # Stripping section
            ratio = (stage - feed_stage) / (N_real - feed_stage)
            x_stage = z_F + ratio * (results['x_B'] - z_F)
        
        x_stage = x_stage / np.sum(x_stage)  # Normalize
        x_profiles[j, :] = x_stage
        
        # Bubble temperature
        try:
            # Note: P is in results context or we assume standard P
            # We need P from somewhere. Let's assume it's passed or standard.
            # For now, we'll re-use the P from the shortcut object if we had it, 
            # but here we might need to pass it.
            # Let's assume P=101325 if not available, or pass it in args.
            P = 101325 
            T_bubble, y_stage = thermo.bubble_temperature(P, x_stage)
            temperatures[j] = T_bubble
            y_profiles[j, :] = y_stage
        except:
            # Fallback estimation
            temperatures[j] = compounds[0].Tb + \
                            (compounds[-1].Tb - compounds[0].Tb) * (j / N_real)
            y_profiles[j, :] = x_stage
            
    return stages, x_profiles, y_profiles, temperatures

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.form
        
        # 1. Parse Inputs
        F = float(data.get('feed_flow', 100.0))
        P = float(data.get('pressure', 101325))
        q = float(data.get('quality', 1.0))
        
        # Compositions
        z_benzene = float(data.get('z_benzene', 0.333))
        z_toluene = float(data.get('z_toluene', 0.333))
        z_xylene = float(data.get('z_xylene', 0.334))
        
        # Normalize composition
        total_z = z_benzene + z_toluene + z_xylene
        z_F = np.array([z_benzene, z_toluene, z_xylene]) / total_z
        
        # Specs
        recovery_LK = float(data.get('recovery_lk', 0.95))
        recovery_HK = float(data.get('recovery_hk', 0.95))
        R_factor = float(data.get('r_factor', 1.3))
        efficiency = float(data.get('efficiency', 0.70))
        
        # 2. Run Simulation
        compound_names = ['benzene', 'toluene', 'o-xylene']
        compounds = [Compound(name) for name in compound_names]
        thermo = ThermodynamicPackage(compounds)
        
        shortcut = ShortcutDistillation(thermo, F, z_F, P)
        
        results = shortcut.complete_shortcut_design(
            recovery_LK_D=recovery_LK,
            recovery_HK_B=recovery_HK,
            R_factor=R_factor,
            q=q,
            efficiency=efficiency
        )
        
        # 3. Generate Profiles for Plots
        stages, x_profiles, y_profiles, temperatures = get_composition_profiles(
            results, thermo, results['N_real'], results['feed_stage'], z_F
        )
        
        # 4. Prepare Data for Plotly
        
        # Plot 1: Composition Profiles
        graphs = []
        
        # Liquid Composition
        data_liq = []
        colors = ['#3b82f6', '#10b981', '#ef4444'] # Blue, Green, Red
        for i, name in enumerate(compound_names):
            data_liq.append(
                go.Scatter(
                    x=x_profiles[:, i].tolist(),
                    y=stages.tolist(),
                    mode='lines+markers',
                    name=f'{name} (Liq)',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6)
                )
            )
        
        layout_liq = go.Layout(
            title='Profil de Composition Liquide',
            xaxis=dict(title='Fraction Molaire (x)', range=[0, 1]),
            yaxis=dict(title='Numéro de Plateau', autorange='reversed'),
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        graphs.append(json.dumps(dict(data=data_liq, layout=layout_liq), cls=plotly.utils.PlotlyJSONEncoder))

        # Temperature Profile
        data_temp = [
            go.Scatter(
                x=(temperatures - 273.15).tolist(),
                y=stages.tolist(),
                mode='lines+markers',
                name='Température',
                line=dict(color='#f97316', width=3), # Orange
                marker=dict(size=8)
            )
        ]
        
        layout_temp = go.Layout(
            title='Profil de Température',
            xaxis=dict(title='Température (°C)'),
            yaxis=dict(title='Numéro de Plateau', autorange='reversed'),
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            shapes=[
                dict(
                    type="line",
                    x0=0, x1=1, xref="paper",
                    y0=results['feed_stage'], y1=results['feed_stage'],
                    line=dict(color="blue", width=2, dash="dash"),
                )
            ],
            annotations=[
                dict(
                    x=0.5, y=results['feed_stage'], xref="paper",
                    text=f"Plateau Alim: {results['feed_stage']}",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="blue")
                )
            ]
        )
        
        graphs.append(json.dumps(dict(data=data_temp, layout=layout_temp), cls=plotly.utils.PlotlyJSONEncoder))
        
        # Material Balance (Sankey-like or Bar)
        # Let's do a simple Bar chart for flows
        data_mb = [
            go.Bar(
                x=['Alimentation', 'Distillat', 'Résidu'],
                y=[F, results['D'], results['B']],
                marker=dict(color=['#3b82f6', '#10b981', '#ef4444'])
            )
        ]
        
        layout_mb = go.Layout(
            title='Bilan Matière (Débits)',
            yaxis=dict(title='Débit (kmol/h)'),
            template='plotly_white',
            height=400
        )
        graphs.append(json.dumps(dict(data=data_mb, layout=layout_mb), cls=plotly.utils.PlotlyJSONEncoder))

        # Convert numpy types to native python types for JSON serialization
        results_serializable = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_serializable[k] = v.tolist()
            elif isinstance(v, (np.float64, np.float32)):
                results_serializable[k] = float(v)
            elif isinstance(v, (np.int64, np.int32)):
                results_serializable[k] = int(v)
            else:
                results_serializable[k] = v

        return render_template('results.html', 
                             results=results_serializable, 
                             graphs=graphs,
                             compound_names=compound_names)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
