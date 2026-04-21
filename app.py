import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="ArboIA - Previsão Inteligente", layout="wide")

def realizar_predicao_completa(doenca_nome, data_alvo):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "modelos_salvos", f"modelo_{doenca_nome.upper()}.joblib")
    feat_path = os.path.join(base_path, "modelos_salvos", f"features_{doenca_nome.upper()}.joblib")
    data_path = os.path.join(base_path, "data", f"{doenca_nome.lower()}.csv")

    if not os.path.exists(model_path):
        return None, None, "Modelo não treinado."

    modelo = joblib.load(model_path)
    features_esperadas = joblib.load(feat_path)
    df = pd.read_csv(data_path)
    
    casos_col = 'casos' if 'casos' in df.columns else 'casos_est'
    date_col = next((c for c in df.columns if 'data' in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    ultima_data_real = df[date_col].max()
    data_alvo_dt = pd.to_datetime(data_alvo)

    projeções = []
    casos_buffer = df.tail(4)[casos_col].values.tolist()
    data_corrente = ultima_data_real

    if data_alvo_dt <= ultima_data_real:
        valor_existente = df[df[date_col] <= data_alvo_dt].tail(1)[casos_col].values[0]
        projeções.append({date_col: data_alvo_dt, 'Casos': valor_existente})
    else:
        while data_corrente < data_alvo_dt:
            data_corrente += timedelta(weeks=1)
            input_dict = {
                'semana': [data_corrente.isocalendar().week],
                'mes_sin': [np.sin(2 * np.pi * data_corrente.month/12)],
                'lag_1': [casos_buffer[-1]],
                'lag_2': [casos_buffer[-2]],
                'lag_4': [casos_buffer[0]],
                'media_4s': [np.mean(casos_buffer)]
            }
            if 'temp_shift' in features_esperadas or 'temp_lag' in features_esperadas:
                col = 'temp_shift' if 'temp_shift' in features_esperadas else 'temp_lag'
                input_dict[col] = [df['tempmed'].mean()]

            X_tmp = pd.DataFrame(input_dict)
            for f in features_esperadas:
                if f not in X_tmp.columns: X_tmp[f] = 0
            
            pred_semana = np.expm1(modelo.predict(X_tmp[features_esperadas]))[0]
            projeções.append({date_col: data_corrente, 'Casos': max(0, pred_semana)})
            casos_buffer.append(pred_semana)
            casos_buffer.pop(0)

    return df[[date_col, casos_col]], pd.DataFrame(projeções), None

st.title("🦟 Inteligência Epidemiológica")

with st.sidebar:
    st.header("Configurações")
    doenca_sel = st.selectbox("Doença", ["DENGUE", "ZIKA", "CHIKUNGUNYA"])
    # Sugere 2 meses a frente
    data_sel = st.date_input("Prever até:", datetime.now() + timedelta(weeks=8))
    btn = st.button("ANALISAR TENDÊNCIA", use_container_width=True)

if btn:
    df_real, df_previsto, erro = realizar_predicao_completa(doenca_sel, data_sel)
    
    if erro:
        st.error(erro)
    else:
        valor_final = df_previsto['Casos'].iloc[-1]
        st.metric(f"Previsão para {data_sel.strftime('%d/%m/%Y')}", f"{int(valor_final)} casos")

        fig = go.Figure()

        df_real_last = df_real.tail(20)
        fig.add_trace(go.Scatter(
            x=df_real_last.iloc[:,0], y=df_real_last.iloc[:,1],
            name='Histórico Real', mode='lines+markers',
            line=dict(color='#2c3e50', width=3)
        ))

        x_prev = [df_real_last.iloc[-1,0]] + df_previsto.iloc[:,0].tolist()
        y_prev = [df_real_last.iloc[-1,1]] + df_previsto.iloc[:,1].tolist()

        fig.add_trace(go.Scatter(
            x=x_prev, y=y_prev,
            name='Projeção IA', mode='lines',
            line=dict(color='#e74c3c', width=4, dash='dash')
        ))

        fig.update_layout(
            title=f"Evolução Epidemiológica - {doenca_sel}",
            xaxis_title="Data",
            yaxis_title="Número de Casos",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"A linha tracejada vermelha representa a projeção da IA baseada no comportamento atual da {doenca_sel}.")