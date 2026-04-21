import pandas as pd
import numpy as np
import glob
import os
import joblib
import time
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from colorama import init, Fore, Style

init(autoreset=True)

class ArboviroseEngine:
    def __init__(self):
        # Define os caminhos baseados na raiz do projeto
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.debug_dir = os.path.join(self.base_dir, "debug")
        self.model_dir = os.path.join(self.base_dir, "modelos_salvos")
        
        os.makedirs(self.model_dir, exist_ok=True)

    def preparar_dados(self, df):
        # Identificação de colunas
        casos_col = 'casos' if 'casos' in df.columns else 'casos_est'
        date_col = next((c for c in df.columns if 'data' in c.lower()), df.columns[0])
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        # Engenharia de Features
        df['semana'] = df[date_col].dt.isocalendar().week.astype(int)
        df['mes_sin'] = np.sin(2 * np.pi * df[date_col].dt.month/12)
        
        # Lags e Tendência
        for i in [1, 2, 4]:
            df[f'lag_{i}'] = df[casos_col].shift(i)
        
        df['media_movel_4s'] = df[casos_col].shift(1).rolling(window=4).mean()
        
        features = ['semana', 'mes_sin', 'lag_1', 'lag_2', 'lag_4', 'media_movel_4s']
        
        # Clima (se disponível)
        if 'tempmed' in df.columns:
            df['temp_shift'] = df['tempmed'].shift(3)
            features.append('temp_shift')

        df_clean = df.dropna(subset=features + [casos_col]).copy()
        return df_clean, features, date_col, casos_col

    def executar_ciclo_completo(self):
        arquivos_treino = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        if not arquivos_treino:
            print(f"{Fore.RED}❌ NENHUM ARQUIVO ENCONTRADO EM: {self.data_dir}")
            return

        print(f"{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.WHITE}   INICIANDO TREINAMENTO E VALIDAÇÃO DE DEBUG")
        print(f"{Fore.MAGENTA}{'='*60}\n")

        for path_treino in arquivos_treino:
            nome_arquivo = os.path.basename(path_treino)
            doenca = nome_arquivo.replace('.csv', '').upper()
            path_debug = os.path.join(self.debug_dir, nome_arquivo)

            print(f"{Fore.CYAN}🚀 Processando: {Fore.YELLOW}{doenca}")

            with tqdm(total=100, desc="Status", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
                # 1. Treinamento
                df_train_raw = pd.read_csv(path_treino)
                df_t, features, date_col, casos_col = self.preparar_dados(df_train_raw)
                pbar.update(30)

                modelo = XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=6, random_state=42)
                modelo.fit(df_t[features], np.log1p(df_t[casos_col]))
                pbar.update(40)

                # Salvar modelo
                joblib.dump(modelo, os.path.join(self.model_dir, f"modelo_{doenca}.joblib"))
                joblib.dump(features, os.path.join(self.model_dir, f"features_{doenca}.joblib"))
                pbar.update(10)

                # 2. Validação com dados de DEBUG
                if os.path.exists(path_debug):
                    df_debug_raw = pd.read_csv(path_debug)
                    df_d, _, _, _ = self.preparar_dados(df_debug_raw)
                    
                    preds_log = modelo.predict(df_d[features])
                    preds_final = np.expm1(preds_log)
                    
                    mae = mean_absolute_error(df_d[casos_col], preds_final)
                    pbar.set_postfix({"MAE": f"{mae:.2f}"})
                    pbar.update(20)
                    
                    print(f"{Fore.GREEN}✅ {doenca} Treinada. Erro Médio (MAE) no Debug: {mae:.2f}")
                else:
                    pbar.update(20)
                    print(f"{Fore.YELLOW}⚠️ Arquivo de debug não encontrado para {doenca}. Pulando validação.")

if __name__ == "__main__":
    engine = ArboviroseEngine()
    engine.executar_ciclo_completo()