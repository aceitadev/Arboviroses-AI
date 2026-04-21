import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from colorama import init, Fore, Style

init(autoreset=True)

class ArbovirosePredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, "modelos_salvos")
        self.data_dir = os.path.join(self.base_dir, "data")

    def buscar_dados_recentes(self, doenca):
        """Carrega o CSV original para ter base para os lags."""
        path = os.path.join(self.data_dir, f"{doenca.lower()}.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    def prever(self, doenca, data_alvo):
        doenca = doenca.upper()
        modelo_path = os.path.join(self.model_dir, f"modelo_{doenca}.joblib")
        features_path = os.path.join(self.model_dir, f"features_{doenca}.joblib")

        if not os.path.exists(modelo_path):
            print(f"{Fore.RED}❌ Modelo para {doenca} não encontrado. Treine-o primeiro.")
            return

        # Carregar Inteligência
        modelo = joblib.load(modelo_path)
        features_salvas = joblib.load(features_path)
        
        # Carregar Histórico
        df = self.buscar_dados_recentes(doenca)
        date_col = next((c for c in df.columns if 'data' in c.lower()), df.columns[0])
        casos_col = 'casos' if 'casos' in df.columns else 'casos_est'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        # Preparar Input para a data desejada
        data_dt = pd.to_datetime(data_alvo)
        
        # Criar dicionário de features baseado no que o modelo espera
        input_data = {}
        
        # 1. Features Temporais
        input_data['semana'] = [data_dt.isocalendar().week]
        input_data['mes_sin'] = [np.sin(2 * np.pi * data_dt.month/12)]
        
        # 2. Features de Histórico (Lags)
        # Pegamos os últimos valores conhecidos antes da data alvo
        ultimos_casos = df[df[date_col] < data_dt].tail(4)[casos_col].values.tolist()
        
        if len(ultimos_casos) < 4:
            print(f"{Fore.YELLOW}⚠️ Histórico insuficiente para prever {data_alvo}")
            return

        input_data['lag_1'] = [ultimos_casos[-1]]
        input_data['lag_2'] = [ultimos_casos[-2]]
        input_data['lag_4'] = [ultimos_casos[0]]
        input_data['media_movel_4s'] = [np.mean(ultimos_casos)]

        # 3. Clima (opcional, se o modelo usar)
        if 'temp_shift' in features_salvas:
            temp_recente = df[df[date_col] < data_dt].tail(3)['tempmed'].iloc[0]
            input_data['temp_shift'] = [temp_recente]

        # Converter para DataFrame na ordem correta das features
        X_input = pd.DataFrame(input_data)[features_salvas]
        
        # Predição
        pred_log = modelo.predict(X_input)
        pred_final = np.expm1(pred_log)[0]

        return pred_final

def menu():
    predictor = ArbovirosePredictor()
    
    print(f"{Fore.GREEN}{'='*50}")
    print(f"{Fore.WHITE}   SISTEMA DE PREVISÃO DE ARBOVIROSES - FLN")
    print(f"{Fore.GREEN}{'='*50}")

    while True:
        print(f"\n{Fore.CYAN}Opções:")
        print("1. Prever Dengue")
        print("2. Prever Zika")
        print("3. Prever Chikungunya")
        print("0. Sair")
        
        opcao = input(f"\nEscolha uma opção: ")
        
        if opcao == '0': break
        
        doencas = {'1': 'DENGUE', '2': 'ZIKA', '3': 'CHIKUNGUNYA'}
        doenca_sel = doencas.get(opcao)
        
        if doenca_sel:
            data = input("Digite a data para previsão (AAAA-MM-DD): ")
            try:
                res = predictor.prever(doenca_sel, data)
                if res is not None:
                    cor = Fore.RED if res > 50 else Fore.YELLOW if res > 10 else Fore.GREEN
                    print(f"\n{Fore.WHITE}>>> Resultado para {doenca_sel}:")
                    print(f"{cor}{res:.2f} casos estimados.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Erro ao processar: {e}")
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    menu()