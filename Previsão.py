import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# Ler dos dados que vão ser utilziados para treinar o modelo

df = pd.read_csv('vendas.csv', sep=';')
df['Data'] = pd.date_range(start="2022-12-06",end="2023-01-20")

# Ler dos dados onde serão armazenadas as previsões

prev = pd.read_csv('predict.csv')
prev['Data'] = pd.date_range(start="2023-01-21",end="2023-01-25")

# Criar uma coluna sequencial para refletir a pasagem do tempo

df['sequencia'] = 0
for n in range(46):
    df['sequencia'][n] = n
    
prev['sequencia'] = 0
x=0
for n in range(46, 51):
    prev['sequencia'][x] = n
    x+=1


# Ajustar o período atípico de entre 23/12 e 01/01.

df['Vendas_Ajust'] = df.Vendas

for n in range(0,7):
    df.at[17+n,'Vendas_Ajust'] = df.at[10+n,'Vendas']

for n in range(0,3):
    df.at[24+n,'Vendas_Ajust'] = df.at[10+n,'Vendas']

# Calcular os fatores sazonais para os dias da semana.

df['Fator'] = df['Vendas_Ajust'] / df['Vendas_Ajust'].mean()
df['dia_da_semana'] = df['Data'].dt.dayofweek
df_fatores = df.groupby(by='dia_da_semana').mean()
df = pd.merge(df, df_fatores['Fator'], how = 'outer', on = 'dia_da_semana')
df = df.sort_values(by=['Data'])

# Apurar as vendas sem a sazonalidade diária.

df['Vendas_sem_Saz'] = df['Vendas'] / df['Fator_y']

# Redimensionar arrays para compatibilidade com modelo
X = df['sequencia'].values.reshape(-1, 1)
y = df['Vendas'].values.reshape(-1, 1)
predict = prev['sequencia'].values.reshape(-1, 1)


# Estabelecer o modelo e colar os resultados na coluna de vendas do segundo DataFrame
model = LinearRegression()
model.fit(X, y)
prev['Vendas'] = model.predict(predict)

# Ajustar a sazonalidade diária
y=0
for i in range(-7, -3):
    prev['Vendas'][y] = round(prev['Vendas'][y] * df.iloc[i]['Fator_y'])
    y+=1


# Mostrar os resultados arredondando as vendas para o número inteiro mais proximo
print('             ***Projeção de Vendas para 5 dias ***' , end='\n\n')
print('Modelo: Regressão Linear, com ajuste para a sazonalidade diária.', end='\n\n', flush=True)
day = 21
for venda in prev.Vendas:
    venda = round(venda, 0)
    print("Projeção para " + str(day) + "/01/2023: " + str(venda))
    day+=1
