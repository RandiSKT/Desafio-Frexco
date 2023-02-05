import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Leitura dos dados que vão ser utilziados para treinar o modelo
df = pd.read_csv('vendas.csv', sep=';')
df['Data'] = pd.date_range(start="2022-12-06",end="2023-01-20")
# Leitura dos dados onde vão ser armazenadas as previsões
prev = pd.read_csv('predict.csv')
prev['Data'] = pd.date_range(start="2023-01-21",end="2023-01-25")


# ajustando o período atípico de entre 23/12 e 01/01.

df['Vendas_Ajust'] = df.Vendas

for n in range(0,7):
    df.at[17+n,'Vendas_Ajust'] = df.at[10+n,'Vendas']

for n in range(0,3):
    df.at[24+n,'Vendas_Ajust'] = df.at[10+n,'Vendas']

# Calculando os fatores sazonais para os dias da semana.

df['Fator'] = df['Vendas_Ajust'] / df['Vendas_Ajust'].mean()
df['dia_da_semana'] = df['Data'].dt.dayofweek
df_fatores = df.groupby(by='dia_da_semana').mean()
df = pd.merge(df, df_fatores['Fator'], how = 'outer', on = 'dia_da_semana')
df = df.sort_values(by=['Data'])

# Apurando as vendas sem a sazonalidade diária.

df['Vendas_sem_Saz'] = df['Vendas'] / df['Fator_y']

# Criando uma coluna compativel com o modelo e com o DataFrame de dados
prev['dia_da_semana'] = prev['Data'].dt.dayofweek

# Redimensionando arrays para compatibilidade com modelo
X = df['dia_da_semana'].values.reshape(-1, 1)
y = df['Vendas_sem_Saz'].values.reshape(-1, 1)
predict = prev['dia_da_semana'].values.reshape(-1, 1)

# Estabelecendo o modelo e colando os resultados na coluna de vendas do segundo DataFrame
model = LinearRegression()
model.fit(X, y)
prev['Vendas'] = model.predict(predict)

# Ajustando sazonalidade diária
x=0
for i in range(-7, -3):
    prev['Vendas'][x] = round(prev['Vendas'][x] * df.iloc[i]['Fator_y'])
    x+=1


# Mostrando os resultados arredondando as vendas para o número inteiro mais proximo
print('             ***Projeção de Vendas para 5 dias ***' , end='\n\n')
print('Modelo: Regressão Linear, com ajuste para a sazonalidade diária.', end='\n\n', flush=True)
day = 21
for venda in prev.Vendas:
    venda = round(venda, 0)
    print("Projeção para " + str(day) + "/01/2023: " + str(venda))
    day+=1
